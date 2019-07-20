from collections import namedtuple
from datetime import datetime
from typing import List, Tuple

import gym
import pandas as pd
import numpy as np
from pandas.api.types import is_float_dtype

from empyrical import sortino_ratio, calmar_ratio, omega_ratio
from gym import spaces
from sklearn.preprocessing import MinMaxScaler

from env.errors import ActionValueError, PositionError, TypePositionError, RenderModeError, DataShapeError
from env.viewer import TradingGraph
from util.logger import get_logger
from util.stationarization import log_and_difference

Account = namedtuple('Account', ['balance', 'bought', 'cost', 'sold', 'sales'])
Position = namedtuple('Position', ['type', 'shares', 'entry_price'])
Return = namedtuple('Return', ['total_return', 'return_per_asset'])


class TradingEnv(gym.Env):
    """Trading environment, wrapped around OpenAI gym"""

    def __init__(self, data: pd.DataFrame, feature_columns: List[str],
                 initial_balance: float = 10000, commission: float = 0.0025,
                 reward_function: str = 'sortino', returns_lookback: int = 100,
                 trade_on_open: bool = True) -> None:
        """
        :param data: pandas DataFrame, containing data for the simulation,
                     'open', 'high', 'low', 'close', 'volume' columns should be present
        :param feature_columns: initial trading balance
        :param initial_balance: initial trading balance
        :param commission: commission to be applied on trading
        :param reward_function: type of reward function, calmar, sortino and omega allowed
        :param returns_lookback: last values in portfolio to be used when computing the reward
        :param trade_on_open: Use next entry open price as price to open/close positions
        """
        super(TradingEnv, self).__init__()
        self._logger = get_logger(self.__class__.__name__)

        self._feature_cols = feature_columns

        self._check_initial_data(data, cols=self._feature_cols)

        self._data = data
        self._scaled_data = log_and_difference(data, feature_columns)
        self._initial_balance = initial_balance
        self._commission = commission
        self._reward_function = reward_function
        self._returns_lookback = returns_lookback
        self._trade_on_open = trade_on_open

        # state and action spaces
        self._obs_shape = (1, len(self._feature_cols))  # todo: add account info here
        self.observation_space = spaces.Box(low=0, high=1, shape=self._obs_shape, dtype=np.float32)
        self.action_space = spaces.Discrete(n=3)

        # placeholders
        self.current_step = 0
        self.cash = self._initial_balance
        self.position = None
        self.position_history = [None]
        self.portfolio = []

        self.viewer = None

    def reset(self) -> np.array:
        """
        Resets the environment
        """
        self.current_step = 0
        self.cash = self._initial_balance
        self.position = None
        self.position_history = [None]
        self.portfolio = [self._initial_balance]

        return self._next_observation()

    def step(self, action: int) -> Tuple[np.array, float, bool, dict]:
        """
        Perform interaction with the environment, for a given action
        :param action: action to be taken
        :return: tuple containing: next_state, reward from action, is done and empty info
        """
        self._take_action(action)
        self.current_step += 1
        self._update_portfolio()
        self.position_history.append(self.position)

        next_state = self._next_observation()
        reward = self._get_reward()
        done = self._is_done()

        return next_state, reward, done, {}

    def close(self) -> None:
        """Close viewer"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _take_action(self, action: int) -> None:
        """
        Actual method for performing an action:
            0: take no action
            1: if not in the market ==> go LONG
               if already SHORT in the market ==> close position
            2: if not in the market ==> go SHORT
               if already LONG in the market ==> close position
        Note that self.position is updated
        :param action: action to be taken
        """
        try:
            assert action in range(env.action_space.n), "action value: {} should be in {}".format(action, range(
                env.action_space.n))
            trade_price = self._get_trade_price()

            if self.position is None:
                if action == 1:
                    self._open_position(price=trade_price, position_type='LONG')
                elif action == 2:
                    self._open_position(price=trade_price, position_type='SHORT')
            else:
                if self.position.type == 'LONG' and action == 2:
                    self._close_position(price=trade_price)
                elif self.position.type == 'SHORT' and action == 1:
                    self._close_position(price=trade_price)
                else:
                    pass
        except AssertionError as error:
            raise ActionValueError(error)

    def _open_position(self, price, position_type) -> None:
        """
        This method opens position of type 'LONG' or 'SHORT'.
        If buys (or sells) only the number of shares, for which capital is available
        If LONG position is opened, money are removed from cash variable, and allocated
        to buy the necessary amount of shares. If SHORT position is opened, cash is not
        touched, but entry position is recorder.
        :param price: current price, at which shares should be bought (commission applies)
        :param position_type: type of position
        """
        if self.position:
            raise PositionError("Cannot open a second position")
        if position_type not in ['LONG', 'SHORT']:
            raise TypePositionError("Position type should be LONG OR SHORT, provided: {}".format(type))

        price_per_asset = price * (1 + self._commission)
        number_assets = int(self.cash / price_per_asset)
        if number_assets > 0:
            if position_type == 'LONG':
                self.cash -= number_assets * price_per_asset
                self.position = Position('LONG', number_assets, price_per_asset)
            else:
                self.position = Position('SHORT', number_assets, price_per_asset)

    def _close_position(self, price: float) -> None:
        """
        This method closes an open position.

        If current position is LONG, shares are sold, commission is applied,
        and the remaining cash is added to the available cash

        If current position is SHORT, the shorted number of shares are bought,
        commission is applied, and the difference between entry and exit prices
        are added to the available cash.

        In both cases, position is set to None
        :param price: price at which shares are bought/sold
        """
        if not self.position:
            raise PositionError("Cannot close nonexistent position")

        if self.position.type == 'LONG':
            amount_per_asset = price * (1 - self._commission)
            total_amount = self.position.shares * amount_per_asset
            self.cash += total_amount
            self.position = None
        else:
            amount_per_asset = price * (1 + self._commission)
            total_amount = self.position.shares * (self.position.entry_price - amount_per_asset)
            self.cash += total_amount
            self.position = None

    def _check_initial_data(self, data: pd.DataFrame, cols: List[str]) -> None:
        """
        Check quality of incoming data: for each column in cols, check if:
            1) column is present
            2) column type os float
            3) there are no missing values
        :param data: pandas DataFrame to be checked
        """
        try:
            for col in cols:
                assert col in data.columns, "{} column not present in data".format(col)
                assert is_float_dtype(data[col]), "{} not of type float".format(col)
                assert not data[col].isnull().values.any(), "Missing values in column: {}".format(col)

        except AssertionError as error:
            self._logger.error(error)

    def _next_observation(self) -> np.array:
        """
        This method returns the next observation by getting the current features, and scaling them
        to [0, 1]
        :return: next observation scaled
        """
        scaler = MinMaxScaler()

        # scale features and add to state the most recent one
        features = self._scaled_data[self._feature_cols]

        scaled = features[:self.current_step + 1].values
        scaled[abs(scaled) == np.inf] = 0
        scaled = scaler.fit_transform(scaled.astype('float32'))
        scaled = pd.DataFrame(scaled, columns=features.columns)

        obs = scaled.values[-1]

        # todo: add account information here

        obs = np.reshape(obs.astype('float32'), self._obs_shape)
        obs[np.bitwise_not(np.isfinite(obs))] = 0

        return obs

    def _get_trade_price(self) -> np.float32:
        """
        This method returns the price, at which assets should be traded. Depending on the initial configuration,
        it either returns the close price of the current bar (if trade_on_open is False) or the open price of
        the next bar (if trade_on_open is True). In this way, a better market simulation is obtained.
        :return: price at which assets should be traded
        """
        if self.current_step == len(self._data) and self._trade_on_open:
            self._logger.error("Last data entry reached")
            raise DataShapeError("Last data entry reached")

        return self._data['open'].values[self.current_step + 1] if self._trade_on_open else \
            self._data['close'].values[self.current_step]

    def _get_reward(self) -> float:
        """
        This method computes the reward from each action, by looking at the annualized
        ratio, provided in the reward_function
        :return: annualized value of the selected reward ratio
        """
        lookback = min(self.current_step, self._returns_lookback)
        returns = np.diff(self.portfolio[-lookback:])

        if np.count_nonzero(returns) < 1:
            return 0

        if np.count_nonzero(returns) < 1:
            return 0

        if self._reward_function == 'sortino':
            reward = sortino_ratio(returns, annualization=365 * 24)
        elif self._reward_function == 'calmar':
            reward = calmar_ratio(returns, annualization=365 * 24)
        elif self._reward_function == 'omega':
            reward = omega_ratio(returns, annualization=365 * 24)
        else:
            reward = returns[-1]

        return reward if np.isfinite(reward) else 0

    def _is_done(self) -> bool:
        """
        This method returns True, if the current portfolio value drops less then 10% of the initial
        investment, or if arrive at the last entry of the available data
        :return: boolean value, indication if the simulation is done
        """
        # return True, if current portfolio value drops less then 10% of the initial valueTrue
        if self.portfolio[-1] < float(self._initial_balance) / 10.:
            return True

        # return True, if arrived at last entry of data
        if self.current_step == len(self._data) - 1 - int(self._trade_on_open):
            return True

        return False

    def _update_portfolio(self) -> None:
        """
        This method appends the current value of the portfolio by considering the available cash,
        plus the value of all open positions

        The updated vale of the portfolio us appended the the portfolio array
        """
        current_price = self._get_trade_price()
        value = self.cash
        if self.position is not None:
            if self.position.type == 'LONG':
                value += self.position.shares * current_price
            else:
                value += self.position.shares * (self.position.entry_price - current_price)
        self.portfolio.append(value)

    def render(self, mode: str = 'system') -> None:
        """
        Renders a step of the simulation.
        If mode == 'system', logs are printed in Terminal.
        If mode == 'human', visualization is displayed.
        :param mode: type of rendering mode (system or human)
        """
        if mode not in ['system', 'human']:
            raise RenderModeError("Render mode should be either 'system' or 'human', provided: {}".format(mode))

        if mode == 'system':
            step = self.current_step
            price = self._get_trade_price()
            position_type = 'None' if not self.position else self.position.type
            shares = 0 if not self.position else self.position.shares
            entry_price = 0 if not self.position else self.position.entry_price
            portfolio = self.portfolio[-1]
            pnl = portfolio - self._initial_balance
            pnl_prc = 100 * float(portfolio - self._initial_balance) / self._initial_balance

            print(f"Step: {step:5d} | Price: {price:4.5f} | Portfolio: {portfolio:5.2f} | "
                  f"PnL: {pnl:5.2f} | PnL (prc) : {pnl_prc:3.2f} | Position: {position_type:5s} | "
                  f"Shares: {shares:4d} | EntryPrice: {entry_price:4.5f}")

        elif mode == 'human':
            if self.viewer is None:
                self.viewer = TradingGraph(self._data)

            self.viewer.render(self.current_step, portfolio=self.portfolio, positions=self.position_history)


if __name__ == '__main__':
    df = pd.read_csv('../data/coinbase_hourly.csv')
    df['open'] = df['Open']
    df['high'] = df['High']
    df['low'] = df['Low']
    df['close'] = df['Close']
    df['volume'] = df['Volume BTC']
    df['date'] = df['Date']
    df['time'] = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %I-%p'))

    feature_cols = ['open', 'high', 'low', 'close', 'volume']

    env = TradingEnv(df, feature_cols, initial_balance=10000, )
    env.reset()

    for _ in range(0, len(df) - 1):
        action = np.random.random_integers(0, 2)
        next_state, reward, done, _ = env.step(action)
        env.render(mode='human')

        if done:
            break
