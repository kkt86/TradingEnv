import ta
from enum import Enum


class SIGNAL(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


def trade_strategy(prices, initial_balance, commission, signal_fn):
    net_worth = [initial_balance]
    balance = initial_balance
    amounth_held = 0

    for i in range(1, len(prices)):
        if amounth_held > 0:
            net_worth.append(balance + amounth_held * prices[i])
        else:
            net_worth.append(balance)

        signal = signal_fn(i)

        if signal == SIGNAL.SELL and amounth_held > 0:
            balance = amounth_held * (prices[i] * (1 - commission))
            amounth_held = 0
        elif signal == SIGNAL.BUY and amounth_held == 0:
            amounth_held = balance / (prices[i] * (1 + commission))
            balance = 0

    return net_worth


def buy_and_hold(prices, initial_balance, commision):
    def signal_fn(i):
        return SIGNAL.BUY

    return trade_strategy(prices, initial_balance, commision, signal_fn)


def rsi_divergence(prices, initial_balance, commision, period=3):
    rsi = ta.rsi(prices)

    def signal_fn(i):
        if i >= period:
            rsiSum = sum(rsi[i - period:i + 1].diff().cumsum().fillna(0))
            priceSum = sum(prices[i - period:i + 1].diff().cumsum().fillna(0))

            if rsiSum < 0 and priceSum >= 0:
                return SIGNAL.SELL
            elif rsiSum > 0 and priceSum <= 0:
                return SIGNAL.BUY

        return SIGNAL.HOLD

    return trade_strategy(prices, initial_balance, commision, signal_fn)


def sma_crossover(prices, initial_balance, commision):
    macd = ta.macd(prices)

    def signal_fn(i):
        if macd[i] > 0 and macd[i-1] <= 0:
            return SIGNAL.SELL
        elif macd[i] < 0 and macd[i-1] >= 0:
            return SIGNAL.BUY
        return SIGNAL.HOLD

    return trade_strategy(prices, initial_balance, commision, signal_fn)