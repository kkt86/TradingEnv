import ta

from util.errors import IndicatorsError
from util.logger import get_logger

LOGGER = get_logger(__file__.__name__)


def add_indicators(data):
    """todo: add docstring
    """
    assert 'open' in data.columns, "open column not present or with different name"
    assert 'high' in data.columns, "high column not present or with different name"
    assert 'low' in data.columns, "low column not present or with different name"
    assert 'close' in data.columns, "close column not present or with different name"

    try:
        data['RSI'] = ta.rsi(data["close"])
        data['TSI'] = ta.tsi(data["close"])
        data['UO'] = ta.uo(data["high"], data["low"], data["close"])
        data['AO'] = ta.ao(data["high"], data["low"])
        data['MACD_diff'] = ta.macd_diff(data["close"])
        data['Vortex_pos'] = ta.vortex_indicator_pos(data["high"], data["low"], data["close"])
        data['Vortex_neg'] = ta.vortex_indicator_neg(data["high"], data["low"], data["close"])
        data['Vortex_diff'] = abs(data['Vortex_pos'] - data['Vortex_neg'])
        data['Trix'] = ta.trix(data["close"])
        data['Mass_index'] = ta.mass_index(data["high"], data["low"])
        data['CCI'] = ta.cci(data["high"], data["low"], data["close"])
        data['DPO'] = ta.dpo(data["close"])
        data['KST'] = ta.kst(data["close"])
        data['KST_sig'] = ta.kst_sig(data["close"])
        data['KST_diff'] = (data['KST'] - data['KST_sig'])
        data['Aroon_up'] = ta.aroon_up(data["close"])
        data['Aroon_down'] = ta.aroon_down(data["close"])
        data['Aroon_ind'] = (data['Aroon_up'] - data['Aroon_down'])
        data['BBH'] = ta.bollinger_hband(data["close"])
        data['BBL'] = ta.bollinger_lband(data["close"])
        data['BBM'] = ta.bollinger_mavg(data["close"])
        data['BBHI'] = ta.bollinger_hband_indicator(data["close"])
        data['BBLI'] = ta.bollinger_lband_indicator(data["close"])
        data['KCHI'] = ta.keltner_channel_hband_indicator(data["high"], data["low"], data["close"])
        data['KCLI'] = ta.keltner_channel_lband_indicator(data["high"], data["low"], data["close"])
        data['DCHI'] = ta.donchian_channel_hband_indicator(data["close"])
        data['DCLI'] = ta.donchian_channel_lband_indicator(data["close"])
        data['DR'] = ta.daily_return(data["close"])
        data['DLR'] = ta.daily_log_return(data["close"])

        if 'volume' in data.columns:
            data['MFI'] = ta.money_flow_index(data["high"], data["low"], data["close"], data["volume"])
            data['ADI'] = ta.acc_dist_index(data["high"], data["low"], data["close"], data["volume"])
            data['OBV'] = ta.on_balance_volume(data["close"], data["volume"])
            data['CMF'] = ta.chaikin_money_flow(data["high"], data["low"], data["close"], data["volume"])
            data['FI'] = ta.force_index(data["close"], data["volume"])
            data['EM'] = ta.ease_of_movement(data["high"], data["low"], data["close"], data["volume"])
            data['VPT'] = ta.volume_price_trend(data["close"], data["volume"])
            data['NVI'] = ta.negative_volume_index(data["close"], data["volume"])

        data.fillna(method='bfill', inplace=True)

        return data

    except (AssertionError, Exception) as error:
        raise IndicatorsError(error)
        LOGGER.error(error)


