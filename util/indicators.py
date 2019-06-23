import ta


def add_indicators(df):
    df['RSI'] = ta.rsi(df["close"])
    df['MFI'] = ta.money_flow_index(
        df["high"], df["low"], df["close"], df["volume"])
    df['TSI'] = ta.tsi(df["close"])
    df['UO'] = ta.uo(df["high"], df["low"], df["close"])
    df['AO'] = ta.ao(df["high"], df["low"])

    df['MACD_diff'] = ta.macd_diff(df["close"])
    df['Vortex_pos'] = ta.vortex_indicator_pos(
        df["high"], df["low"], df["close"])
    df['Vortex_neg'] = ta.vortex_indicator_neg(
        df["high"], df["low"], df["close"])
    df['Vortex_diff'] = abs(
        df['Vortex_pos'] -
        df['Vortex_neg'])
    df['Trix'] = ta.trix(df["close"])
    df['Mass_index'] = ta.mass_index(df["high"], df["low"])
    df['CCI'] = ta.cci(df["high"], df["low"], df["close"])
    df['DPO'] = ta.dpo(df["close"])
    df['KST'] = ta.kst(df["close"])
    df['KST_sig'] = ta.kst_sig(df["close"])
    df['KST_diff'] = (
            df['KST'] -
            df['KST_sig'])
    df['Aroon_up'] = ta.aroon_up(df["close"])
    df['Aroon_down'] = ta.aroon_down(df["close"])
    df['Aroon_ind'] = (
            df['Aroon_up'] -
            df['Aroon_down']
    )

    df['BBH'] = ta.bollinger_hband(df["close"])
    df['BBL'] = ta.bollinger_lband(df["close"])
    df['BBM'] = ta.bollinger_mavg(df["close"])
    df['BBHI'] = ta.bollinger_hband_indicator(
        df["close"])
    df['BBLI'] = ta.bollinger_lband_indicator(
        df["close"])
    df['KCHI'] = ta.keltner_channel_hband_indicator(df["high"],
                                                    df["low"],
                                                    df["close"])
    df['KCLI'] = ta.keltner_channel_lband_indicator(df["high"],
                                                    df["low"],
                                                    df["close"])
    df['DCHI'] = ta.donchian_channel_hband_indicator(df["close"])
    df['DCLI'] = ta.donchian_channel_lband_indicator(df["close"])

    df['ADI'] = ta.acc_dist_index(df["high"],
                                  df["low"],
                                  df["close"],
                                  df["Volume BTC"])
    df['OBV'] = ta.on_balance_volume(df["close"],
                                     df["Volume BTC"])
    df['CMF'] = ta.chaikin_money_flow(df["high"],
                                      df["low"],
                                      df["close"],
                                      df["volume"])
    df['FI'] = ta.force_index(df["close"],
                              df["volume"])
    df['EM'] = ta.ease_of_movement(df["high"],
                                   df["low"],
                                   df["close"],
                                   df["volume"])
    df['VPT'] = ta.volume_price_trend(df["close"],
                                      df["volume"])
    df['NVI'] = ta.negative_volume_index(df["close"],
                                         df["volume"])

    df['DR'] = ta.daily_return(df["close"])
    df['DLR'] = ta.daily_log_return(df["close"])

    df.fillna(method='bfill', inplace=True)

    return df
