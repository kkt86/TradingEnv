import numpy as np
import pandas as pd


def difference(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    transformed_df = df.copy()
    for column in columns:
        transformed_df[column] = df[column] - df[column].shift(1)

    transformed_df = transformed_df.fillna(method='bfill')

    return transformed_df


def log_and_difference(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    transformed_df = df.copy()

    for column in columns:
        transformed_df.loc[df[column] == 0] = 1e-10
        transformed_df[column] = np.log(transformed_df[column]) - np.log(transformed_df[column]).shift(1)

    transformed_df = transformed_df.fillna(method='bfill')

    return transformed_df
