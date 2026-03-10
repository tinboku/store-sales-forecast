import pandas as pd
import numpy as np


def add_lag_features(df, col="sales", lags=[1, 2, 3, 6, 12]):
    """Add lagged values as features."""
    data = df.copy()
    for lag in lags:
        data[f"{col}_lag{lag}"] = data[col].shift(lag)
    return data


def add_rolling_features(df, col="sales", windows=[3, 6, 12]):
    """Rolling mean and std."""
    data = df.copy()
    for w in windows:
        data[f"{col}_rmean{w}"] = data[col].shift(1).rolling(window=w).mean()
        data[f"{col}_rstd{w}"] = data[col].shift(1).rolling(window=w).std()
    return data


def add_time_features(df, date_col="date"):
    """Extract calendar features from date column."""
    data = df.copy()

    if date_col in data.columns:
        dt = pd.to_datetime(data[date_col])
    else:
        # assume date is the index
        dt = data.index

    data["month"] = dt.month
    data["quarter"] = dt.quarter
    data["year"] = dt.year
    data["day_of_week"] = dt.dayofweek

    # simple holiday proxy: Nov and Dec
    data["is_holiday_season"] = dt.month.isin([11, 12]).astype(int)

    # back to school: Aug-Sep
    data["is_back_to_school"] = dt.month.isin([8, 9]).astype(int)

    return data


def add_diff_features(df, col="sales", periods=[1, 12]):
    """Add differenced features."""
    data = df.copy()
    for p in periods:
        data[f"{col}_diff{p}"] = data[col].diff(p)
    return data


def build_supervised_data(series, n_lags=12):
    """
    Convert a time series into supervised learning format.
    Returns X (lagged features) and y (target).
    """
    df = pd.DataFrame({"sales": series.values}, index=series.index)

    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = df["sales"].shift(i)

    df = df.dropna()

    X = df.drop("sales", axis=1)
    y = df["sales"]
    return X, y


def create_feature_matrix(monthly_sales):
    """
    Full feature engineering pipeline for monthly sales.
    Returns a DataFrame ready for modeling.
    """
    df = pd.DataFrame({"sales": monthly_sales})
    df["date"] = df.index

    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_time_features(df)
    df = add_diff_features(df)

    # drop rows with NaN from lagging
    df = df.dropna()

    return df
