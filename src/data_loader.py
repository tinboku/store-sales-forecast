import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(filepath="data/raw/train.csv"):
    """Load the superstore sales CSV and do basic cleaning."""
    df = pd.read_csv(filepath, encoding="latin-1")

    # parse dates - the format varies in some kaggle versions
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="mixed", dayfirst=False)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="mixed", dayfirst=False)

    # drop duplicates if any
    df = df.drop_duplicates(subset=["Row ID"])

    # sort by date
    df = df.sort_values("Order Date").reset_index(drop=True)
    return df


def aggregate_sales(df, freq="M", group_cols=None):
    """
    Aggregate sales to a given frequency.
    freq: 'D' for daily, 'W' for weekly, 'M' for monthly
    group_cols: optional list of columns to group by (e.g. ['Category'])
    """
    df = df.copy()
    df = df.set_index("Order Date")

    if group_cols:
        grouped = df.groupby(group_cols).resample(freq)["Sales"].sum()
        result = grouped.reset_index()
    else:
        result = df.resample(freq)["Sales"].sum().reset_index()
        result.columns = ["date", "sales"]

    return result


def get_monthly_sales(df):
    """Quick helper to get total monthly sales series."""
    monthly = aggregate_sales(df, freq="M")
    monthly = monthly.set_index("date")["sales"]
    monthly.index.freq = pd.infer_freq(monthly.index)
    return monthly


def train_test_split_ts(series, test_year=2018):
    """Split time series by year for train/test."""
    train = series[series.index.year < test_year]
    test = series[series.index.year >= test_year]
    return train, test


def get_category_monthly(df):
    """Get monthly sales broken down by category."""
    cat_monthly = aggregate_sales(df, freq="M", group_cols=["Category"])
    cat_monthly.columns = ["category", "date", "sales"]
    return cat_monthly


if __name__ == "__main__":
    # quick sanity check
    df = load_raw_data()
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['Order Date'].min()} to {df['Order Date'].max()}")
    monthly = get_monthly_sales(df)
    print(f"\nMonthly sales shape: {monthly.shape}")
    print(monthly.head())
