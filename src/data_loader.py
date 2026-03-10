import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(filepath="data/raw/train.csv"):
    df = pd.read_csv(filepath, encoding="latin-1")

    df["Order Date"] = pd.to_datetime(df["Order Date"], format="mixed", dayfirst=False)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="mixed", dayfirst=False)

    df = df.drop_duplicates(subset=["Row ID"])
    df = df.dropna(subset=["Order Date", "Sales"])
    df = df.sort_values("Order Date").reset_index(drop=True)
    return df


def aggregate_sales(df, freq="ME", group_cols=None):
    """Aggregate order-level sales to a given time frequency."""
    data = df.copy()
    data = data.set_index("Order Date")

    if group_cols:
        grouped = data.groupby(group_cols).resample(freq)["Sales"].sum()
        result = grouped.reset_index()
    else:
        result = data.resample(freq)["Sales"].sum().reset_index()
        result.columns = ["date", "sales"]

    return result


def get_monthly_sales(df):
    monthly = aggregate_sales(df, freq="ME")
    monthly = monthly.set_index("date")["sales"]
    monthly.index.freq = pd.infer_freq(monthly.index)
    return monthly


def train_test_split_ts(series, test_year=2018):
    train = series[series.index.year < test_year]
    test = series[series.index.year >= test_year]
    return train, test


def get_category_monthly(df):
    cat_monthly = aggregate_sales(df, freq="ME", group_cols=["Category"])
    cat_monthly.columns = ["category", "date", "sales"]
    return cat_monthly


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['Order Date'].min()} to {df['Order Date'].max()}")
    monthly = get_monthly_sales(df)
    print(f"\nMonthly sales shape: {monthly.shape}")
    # print(monthly.describe())
    print(monthly.head())
