import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_raw_data, aggregate_sales, get_monthly_sales, train_test_split_ts

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "train.csv")


@pytest.fixture
def raw_df():
    return load_raw_data(DATA_PATH)


@pytest.fixture
def monthly_sales(raw_df):
    return get_monthly_sales(raw_df)


def test_load_basic(raw_df):
    assert len(raw_df) > 0
    assert pd.api.types.is_datetime64_any_dtype(raw_df["Order Date"])


def test_no_duplicate_rows(raw_df):
    assert raw_df["Row ID"].is_unique


def test_sorted_by_date(raw_df):
    dates = raw_df["Order Date"].values
    assert (dates[1:] >= dates[:-1]).all()


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_raw_data("doesnt_exist.csv")


def test_monthly_agg(raw_df):
    result = aggregate_sales(raw_df, freq="ME")
    assert "date" in result.columns and "sales" in result.columns
    assert len(result) <= 48


def test_weekly_has_more_rows(raw_df):
    weekly = aggregate_sales(raw_df, freq="W")
    monthly = aggregate_sales(raw_df, freq="ME")
    assert len(weekly) > len(monthly)


def test_grouped_agg(raw_df):
    result = aggregate_sales(raw_df, freq="ME", group_cols=["Category"])
    assert len(result) > 48  # 3 categories


def test_train_test_split(monthly_sales):
    train, test = train_test_split_ts(monthly_sales, test_year=2018)
    assert train.index.max().year < 2018
    assert test.index.min().year >= 2018
    assert len(train) + len(test) == len(monthly_sales)
