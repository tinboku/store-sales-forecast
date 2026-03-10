import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_raw_data, aggregate_sales, get_monthly_sales, train_test_split_ts


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "train.csv")


@pytest.fixture
def raw_df():
    return load_raw_data(DATA_PATH)


@pytest.fixture
def monthly_sales(raw_df):
    return get_monthly_sales(raw_df)


class TestLoadRawData:
    def test_loads_without_error(self, raw_df):
        assert len(raw_df) > 0

    def test_date_columns_parsed(self, raw_df):
        assert pd.api.types.is_datetime64_any_dtype(raw_df["Order Date"])
        assert pd.api.types.is_datetime64_any_dtype(raw_df["Ship Date"])

    def test_no_duplicate_row_ids(self, raw_df):
        assert raw_df["Row ID"].is_unique

    def test_sorted_by_date(self, raw_df):
        dates = raw_df["Order Date"].values
        assert (dates[1:] >= dates[:-1]).all()

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_raw_data("nonexistent.csv")


class TestAggregateSales:
    def test_monthly_aggregation(self, raw_df):
        result = aggregate_sales(raw_df, freq="M")
        assert "date" in result.columns
        assert "sales" in result.columns
        assert len(result) <= 48  # 4 years max

    def test_weekly_aggregation(self, raw_df):
        result = aggregate_sales(raw_df, freq="W")
        assert len(result) > 48  # more than monthly

    def test_grouped_aggregation(self, raw_df):
        result = aggregate_sales(raw_df, freq="M", group_cols=["Category"])
        assert len(result) > 48  # multiple categories


class TestTrainTestSplit:
    def test_split_year(self, monthly_sales):
        train, test = train_test_split_ts(monthly_sales, test_year=2018)
        assert train.index.max().year < 2018
        assert test.index.min().year >= 2018

    def test_no_data_loss(self, monthly_sales):
        train, test = train_test_split_ts(monthly_sales, test_year=2018)
        assert len(train) + len(test) == len(monthly_sales)
