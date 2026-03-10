import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.arima_model import naive_forecast, seasonal_naive


def _make_monthly_series(n=36, start="2015-01-01"):
    """Helper to create a simple monthly series for testing."""
    idx = pd.date_range(start, periods=n, freq="MS")
    values = np.arange(n, dtype=float) * 100 + 1000
    return pd.Series(values, index=idx)


class TestNaiveForecast:
    def test_output_length(self):
        train = _make_monthly_series(36)
        test = _make_monthly_series(12, start="2018-01-01")
        pred = naive_forecast(train, test)
        assert len(pred) == len(test)

    def test_all_same_value(self):
        train = _make_monthly_series(36)
        test = _make_monthly_series(12, start="2018-01-01")
        pred = naive_forecast(train, test)
        assert (pred == train.iloc[-1]).all()


class TestSeasonalNaive:
    def test_output_length(self):
        train = _make_monthly_series(36)
        test = _make_monthly_series(12, start="2018-01-01")
        pred = seasonal_naive(train, test, period=12)
        assert len(pred) == 12

    def test_uses_last_year(self):
        train = _make_monthly_series(36)
        test = _make_monthly_series(12, start="2018-01-01")
        pred = seasonal_naive(train, test, period=12)
        # should equal months 25-36 of training data (last year)
        expected = train.iloc[-12:].values
        np.testing.assert_array_equal(pred.values, expected)

    def test_raises_on_short_data(self):
        train = _make_monthly_series(6)
        test = _make_monthly_series(3, start="2015-07-01")
        with pytest.raises(ValueError):
            seasonal_naive(train, test, period=12)
