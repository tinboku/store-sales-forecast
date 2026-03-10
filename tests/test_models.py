import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.arima_model import naive_forecast, seasonal_naive


def _monthly(n=36, start="2015-01-01"):
    idx = pd.date_range(start, periods=n, freq="MS")
    return pd.Series(np.arange(n, dtype=float) * 100 + 1000, index=idx)


def test_naive_repeats_last():
    train = _monthly(36)
    test = _monthly(12, "2018-01-01")
    pred = naive_forecast(train, test)
    assert len(pred) == 12
    assert (pred == train.iloc[-1]).all()


def test_snaive_uses_last_year():
    train = _monthly(36)
    test = _monthly(12, "2018-01-01")
    pred = seasonal_naive(train, test, period=12)
    np.testing.assert_array_equal(pred.values, train.iloc[-12:].values)


def test_snaive_short_data_raises():
    train = _monthly(6)
    test = _monthly(3, "2015-07-01")
    with pytest.raises(ValueError):
        seasonal_naive(train, test, period=12)
