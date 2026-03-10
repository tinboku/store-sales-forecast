import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluate import rmse, mae, mape, evaluate_forecast, compare_models


class TestMetrics:
    def test_rmse_perfect(self):
        y = np.array([1, 2, 3])
        assert rmse(y, y) == 0.0

    def test_rmse_known(self):
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        result = rmse(y_true, y_pred)
        assert 0.6 < result < 0.7

    def test_mae_perfect(self):
        y = np.array([1, 2, 3])
        assert mae(y, y) == 0.0

    def test_mape_skips_zeros(self):
        y_true = np.array([0, 10, 20])
        y_pred = np.array([5, 10, 20])
        # should skip the zero entry
        assert mape(y_true, y_pred) == 0.0

    def test_mape_known_value(self):
        y_true = np.array([100, 200])
        y_pred = np.array([90, 210])
        # (10/100 + 10/200) / 2 * 100 = 7.5
        assert abs(mape(y_true, y_pred) - 7.5) < 0.01


class TestEvaluateForecast:
    def test_returns_dict(self):
        result = evaluate_forecast(
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            "test_model"
        )
        assert result["model"] == "test_model"
        assert result["RMSE"] == 0.0
        assert result["MAE"] == 0.0

    def test_compare_models(self):
        results = [
            {"model": "A", "RMSE": 10, "MAE": 8, "MAPE": 5},
            {"model": "B", "RMSE": 5, "MAE": 4, "MAPE": 3},
        ]
        df = compare_models(results)
        # should be sorted by RMSE
        assert df.index[0] == "B"
