import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluate import rmse, mae, mape, evaluate_forecast, compare_models


def test_rmse_zero():
    y = np.array([1, 2, 3])
    assert rmse(y, y) == 0.0


def test_rmse_value():
    assert 0.6 < rmse([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]) < 0.7


def test_mae_zero():
    assert mae([1, 2], [1, 2]) == 0.0


def test_mape_skips_zeros():
    # zero entries should be ignored
    assert mape([0, 10, 20], [5, 10, 20]) == 0.0


def test_mape_value():
    # (10/100 + 10/200) / 2 * 100 = 7.5
    assert abs(mape([100, 200], [90, 210]) - 7.5) < 0.01


def test_evaluate_returns_dict():
    r = evaluate_forecast([1, 2, 3], [1, 2, 3], "dummy")
    assert r["model"] == "dummy"
    assert r["RMSE"] == 0.0


def test_compare_sorted_by_rmse():
    res = [
        {"model": "bad", "RMSE": 10, "MAE": 8, "MAPE": 5},
        {"model": "good", "RMSE": 5, "MAE": 4, "MAPE": 3},
    ]
    df = compare_models(res)
    assert df.index[0] == "good"
