import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error. Skips zeros in y_true."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_forecast(y_true, y_pred, model_name="model"):
    """Calculate all metrics and return as dict."""
    metrics = {
        "model": model_name,
        "RMSE": round(rmse(y_true, y_pred), 2),
        "MAE": round(mae(y_true, y_pred), 2),
        "MAPE": round(mape(y_true, y_pred), 2),
    }
    return metrics


def compare_models(results_list):
    """
    Takes a list of metric dicts and returns a comparison DataFrame.
    """
    df = pd.DataFrame(results_list)
    df = df.set_index("model")
    # sort by RMSE
    df = df.sort_values("RMSE")
    return df
