"""
Usage:
    python run_experiment.py
    python run_experiment.py --config configs/config.yaml
"""
import argparse
import os
import warnings
import sys

import numpy as np
import pandas as pd

from src.data_loader import load_raw_data, get_monthly_sales, train_test_split_ts
from src.models.arima_model import SarimaForecaster, naive_forecast, seasonal_naive
from src.models.prophet_model import ProphetForecaster
from src.models.lstm_model import LSTMForecaster
from src.evaluate import evaluate_forecast, compare_models
from src.utils import load_config, save_figure, plot_forecast, set_seed

warnings.filterwarnings("ignore")


def run(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(42)

    print("=" * 60)
    print("  Superstore Sales Forecasting Experiment")
    print("=" * 60)

    # load & prep data
    print("\n[1/5] Loading data...")
    raw = load_raw_data(cfg["data"]["raw_path"])
    monthly = get_monthly_sales(raw)
    print(f"  {len(raw)} records -> {len(monthly)} monthly points")
    print(f"  Range: {monthly.index[0].strftime('%Y-%m')} ~ {monthly.index[-1].strftime('%Y-%m')}")

    train, test = train_test_split_ts(monthly, test_year=cfg["data"]["test_year"])
    print(f"  Train: {len(train)}, Test: {len(test)}")

    results = []
    fig_dir = cfg["output"]["figures_dir"]
    os.makedirs(fig_dir, exist_ok=True)

    # baselines
    print("\n[2/5] Baselines...")
    naive_pred = naive_forecast(train, test)
    results.append(evaluate_forecast(test, naive_pred, "Naive"))

    snaive_pred = seasonal_naive(train, test)
    results.append(evaluate_forecast(test, snaive_pred, "Seasonal Naive"))
    print(f"  Naive RMSE: {results[0]['RMSE']}")
    print(f"  Seasonal Naive RMSE: {results[1]['RMSE']}")

    # SARIMA
    print("\n[3/5] SARIMA...")
    sarima = SarimaForecaster(
        order=cfg["sarima"].get("order"),
        seasonal_order=cfg["sarima"].get("seasonal_order"),
    )
    sarima.fit(train)
    sarima_pred = sarima.predict(steps=len(test))
    sarima_pred.index = test.index
    results.append(evaluate_forecast(test, sarima_pred, "SARIMA"))
    print(f"  SARIMA RMSE: {results[-1]['RMSE']}")

    fig = plot_forecast(train, test, sarima_pred, "SARIMA Forecast")
    save_figure(fig, "sarima_forecast.png", output_dir=fig_dir)

    # Prophet
    print("\n[4/5] Prophet...")
    pcfg = cfg["prophet"]
    prophet = ProphetForecaster(
        yearly_seasonality=pcfg["yearly_seasonality"],
        weekly_seasonality=pcfg["weekly_seasonality"],
        changepoint_prior_scale=pcfg["changepoint_prior_scale"],
    )
    prophet.fit(train)
    prophet_pred = prophet.predict(periods=len(test))
    prophet_pred.index = test.index
    results.append(evaluate_forecast(test, prophet_pred.values, "Prophet"))
    print(f"  Prophet RMSE: {results[-1]['RMSE']}")

    fig = plot_forecast(train, test, prophet_pred.values, "Prophet Forecast")
    save_figure(fig, "prophet_forecast.png", output_dir=fig_dir)

    # LSTM
    print("\n[5/5] LSTM...")
    lcfg = cfg["lstm"]
    lstm = LSTMForecaster(
        seq_len=lcfg["seq_len"],
        hidden_size=lcfg["hidden_size"],
        num_layers=lcfg["num_layers"],
        lr=lcfg["lr"],
        epochs=lcfg["epochs"],
        batch_size=lcfg["batch_size"],
    )
    lstm.fit(train)
    lstm_pred = lstm.predict(steps=len(test), test_index=test.index)
    results.append(evaluate_forecast(test, lstm_pred, "LSTM"))
    print(f"  LSTM RMSE: {results[-1]['RMSE']}")

    fig = plot_forecast(train, test, lstm_pred, "LSTM Forecast")
    save_figure(fig, "lstm_forecast.png", output_dir=fig_dir)

    fig = lstm.plot_loss()
    save_figure(fig, "lstm_loss.png", output_dir=fig_dir)

    # comparison
    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    comparison = compare_models(results)
    print(comparison.to_string())

    out_path = os.path.join("results", "model_comparison.csv")
    os.makedirs("results", exist_ok=True)
    comparison.to_csv(out_path)
    print(f"\nSaved to {out_path}")

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(args.config)
