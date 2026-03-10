"""
Main entry point for running the sales forecasting experiment.

Usage:
    python run_experiment.py
    python run_experiment.py --config configs/config.yaml
"""

import argparse
import sys
import os
import warnings

import pandas as pd
import numpy as np

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

    # --- Load data ---
    print("\n[1/5] Loading data...")
    df = load_raw_data(cfg["data"]["raw_path"])
    monthly = get_monthly_sales(df)
    print(f"  Total records: {len(df)}")
    print(f"  Monthly series: {len(monthly)} points")
    print(f"  Date range: {monthly.index[0].strftime('%Y-%m')} to {monthly.index[-1].strftime('%Y-%m')}")

    train, test = train_test_split_ts(monthly, test_year=cfg["data"]["test_year"])
    print(f"  Train: {len(train)} months, Test: {len(test)} months")

    results = []
    os.makedirs(cfg["output"]["figures_dir"], exist_ok=True)

    # --- Naive baselines ---
    print("\n[2/5] Running baseline models...")
    naive_pred = naive_forecast(train, test)
    results.append(evaluate_forecast(test, naive_pred, "Naive"))

    snaive_pred = seasonal_naive(train, test)
    results.append(evaluate_forecast(test, snaive_pred, "Seasonal Naive"))
    print(f"  Naive RMSE: {results[0]['RMSE']}")
    print(f"  Seasonal Naive RMSE: {results[1]['RMSE']}")

    # --- SARIMA ---
    print("\n[3/5] Fitting SARIMA model...")
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
    save_figure(fig, "sarima_forecast.png", output_dir=cfg["output"]["figures_dir"])

    # --- Prophet ---
    print("\n[4/5] Fitting Prophet model...")
    prophet_cfg = cfg["prophet"]
    prophet = ProphetForecaster(
        yearly_seasonality=prophet_cfg["yearly_seasonality"],
        weekly_seasonality=prophet_cfg["weekly_seasonality"],
        changepoint_prior_scale=prophet_cfg["changepoint_prior_scale"],
    )
    prophet.fit(train)
    prophet_pred = prophet.predict(periods=len(test))
    prophet_pred.index = test.index
    results.append(evaluate_forecast(test, prophet_pred.values, "Prophet"))
    print(f"  Prophet RMSE: {results[-1]['RMSE']}")

    fig = plot_forecast(train, test, prophet_pred.values, "Prophet Forecast")
    save_figure(fig, "prophet_forecast.png", output_dir=cfg["output"]["figures_dir"])

    # --- LSTM ---
    print("\n[5/5] Training LSTM model...")
    lstm_cfg = cfg["lstm"]
    lstm = LSTMForecaster(
        seq_len=lstm_cfg["seq_len"],
        hidden_size=lstm_cfg["hidden_size"],
        num_layers=lstm_cfg["num_layers"],
        lr=lstm_cfg["lr"],
        epochs=lstm_cfg["epochs"],
        batch_size=lstm_cfg["batch_size"],
    )
    lstm.fit(train)
    lstm_pred = lstm.predict(steps=len(test), test_index=test.index)
    results.append(evaluate_forecast(test, lstm_pred, "LSTM"))
    print(f"  LSTM RMSE: {results[-1]['RMSE']}")

    fig = plot_forecast(train, test, lstm_pred, "LSTM Forecast")
    save_figure(fig, "lstm_forecast.png", output_dir=cfg["output"]["figures_dir"])

    # save training loss curve
    fig = lstm.plot_loss()
    save_figure(fig, "lstm_loss.png", output_dir=cfg["output"]["figures_dir"])

    # --- Results comparison ---
    print("\n" + "=" * 60)
    print("  Model Comparison")
    print("=" * 60)
    comparison = compare_models(results)
    print(comparison.to_string())

    # save results
    comparison.to_csv("results/model_comparison.csv")
    print(f"\nResults saved to results/model_comparison.csv")
    print("Figures saved to results/figures/")

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sales forecasting experiment")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    run(args.config)
