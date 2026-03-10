import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import yaml
import numpy as np


def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_figure(fig, name, dpi=150, output_dir="results/figures"):
    os.makedirs(output_dir, exist_ok=True)
    fpath = os.path.join(output_dir, name)
    fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {fpath}")


def plot_forecast(train, test, forecast, title="Forecast vs Actual", save_path=None):
    """Plot train, test, and forecasted values."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(train.index, train.values, label="Train", color="steelblue")
    ax.plot(test.index, test.values, label="Actual", color="black", linewidth=2)
    ax.plot(test.index, forecast, label="Forecast", color="tomato",
            linestyle="--", linewidth=2)

    ax.set_title(title)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_series(series, title="", ylabel="Sales ($)", save_path=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series.index, series.values, color="steelblue")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def set_seed(seed=42):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
