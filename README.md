# Store Sales Forecasting

Time series forecasting on the Superstore Sales dataset. Compares statistical models (SARIMA, Prophet) with deep learning (LSTM) for predicting monthly retail sales.

## Dataset

[Kaggle Superstore Sales](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting-dataset) — ~9,800 retail orders from 2015-2018 across 3 product categories and 4 US regions.

## Project Structure

```
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory analysis
│   ├── 02_baseline_models.ipynb  # SARIMA & Prophet
│   └── 03_deep_learning.ipynb    # LSTM experiments
├── src/
│   ├── data_loader.py            # Data loading & aggregation
│   ├── features.py               # Feature engineering
│   ├── evaluate.py               # Metrics (RMSE, MAE, MAPE)
│   ├── utils.py                  # Plotting & config helpers
│   └── models/
│       ├── arima_model.py
│       ├── prophet_model.py
│       └── lstm_model.py
├── configs/config.yaml           # Hyperparameters
├── run_experiment.py             # End-to-end pipeline
└── results/                      # Output figures & metrics
```

## Setup

```bash
pip install -r requirements.txt
```

Put the dataset CSV in `data/raw/train.csv`.

## Usage

Run the full pipeline:

```bash
python run_experiment.py
```

Or explore step by step in the notebooks.

## Models

| Model | Description |
|-------|-------------|
| Naive | Repeat last observed value |
| Seasonal Naive | Repeat same month from last year |
| SARIMA | Auto-fitted with pmdarima |
| Prophet | With US holiday effects |
| LSTM | 2-layer LSTM, recursive multi-step forecast |

Train period: 2015-2017, Test period: 2018.

## Key Findings

- Clear upward trend with strong yearly seasonality (peaks in Sep, Nov-Dec)
- Statistical models (SARIMA, Prophet) perform well on this small dataset
- LSTM struggles with only 36 training data points — not enough for deep learning to shine
- Seasonal naive is a surprisingly strong baseline when seasonality dominates

## Requirements

- Python 3.8+
- PyTorch, Prophet, statsmodels, pmdarima
- See `requirements.txt` for full list
