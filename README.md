# Store Sales Forecasting

Forecasting monthly retail sales using statistical and deep learning approaches. Built on the [Kaggle Superstore Sales](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting-dataset) dataset (~10K orders, 2015-2018).

The goal is to compare how classical time series models (SARIMA, Prophet) stack up against neural approaches (LSTM) on a relatively small dataset with strong seasonality.

## Dataset

- ~10,000 retail orders across Furniture, Office Supplies, and Technology
- 4 US regions (West, East, Central, South)
- Time span: Jan 2015 – Dec 2018
- Aggregated to **monthly total sales** for forecasting

## Project Structure

```
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory analysis & stationarity tests
│   ├── 02_baseline_models.ipynb  # SARIMA & Prophet
│   └── 03_deep_learning.ipynb    # LSTM experiments
├── src/
│   ├── data_loader.py            # Data loading & aggregation
│   ├── features.py               # Feature engineering (lags, rolling stats)
│   ├── evaluate.py               # Metrics (RMSE, MAE, MAPE)
│   ├── utils.py                  # Plotting & config helpers
│   └── models/
│       ├── arima_model.py        # SARIMA + naive baselines
│       ├── prophet_model.py      # Facebook Prophet wrapper
│       └── lstm_model.py         # PyTorch LSTM
├── tests/                        # Unit tests
├── configs/config.yaml           # Hyperparameters
├── run_experiment.py             # End-to-end pipeline
└── results/                      # Generated figures & metrics (gitignored)
```

## Setup

```bash
git clone https://github.com/<your-username>/store-sales-forecast.git
cd store-sales-forecast
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting-dataset) and place it at `data/raw/train.csv`.

## Quick Start

```bash
# run full experiment pipeline
python run_experiment.py

# or with custom config
python run_experiment.py --config configs/config.yaml

# run tests
pytest tests/ -v
```

## Results

Train: 2015-2017 (36 months) | Test: 2018 (12 months)

| Model | RMSE | MAE | MAPE (%) |
|-------|------|-----|----------|
| Seasonal Naive | 18,986 | 15,468 | 24.5 |
| SARIMA(1,1,0)(0,1,1,12) | 21,992 | 18,947 | 44.8 |
| Prophet | 26,629 | 19,075 | 30.9 |
| LSTM (h=64, seq=12) | 29,578 | 23,403 | 37.8 |

**Seasonal Naive wins.** With only 36 months of training data and a dominant yearly pattern, simple methods that directly leverage the seasonal structure outperform more complex models. SARIMA captures the trend reasonably well but overshoots on some months. LSTM suffers from error accumulation in recursive multi-step prediction.

### Forecast Comparison

The `results/figures/` directory contains forecast plots for each model after running the pipeline.

## Key Takeaways

- Strong yearly seasonality (peaks in Sep, Nov-Dec) dominates the signal
- First differencing achieves stationarity (ADF test confirms d=1 is sufficient)
- On small datasets, statistical models have a clear edge over deep learning
- LSTM's recursive forecasting accumulates errors — teacher forcing or direct multi-output would help but needs more data

## Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list
