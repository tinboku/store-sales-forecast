# Store Sales Forecasting

> Originally done as a course project (Aug-Oct 2024). Cleaned up and uploaded to GitHub.

Forecasting monthly retail sales using SARIMA, Prophet and LSTM. Compares how statistical models and deep learning perform on a small dataset with strong seasonality.

Data: [Kaggle Superstore Sales](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting-dataset) (~10K orders, 2015-2018, 3 categories, 4 US regions).

## How to run

```bash
pip install -r requirements.txt
python run_experiment.py
```

Download the dataset from Kaggle and place it at `data/raw/train.csv`.

Notebooks can be explored independently:
- `01_eda.ipynb` — EDA and stationarity tests
- `02_baseline_models.ipynb` — SARIMA + Prophet
- `03_deep_learning.ipynb` — LSTM experiments

Run tests:
```bash
pytest tests/ -v
```

## Results

Train: 2015-2017 (36 months), Test: 2018 (12 months)

| Model | RMSE | MAE | MAPE |
|-------|------|-----|------|
| Seasonal Naive | 18,986 | 15,468 | 24.5% |
| SARIMA(1,1,0)(0,1,1,12) | 21,992 | 18,947 | 44.8% |
| Prophet | 26,629 | 19,075 | 30.9% |
| LSTM (h=64, seq=12) | 29,578 | 23,403 | 37.8% |

Seasonal Naive performs best here. With only 36 months of training data and a dominant yearly pattern, directly using last year's values is hard to beat. LSTM suffers from error accumulation in recursive multi-step prediction.

## Project structure

```
src/
  data_loader.py      # data loading and aggregation
  features.py         # lag, rolling, calendar features
  evaluate.py         # RMSE / MAE / MAPE
  utils.py            # plotting helpers
  models/             # SARIMA / Prophet / LSTM
notebooks/            # EDA + modeling experiments
tests/                # unit tests
configs/config.yaml   # hyperparameters
results/              # output figures (gitignored)
```

## Requirements

Python 3.9+. See `requirements.txt` for dependencies.
