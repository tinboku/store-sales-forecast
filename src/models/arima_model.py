import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")


class SarimaForecaster:
    def __init__(self, order=None, seasonal_order=None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted = None

    def auto_fit(self, train_series, seasonal=True, m=12):
        """Use auto_arima to find best params then fit."""
        print("Running auto_arima... this may take a minute")
        stepwise_fit = auto_arima(
            train_series,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            seasonal=seasonal,
            m=m,
            d=None,  # let it figure out
            D=1,
            trace=True,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )
        self.order = stepwise_fit.order
        self.seasonal_order = stepwise_fit.seasonal_order
        print(f"Best ARIMA: {self.order} x {self.seasonal_order}")

        # refit with statsmodels for more control
        self.model = SARIMAX(
            train_series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.fitted = self.model.fit(disp=False)
        return self

    def fit(self, train_series):
        """Fit with pre-specified orders."""
        if self.order is None:
            return self.auto_fit(train_series)

        self.model = SARIMAX(
            train_series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.fitted = self.model.fit(disp=False)
        return self

    def predict(self, steps):
        """Forecast n steps ahead."""
        forecast = self.fitted.forecast(steps=steps)
        return forecast

    def get_summary(self):
        return self.fitted.summary()


def naive_forecast(train, test):
    """Naive baseline: last observed value repeated."""
    last_val = train.iloc[-1]
    pred = np.full(len(test), last_val)
    return pd.Series(pred, index=test.index)


def seasonal_naive(train, test, period=12):
    """Seasonal naive: use same month from last year."""
    preds = []
    for i in range(len(test)):
        # grab value from `period` steps back
        idx = len(train) - period + (i % period)
        preds.append(train.iloc[idx])
    return pd.Series(preds, index=test.index)
