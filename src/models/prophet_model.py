import pandas as pd
import numpy as np
import logging
from prophet import Prophet

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


class ProphetForecaster:
    def __init__(self, yearly_seasonality=True, weekly_seasonality=False,
                 changepoint_prior_scale=0.05):
        self.yearly = yearly_seasonality
        self.weekly = weekly_seasonality
        self.cp_prior = changepoint_prior_scale
        self.model = None

    def fit(self, train_series):
        # prophet needs 'ds' and 'y' columns
        df = pd.DataFrame({
            "ds": train_series.index,
            "y": train_series.values
        })

        self.model = Prophet(
            yearly_seasonality=self.yearly,
            weekly_seasonality=self.weekly,
            daily_seasonality=False,
            changepoint_prior_scale=self.cp_prior,
        )
        self.model.add_country_holidays(country_name="US")
        self.model.fit(df)
        return self

    def predict(self, periods, freq="MS"):
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)

        pred = forecast.tail(periods)
        return pd.Series(pred["yhat"].values, index=pred["ds"].values)

    def get_components(self, periods, freq="MS"):
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        return self.model.predict(future)

    def plot(self):
        if self.model is None:
            raise ValueError("fit first")
        fig = self.model.plot(self.model.predict(
            self.model.make_future_dataframe(periods=12, freq="MS")
        ))
        return fig
