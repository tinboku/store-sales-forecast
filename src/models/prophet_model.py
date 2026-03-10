import pandas as pd
import numpy as np
from prophet import Prophet
import logging

# prophet is super noisy
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
        """
        Fit Prophet model. Expects a pandas Series with datetime index.
        """
        # prophet wants columns named 'ds' and 'y'
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

        # add US holidays
        self.model.add_country_holidays(country_name="US")

        self.model.fit(df)
        return self

    def predict(self, periods, freq="MS"):
        """Generate forecast for future periods."""
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)

        # only return the forecasted part
        pred = forecast.tail(periods)
        result = pd.Series(
            pred["yhat"].values,
            index=pred["ds"].values
        )
        return result

    def get_components(self, periods, freq="MS"):
        """Return full forecast df with components (trend, seasonality etc)."""
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        return forecast

    def plot(self):
        """Use Prophet's built-in plotting."""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        from prophet.plot import plot_plotly
        # just use matplotlib instead
        fig = self.model.plot(self.model.predict(
            self.model.make_future_dataframe(periods=12, freq="MS")
        ))
        return fig
