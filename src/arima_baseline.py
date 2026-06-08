import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


df = pd.read_csv(
    "data/raw/climate_data.csv"
)

series = df["temperature"]

train_size = int(
    len(series) * 0.8
)

train = series[:train_size]
test = series[train_size:]

model = ARIMA(
    train,
    order=(2, 0, 2)
)

model_fit = model.fit()

predictions = model_fit.forecast(
    steps=len(test)
)

rmse = np.sqrt(
    mean_squared_error(
        test,
        predictions
    )
)

print("ARIMA RMSE:", rmse)