import pandas as pd
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from create_sequences import create_sequences
from models.lstm import SimpleLSTM


df = pd.read_csv(
    "data/raw/climate_data.csv"
)

values = df[
    ["temperature",
     "humidity",
     "pressure"]
].values

X, y = create_sequences(
    values,
    seq_length=10
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

model = SimpleLSTM()

model.load_state_dict(
    torch.load(
        "results/lstm_model.pth"
    )
)

model.eval()

with torch.no_grad():

    predictions = model(
        X_test
    )

predictions = predictions.numpy()

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

print("RMSE:", rmse)

print("Predictions Shape:",
      predictions.shape)