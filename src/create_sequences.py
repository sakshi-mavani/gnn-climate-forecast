import pandas as pd
import numpy as np


def create_sequences(data, seq_length=10):

    X = []
    y = []

    for i in range(len(data) - seq_length):

        X.append(
            data[i:i + seq_length]
        )

        y.append(
            data[i + seq_length][0]
        )

    return (
        np.array(X),
        np.array(y)
    )


if __name__ == "__main__":

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

    print("X Shape:", X.shape)
    print("y Shape:", y.shape)