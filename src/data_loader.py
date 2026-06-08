import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_climate_data(file_path, feature_columns, target_column):

    df = pd.read_csv(file_path)

    print("Original Shape:", df.shape)

    df = df.dropna()

    scaler = StandardScaler()

    df[feature_columns] = scaler.fit_transform(
        df[feature_columns]
    )

    print("Processed Shape:", df.shape)

    return df


if __name__ == "__main__":

    feature_cols = [
        "temperature",
        "humidity",
        "pressure"
    ]

    target_col = "temperature"

    df = load_climate_data(
        "data/raw/climate_data.csv",
        feature_cols,
        target_col
    )

    print(df.head())