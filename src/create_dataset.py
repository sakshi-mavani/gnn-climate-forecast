import pandas as pd
import numpy as np

np.random.seed(42)

rows = 1000

df = pd.DataFrame({
    "temperature": np.random.normal(30, 5, rows),
    "humidity": np.random.normal(65, 10, rows),
    "pressure": np.random.normal(1010, 5, rows)
})

df.to_csv(
    "data/raw/climate_data.csv",
    index=False
)

print("Dataset Created")
print(df.head())