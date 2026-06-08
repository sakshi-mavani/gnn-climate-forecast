import pandas as pd
import torch

from sklearn.model_selection import train_test_split

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

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
).reshape(-1, 1)

print("Train X:", X_train.shape)
print("Train y:", y_train.shape)

model = SimpleLSTM()

output = model(
    X_train[:32]
)

print(
    "Model Output:",
    output.shape
)

import torch.nn as nn

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(10):

    optimizer.zero_grad()

    predictions = model(X_train)

    loss = criterion(
        predictions,
        y_train
    )

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1}/10 - Loss: {loss.item():.4f}"
    )
    
torch.save(
    model.state_dict(),
    "results/lstm_model.pth"
)

print("Model Saved!")