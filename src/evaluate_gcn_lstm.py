import torch
import numpy as np

from models.gcn_lstm import GCN_LSTM

x = torch.randn(5, 3)

edge_index = torch.tensor(
    [
        [0,1,2,3,4,0,2,4],
        [1,0,3,2,0,4,4,2]
    ],
    dtype=torch.long
)

target = torch.randn(1,1)

model = GCN_LSTM(
    node_feature_dim=3,
    gcn_hidden_dim=16,
    lstm_hidden_dim=32,
    output_dim=1
)

model.load_state_dict(
    torch.load(
        "results/gcn_lstm_model.pth"
    )
)

model.eval()

with torch.no_grad():

    prediction = model(
        x,
        edge_index
    )

rmse = np.sqrt(
    (
        (prediction.numpy() - target.numpy()) ** 2
    ).mean()
)

print("GCN_LSTM RMSE:", rmse)