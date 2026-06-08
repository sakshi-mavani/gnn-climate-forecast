import torch

from models.gcn_lstm import GCN_LSTM

x = torch.randn(
    5,
    3
)

edge_index = torch.tensor(
    [
        [0, 1, 2, 3, 4, 0, 2, 4],
        [1, 0, 3, 2, 0, 4, 4, 2]
    ],
    dtype=torch.long
)

model = GCN_LSTM(
    node_feature_dim=3,
    gcn_hidden_dim=16,
    lstm_hidden_dim=32,
    output_dim=1
)

output = model(
    x,
    edge_index
)

print(
    "Output Shape:",
    output.shape
)