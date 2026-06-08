import torch
import torch.nn as nn
import torch.optim as optim

from models.gcn_lstm import GCN_LSTM

x = torch.randn(5, 3)

edge_index = torch.tensor(
    [
        [0, 1, 2, 3, 4, 0, 2, 4],
        [1, 0, 3, 2, 0, 4, 4, 2]
    ],
    dtype=torch.long
)

target = torch.randn(1, 1)

model = GCN_LSTM(
    node_feature_dim=3,
    gcn_hidden_dim=16,
    lstm_hidden_dim=32,
    output_dim=1
)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(10):

    optimizer.zero_grad()

    output = model(
        x,
        edge_index
    )

    loss = criterion(
        output,
        target
    )

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1}/10 Loss: {loss.item():.4f}"
    )

torch.save(
    model.state_dict(),
    "results/gcn_lstm_model.pth"
)

print("GCN_LSTM Saved!")