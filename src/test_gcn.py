import torch

from models.gcn import SimpleGCN


x = torch.randn(
    5,
    3
)

adj = torch.eye(
    5
)

model = SimpleGCN(
    input_dim=3,
    hidden_dim=16,
    output_dim=1
)

output = model(
    x,
    adj
)

print(
    "Output Shape:",
    output.shape
)