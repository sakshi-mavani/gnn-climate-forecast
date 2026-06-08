import torch

from models.lstm import SimpleLSTM


model = SimpleLSTM()

dummy_input = torch.randn(
    32,
    10,
    3
)

output = model(
    dummy_input
)

print(
    "Output Shape:",
    output.shape
)