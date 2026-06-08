import torch
import numpy as np

from models.gat import SimpleGAT

x = torch.randn(5, 3)

adj = torch.eye(5)

target = torch.randn(5, 1)

model = SimpleGAT(
    input_dim=3,
    hidden_dim=16,
    output_dim=1
)

model.load_state_dict(
    torch.load(
        "results/gat_model.pth"
    )
)

model.eval()

with torch.no_grad():
    predictions = model(
        x,
        adj
    )

rmse = np.sqrt(
    ((predictions.numpy() -
      target.numpy()) ** 2).mean()
)

print(
    "GAT RMSE:",
    rmse
)