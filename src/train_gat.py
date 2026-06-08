import torch
import torch.nn as nn
import torch.optim as optim

from models.gat import SimpleGAT


x = torch.randn(5, 3)

adj = torch.eye(5)

target = torch.randn(5, 1)

model = SimpleGAT(
    input_dim=3,
    hidden_dim=16,
    output_dim=1
)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)

for epoch in range(10):

    optimizer.zero_grad()

    output = model(
        x,
        adj
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
    "results/gat_model.pth"
)

print("GAT Saved!")