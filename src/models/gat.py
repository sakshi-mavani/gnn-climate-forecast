import torch
import torch.nn as nn


class SimpleGAT(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            output_dim
        )

    def forward(
        self,
        x,
        adj
    ):

        attention = torch.softmax(
            adj,
            dim=1
        )

        x = torch.matmul(
            attention,
            x
        )

        x = torch.relu(
            self.fc1(x)
        )

        x = self.fc2(x)

        return x