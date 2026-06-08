import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCN_LSTM(nn.Module):

    def __init__(
        self,
        node_feature_dim,
        gcn_hidden_dim,
        lstm_hidden_dim,
        output_dim
    ):
        super().__init__()

        self.gcn = GCNConv(
            node_feature_dim,
            gcn_hidden_dim
        )

        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=gcn_hidden_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(
            lstm_hidden_dim,
            output_dim
        )

    def forward(self, x, edge_index):

        x = self.gcn(
            x,
            edge_index
        )

        x = self.relu(x)

        x = x.unsqueeze(0)

        out, _ = self.lstm(x)

        last = out[:, -1, :]

        prediction = self.fc(last)

        return prediction