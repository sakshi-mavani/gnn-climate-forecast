import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):

    def __init__(
        self,
        input_size=3,
        hidden_size=32,
        output_size=1
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        last_hidden = output[:, -1, :]

        prediction = self.fc(
            last_hidden
        )

        return prediction