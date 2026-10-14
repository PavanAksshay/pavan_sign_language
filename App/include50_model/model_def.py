import torch
import torch.nn as nn
import torch.nn.functional as F

class INCLUDE50LSTM(nn.Module):
    def __init__(self, input_size=134, hidden_size=256, num_layers=5, bidirectional=True, dropout=0.2, n_classes=50):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )
        in_features = hidden_size * 2 if bidirectional else hidden_size
        self.l1 = nn.Linear(in_features=in_features, out_features=n_classes)

    def forward(self, x):
        # x: (batch_size, seq_len, 134)
        x, _ = self.lstm(x)
        x = torch.max(x, dim=1).values
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.l1(x)
        return x
