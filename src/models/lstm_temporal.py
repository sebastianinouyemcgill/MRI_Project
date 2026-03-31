"""
LSTMTemporal

Input:
    features: torch.Tensor of shape (B, T, FEATURE_DIM)  # features from CNN for each timepoint
    hidden: optional hidden state tuple (h_0, c_0)

Output:
    y_pred: torch.Tensor of shape (B, 1)  # predicted progression probability
    hidden: optional hidden state tuple

Usage:
    lstm = LSTMTemporal()
    y_pred, hidden = lstm(features)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import FEATURE_DIM, BATCH_SIZE ,SEQ_LEN
import torch
import torch.nn as nn


class LSTMTemporal(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_layers=1,
        dropout=0.0
    ):
        super(LSTMTemporal, self).__init__()

        self.feature_dim = FEATURE_DIM
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Final classifier
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, features, hidden=None):
        """
        features: (B, T, FEATURE_DIM)
        hidden: optional (h_0, c_0)

        Returns:
            y_pred: (B, 1)
            hidden: (h_n, c_n)
        """

        lstm_out, hidden = self.lstm(features, hidden)
        # lstm_out: (B, T, hidden_dim)

        # Use last timestep
        last_out = lstm_out[:, -1, :]   # (B, hidden_dim)

        logits = self.fc(last_out)      # (B, 1)

        return logits, hidden   # note: return logits (better for BCEWithLogitsLoss)
    
if __name__ == "__main__":

    model = LSTMTemporal()
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, FEATURE_DIM)
    logits, hidden = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", logits.shape)  # should be (BATCH_SIZE, 1)