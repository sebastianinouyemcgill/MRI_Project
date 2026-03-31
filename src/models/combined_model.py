"""
combined_model

Input:
    x: torch.Tensor of shape (B, T, 1, 128, 128, 128)  # raw POST scans
    hidden: optional hidden state tuple for LSTM

Pipeline:
    1. Flatten time: (B*T, 1, 128, 128, 128)
    2. CNNEncoder → (B*T, FEATURE_DIM)
    3. Reshape: (B, T, FEATURE_DIM)
    4. LSTMTemporal → (B, 1)

Output:
    y_pred: torch.Tensor of shape (B, 1)  # predicted progression probability
    hidden: optional hidden state tuple

Usage:
    model = combined_model()
    y_pred, hidden = model(x)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from models.cnn_encoder import CNNEncoder
from models.lstm_temporal import LSTMTemporal
from utils.config import FEATURE_DIM


class combined_model(nn.Module):
    def __init__(self):
        super(combined_model, self).__init__()

        self.encoder = CNNEncoder()
        self.temporal = LSTMTemporal()

    def forward(self, x, hidden=None):
        """
        x: (B, T, 1, 128, 128, 128)
        hidden: optional LSTM hidden state

        Returns:
            y_pred: (B, 1)
            hidden: (h_n, c_n)
        """

        B, T, C, D, H, W = x.shape

        # Step 1: Flatten time
        x = x.view(B * T, C, D, H, W)  # (B*T, 1, 128, 128, 128)

        # Step 2: CNN Encoder
        features = self.encoder(x)  # (B*T, FEATURE_DIM)

        # Step 3: Reshape for LSTM
        features = features.view(B, T, -1)  # (B, T, FEATURE_DIM)

        # Step 4: LSTM
        y_pred, hidden = self.temporal(features, hidden)  # (B, 1)

        return y_pred, hidden


if __name__ == "__main__":
    from utils.config import BATCH_SIZE, SEQ_LEN

    # Create dummy input
    x = torch.randn(BATCH_SIZE, SEQ_LEN, 1, 128, 128, 128)

    model = combined_model()

    y_pred, hidden = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y_pred.shape)  # should be (BATCH_SIZE, 1)

    if hidden is not None:
        h_n, c_n = hidden
        print("Hidden h_n shape:", h_n.shape)
        print("Hidden c_n shape:", c_n.shape)