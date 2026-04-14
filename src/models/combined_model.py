"""
Input:
    x: torch.Tensor of shape (B, T, 1, 128, 128, 128)  # raw POST scans
    days: torch.Tensor of shape (B, T) # log-normalized days_elapsed per timestep
    hidden: optional hidden state tuple for LSTM

Pipeline:
    1. Flatten time: (B*T, 1, 128, 128, 128)
    2. CNNEncoder → (B*T, FEATURE_DIM)
    3. Reshape: (B, T, FEATURE_DIM)
    4. Inject days_elapsed: (B, T, FEATURE_DIM + 1)
    5. LSTMTemporal → (B, 1)

Output:
    y_pred: torch.Tensor of shape (B, 1)
    hidden: optional hidden state tuple
"""

import torch
import torch.nn as nn
from cnn_encoder import CNNEncoder
from lstm_temporal import LSTMTemporal
from utils.config import cfg

class combined_model(nn.Module):
    def __init__(self):
        super(combined_model, self).__init__()
        self.encoder  = CNNEncoder()
        self.temporal = LSTMTemporal()

    def forward(self, x, days, hidden=None):
        """
        x:      (B, T, 1, 128, 128, 128)
        days:   (B, T) log-normalized days_elapsed
        hidden: optional LSTM hidden state

        Returns:
            y_pred: (B, 1)
            hidden: (h_n, c_n)
        """
        B, T, C, D, H, W = x.shape

        # Step 1: Flatten time
        x = x.view(B * T, C, D, H, W)           # (B*T, 1, 128, 128, 128)

        # Step 2: CNN Encoder
        features = self.encoder(x)              # (B*T, FEATURE_DIM)

        # Step 3: Reshape for LSTM
        features = features.view(B, T, -1)      # (B, T, FEATURE_DIM)

        # Step 4: Inject days_elapsed
        days = days.unsqueeze(-1)               # (B, T, 1)
        features = torch.cat([features, days], dim=-1)  # (B, T, FEATURE_DIM + 1)

        # Step 5: LSTM
        y_pred, hidden = self.temporal(features, hidden)  # (B, 1)

        return y_pred, hidden


if __name__ == "__main__":
    from utils.config import cfg

    x = torch.randn(cfg.BATCH_SIZE, cfg.SEQ_LEN, 1, 128, 128, 128)
    days = torch.randn(cfg.BATCH_SIZE, cfg.SEQ_LEN)

    model = combined_model()
    y_pred, hidden = model(x, days)

    print("Input shape:", x.shape)
    print("Days shape:", days.shape)
    print("Output shape:", y_pred.shape)  # (BATCH_SIZE, 1)