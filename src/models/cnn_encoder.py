"""
Input:
    x: torch.Tensor of shape (B*T, 1, 128, 128, 128)  # flattened batch-time sequences

Output:
    features: torch.Tensor of shape (B*T, FEATURE_DIM)  # latent features per volume

Usage:
    encoder = CNNEncoder()
    features = encoder(x)
"""

import torch
import torch.nn as nn
from utils.config import cfg

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()

        self.conv_block1 = self._conv_block(1, 16)     # (128 -> 64)
        self.conv_block2 = self._conv_block(16, 32)    # (64 -> 32)
        self.conv_block3 = self._conv_block(32, 64)    # (32 -> 16)
        self.conv_block4 = self._conv_block(64, 128)   # (16 -> 8)

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(128, cfg.FEATURE_DIM)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)  # halves each dimension
        )

    def forward(self, x):
        """
        x: (B*T, 1, 128, 128, 128)

        returns:
            features: (B*T, FEATURE_DIM)
        """

        x = self.conv_block1(x)  # -> (B*T, 16, 64, 64, 64)
        x = self.conv_block2(x)  # -> (B*T, 32, 32, 32, 32)
        x = self.conv_block3(x)  # -> (B*T, 64, 16, 16, 16)
        x = self.conv_block4(x)  # -> (B*T, 128, 8, 8, 8)

        x = self.global_pool(x)  # -> (B*T, 128, 1, 1, 1)
        x = x.view(x.size(0), -1)  # -> (B*T, 128)

        features = self.fc(x)  # -> (B*T, FEATURE_DIM)

        return features


if __name__ == "__main__":
    B, T = 2, 3
    x = torch.randn(B*T, 1, 128, 128, 128)

    model = CNNEncoder()
    out = model(x)

    print("Input:", x.shape)
    print("Output:", out.shape)  # should be (B*T, FEATURE_DIM)