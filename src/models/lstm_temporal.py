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