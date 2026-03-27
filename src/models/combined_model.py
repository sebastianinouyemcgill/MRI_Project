"""
CombinedModel

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
    model = CombinedModel()
    y_pred, hidden = model(x)
"""