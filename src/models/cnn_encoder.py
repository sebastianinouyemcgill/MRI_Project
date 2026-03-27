"""
CNNEncoder

Input:
    x: torch.Tensor of shape (B*T, 1, 128, 128, 128)  # flattened batch-time sequences

Output:
    features: torch.Tensor of shape (B*T, FEATURE_DIM)  # latent features per volume

Usage:
    encoder = CNNEncoder()
    features = encoder(x)
"""