import torch
import torch.nn as nn

class BinaryClassificationLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.loss_fn = nn.BCEWithLogitsLoss()

  def forward(self, pred, target):
    """
    pred: (B, 1) logits (raw model output)
    target: (B, 1) labels (0 or 1)
    """
    loss = self.loss_fn(pred, target)
    return loss

if __name__ == "__main__":
  loss_fn = BinaryClassificationLoss()

  B = 8 # Placeholder, will replace later

  pred = torch.randn(B, 1)  # Mock data
  target = torch.randint(0, 2, (B, 1)).float()
  loss = loss_fn(pred, target)

  print("Loss:", loss.item())
