import torch
import torch.nn as nn


class BinaryClassificationLoss(nn.Module):
    """
    Loss function for tumor progression binary classification.

    Args:
        pos_weight (float): weight for positive class to handle class imbalance.
                           Set to neg_count / pos_count (e.g. 2978 / 1512 ≈ 1.97) based on seq_len=2 distribution.
        label_smoothing (float): smooths hard 0/1 labels to reduce overconfidence.
                                 0.0 = no smoothing, 0.1 = slight smoothing.
    """
    def __init__(self, pos_weight=None, label_smoothing=0.0):
        super().__init__()

        self.label_smoothing = label_smoothing

        weight = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)

    def forward(self, pred, target):
        """
        pred:   (B, 1) raw logits from model
        target: (B, 1) float labels (0.0 or 1.0)
        """
        if self.loss_fn.pos_weight is not None:
            self.loss_fn.pos_weight = self.loss_fn.pos_weight.to(pred.device)

        if self.label_smoothing > 0.0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        return self.loss_fn(pred, target)


if __name__ == "__main__":
    # simulate your class distribution
    POS_WEIGHT = 2470 / 1347 # update based on seq_len

    loss_fn = BinaryClassificationLoss(pos_weight=POS_WEIGHT, label_smoothing=0.1)

    B      = 8
    pred   = torch.randn(B, 1)
    target = torch.randint(0, 2, (B, 1)).float()
    loss   = loss_fn(pred, target)

    print("Loss:", loss.item())