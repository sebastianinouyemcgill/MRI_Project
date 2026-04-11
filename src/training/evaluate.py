import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import DATA_ROOT, JSON_ROOT, SEQ_LEN, SPLIT_ROOT, CHECKPOINT_ROOT
from preprocessing.dataset import MRIDataset
from models.combined_model import combined_model


def compute_metrics(preds, targets):
    """
    preds, targets: torch tensors (N,)
    """
    preds   = preds.cpu()
    targets = targets.cpu()

    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1
    }


def evaluate(model, dataloader, device):
    """
    Evaluation loop.
    Returns dict of metrics.
    """
    model.eval()
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)
            probs     = torch.sigmoid(logits)
            preds     = (probs > 0.5).float()

            all_preds.append(preds)
            all_targets.append(y)

    all_preds   = torch.cat(all_preds).view(-1)
    all_targets = torch.cat(all_targets).view(-1)

    return compute_metrics(all_preds, all_targets)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validation dataset
    val_file = os.path.join(SPLIT_ROOT, "val.txt")

    dataset = MRIDataset(
        data_root=DATA_ROOT,
        label_json_path=os.path.join(JSON_ROOT, "labels.json"),
        seq_len=SEQ_LEN,
        split=val_file
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)

    # Sort checkpoints by epoch
    ckpts = sorted(
        [f for f in os.listdir(CHECKPOINT_ROOT)
         if f.startswith("combined_model_epoch") and f.endswith(".pt")],
        key=lambda x: int(x.split("epoch")[1].split(".pt")[0])
    )

    if not ckpts:
        print("No checkpoints found in", CHECKPOINT_ROOT)
        exit()

    results = {}

    for ckpt_name in ckpts:
        print(f"\nEvaluating {ckpt_name}...")

        model      = combined_model().to(device)
        ckpt_path  = os.path.join(CHECKPOINT_ROOT, ckpt_name)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        metrics   = evaluate(model, dataloader, device)
        epoch_num = int(ckpt_name.split("epoch")[1].split(".pt")[0])
        results[epoch_num] = metrics

        print(f"Epoch {epoch_num}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # Summary: best epoch by F1
    best_epoch = max(results, key=lambda e: results[e]['f1'])
    print(f"\nBest epoch by F1: {best_epoch} — F1: {results[best_epoch]['f1']:.4f}")