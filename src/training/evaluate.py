import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.config import cfg
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
        for x, days, y in tqdm(dataloader, desc="Evaluating"):
            x = x.to(device)
            days = days.to(device).float()
            y = y.to(device).float()

            logits, _ = model(x, days)
            probs     = torch.sigmoid(logits)

            all_preds.append(probs)
            all_targets.append(y)

    all_probs   = torch.cat(all_probs).view(-1)
    all_targets = torch.cat(all_targets).view(-1)

    best_f1 = 0.0
    best_t  = 0.5

    for t in torch.linspace(0.1, 0.9, 81):
        preds = (all_probs > t).float()

        metrics = compute_metrics(preds, all_targets)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_t  = t.item()
    
    best_preds = (all_probs > best_t).float()
    final_metrics = compute_metrics(best_preds, all_targets)

    return {
        **final_metrics,
        "best_threshold": best_t
    }

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # validation dataset
    val_file = os.path.join(cfg.SPLIT_ROOT, "val.txt")

    dataset = MRIDataset(
        data_root=cfg.DATA_ROOT,
        label_json_path=os.path.join(cfg.JSON_ROOT, "labels.json"),
        seq_len=cfg.SEQ_LEN,
        split=val_file
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)

    # sort checkpoints by epoch
    ckpts = sorted(
        [f for f in os.listdir(cfg.CHECKPOINT_ROOT)
         if f.startswith("combined_model_epoch") and f.endswith(".pt")],
        key=lambda x: int(x.split("epoch")[1].split(".pt")[0])
    )

    if not ckpts:
        print("No checkpoints found in", cfg.CHECKPOINT_ROOT)
        exit()

    results = {}

    for ckpt_name in ckpts:
        print(f"\nEvaluating {ckpt_name}...")

        model      = combined_model().to(device)
        ckpt_path  = os.path.join(cfg.CHECKPOINT_ROOT, ckpt_name)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        metrics   = evaluate(model, dataloader, device)
        epoch_num = int(ckpt_name.split("epoch")[1].split(".pt")[0])
        results[epoch_num] = metrics

        print(f"Epoch {epoch_num}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # summary of best epoch by F1
    best_epoch = max(results, key=lambda e: results[e]['f1'])
    print(f"\nBest epoch by F1: {best_epoch} — F1: {results[best_epoch]['f1']:.4f}")