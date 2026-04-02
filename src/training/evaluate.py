import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.config import SPLIT_ROOT, CHECKPOINT_ROOT
from preprocessing.dataset import MRIDataset
from models.combined_model import combined_model

def load_patient_ids(split_file):
  with open(split_file, "r") as f:
    patient_ids = [line.strip() for line in f.readlines()]
  return patient_ids

def compute_metrics(preds, targets):
  """
  preds, targets: torch tensors (N,)
  """
  preds = preds.cpu()
  targets = targets.cpu()

  tp = ((preds == 1) & (targets == 1)).sum().item()
  tn = ((preds == 0) & (targets == 0)).sum().item()
  fp = ((preds == 1) & (targets == 0)).sum().item()
  fn = ((preds == 0) & (targets == 1)).sum().item()

  accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
  precision = tp / (tp + fp + 1e-8)
  recall = tp / (tp + fn + 1e-8)
  f1 = 2 * precision * recall / (precision + recall + 1e-8)

  return {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1
  }

def evaluate(model, dataloader, device):
  """
  Evaluation loop
  """
  model.eval()
  all_preds = []
  all_targets = []

  with torch.no_grad():  # Disable gradients
    for batch in tqdm(dataloader, desc="Evaluating"):
      x = batch["image"].to(device)    # (B, T, 1, 128,128,128)
      y = batch["label"].to(device)    # (B, 1)

      # Forward pass
      logits, _ = model(x)             # (B, 1)

      # Convert logits → probabilities
      probs = torch.sigmoid(logits)    # (B, 1)

      # Convert to binary predictions
      preds = (probs > 0.5).float()    # (B, 1)

      all_preds.append(preds)
      all_targets.append(y)

  # Concatenate all batches
  all_preds = torch.cat(all_preds).view(-1)
  all_targets = torch.cat(all_targets).view(-1)
  
  metrics = compute_metrics(all_preds, all_targets)
  return metrics


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Load validation patient IDs
  val_file = os.path.join(SPLIT_ROOT, "val.txt")
  val_patient_ids = load_patient_ids(val_file)

  print(f"Loaded {len(val_patient_ids)} validation patients")

  dataset = MRIDataset(Dataset=val_patient_ids)
  dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

  ckpts = [f for f in os.listdir(CHECKPOINT_ROOT)
         if f.startswith("combined_model_epoch") and f.endswith(".pt")]
  # Sort by epoch number
  ckpts = sorted(ckpts, key=lambda x: int(x.split("epoch")[1].split(".pt")[0]))

  results = {}

  # Loop through model checkpoints from training
  for ckpt_name in ckpts:
    print(f"\nEvaluating {ckpt_name}...")
    
    model = combined_model()
    model.to(device)
    model_path = os.path.join(CHECKPOINT_ROOT, ckpt_name)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Run evaluation
    metrics = evaluate(model, dataloader, device)
    epoch_num = int(ckpt_name.split("epoch")[1].split(".pt")[0])
    results[epoch_num] = metrics

    print(f"Epoch {epoch_num}:")
    for k, v in metrics.items():
      print(f"{k}: {v:.4f}")
