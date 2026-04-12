"""
train.py: Training loop for CNN+LSTM CombinedModel

Inputs:
    - Dataset:
        MRIDataset instance
        - Returns:
            x_tensor: torch.Tensor of shape (T, 1, 128, 128, 128)
            y_label: dict with key "label" -> int {0,1}

    - Config (utils.config):
        DATA_ROOT: str      # path to preprocessed tensors
        JSON_ROOT: str      # path to labels.json
        SEQ_LEN: int        # number of consecutive timepoints per sequence
        FEATURE_DIM: int    # output feature dimension of CNN
        BATCH_SIZE: int
        CHECKPOINT_ROOT: str # directory to save model weights

    - Model:
        CombinedModel (CNN + LSTM)
        - Input: (B, T, 1, 128, 128, 128)
        - Output: y_pred: (B,1), hidden: optional tuple

Outputs:
    - Saved files in CHECKPOINT_ROOT:
        - combined_model_epoch{n}.pt (model state dict per epoch)
    - Console output (tqdm progress bar + epoch loss)

Exceptions / errors:
    - FileNotFoundError if DATA_ROOT or JSON_ROOT not found
    - RuntimeError if CUDA OOM occurs (GPU memory exceeded)
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import DATA_ROOT, JSON_ROOT, SEQ_LEN, BATCH_SIZE, CHECKPOINT_ROOT, SPLIT_ROOT, NUM_EPOCHS, LR
from preprocessing.dataset import MRIDataset
from models.combined_model import combined_model
from training.losses import BinaryClassificationLoss

def train():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & DataLoader
    LABEL_JSON = os.path.join(JSON_ROOT, "labels.json")
    SPLIT_FILE = os.path.join(SPLIT_ROOT, "train.txt")

    dataset = MRIDataset(DATA_ROOT, LABEL_JSON, seq_len=SEQ_LEN, split=SPLIT_FILE)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,   # parallel data loading
        pin_memory=True
    )

    # Model, loss, optimizer
    model = combined_model().to(device)

    criterion = BinaryClassificationLoss(
    pos_weight=2978 / 1512,   # from your Counter output
    label_smoothing=0.1        # optional, start with 0.0 if you want baseline first
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

    # Optional: resume from a checkpoint
    # List all saved checkpoints
    all_ckpts = [f for f in os.listdir(CHECKPOINT_ROOT) if f.startswith("combined_model_epoch") and f.endswith(".pt")]
    if all_ckpts:
        # Pick the one with the highest epoch number
        last_ckpt = max(all_ckpts, key=lambda x: int(x.split("epoch")[1].split(".pt")[0]))
        resume_path = os.path.join(CHECKPOINT_ROOT, last_ckpt)
        
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 1

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        tqdm_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False)

        for x_seq, y_dict in tqdm_bar:
            # x_seq: (B, T, 1, 128, 128, 128)
            # y_dict: dict from dataset __getitem__
            y_batch = y_dict

            # Move to device
            x_seq = x_seq.to(device).float()
            y_batch = y_batch.to(device).float()

            # Forward + backward
            optimizer.zero_grad()
            y_pred, _ = model(x_seq)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_seq.size(0)
            tqdm_bar.set_postfix({"batch_loss": loss.item()})

        # Average epoch loss
        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch}/{NUM_EPOCHS} - Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_ROOT, f"combined_model_epoch{epoch}.pt")
        
        # Save checkpoint including optimizer
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)

    print("Training complete.")

if __name__ == "__main__":
    train()