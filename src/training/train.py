import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
from src.utils.config import cfg
from src.preprocessing.dataset import MRIDataset
from src.models.combined_model import combined_model
from src.training.losses import BinaryClassificationLoss
from src.training.evaluate import evaluate

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading datasets...")
    # datasets
    LABEL_JSON  = os.path.join(cfg.JSON_ROOT, "labels.json")
    TRAIN_SPLIT = os.path.join(cfg.SPLIT_ROOT, "train.txt")
    VAL_SPLIT   = os.path.join(cfg.SPLIT_ROOT, "val.txt")

    train_dataset = MRIDataset(cfg.COLAB_ROOT, LABEL_JSON, seq_len=cfg.SEQ_LEN, split=TRAIN_SPLIT)
    val_dataset   = MRIDataset(cfg.COLAB_ROOT, LABEL_JSON, seq_len=cfg.SEQ_LEN, split=VAL_SPLIT)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    print("Creating dataloaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print("Initializing model")

    # model, loss, optimizer, scheduler
    model = combined_model().to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    criterion = BinaryClassificationLoss(
        pos_weight=cfg.POS_WEIGHT, # update based on SEQ_LEN
        label_smoothing=0.0 # keep at 0.0 until baseline stable
    ).to(device)

    print(f"Using pos_weight: {cfg.POS_WEIGHT:.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)

    print("Starting training...")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    print(f"Training for {cfg.NUM_EPOCHS} epochs with batch size {cfg.BATCH_SIZE}...")

    """
    # resume from checkpoint
    os.makedirs(cfg.CHECKPOINT_ROOT, exist_ok=True)
    all_ckpts = [f for f in os.listdir(cfg.CHECKPOINT_ROOT)
                 if f.startswith("combined_model_epoch") and f.endswith(".pt")]
    if all_ckpts:
        last_ckpt   = max(all_ckpts, key=lambda x: int(x.split("epoch")[1].split(".pt")[0]))
        resume_path = os.path.join(cfg.CHECKPOINT_ROOT, last_ckpt)
        checkpoint  = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1     = checkpoint.get('val_f1', 0.0)
        print(f"Resuming from epoch {start_epoch}, best F1 so far: {best_f1:.4f}")
    else:
        start_epoch = 1
        best_f1 = 0.0
    """

    # early stopping
    patience = 5
    epochs_no_imp = 0
    start_epoch = 1

    # training loop
    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):

        print(f"\nEpoch {epoch}/{cfg.NUM_EPOCHS}")
        
        model.train()
        epoch_loss = 0.0

        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.NUM_EPOCHS}", leave=False)
        for x_seq, days, y_batch in tqdm_bar:
            x_seq   = x_seq.to(device).float()
            days = days.to(device).float()
            y_batch = y_batch.to(device).float()

            optimizer.zero_grad()
            y_pred, _ = model(x_seq, days)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * x_seq.size(0)
            tqdm_bar.set_postfix({"batch_loss": loss.item()})

        avg_loss = epoch_loss / len(train_dataset)

        # validation
        metrics = evaluate(model, val_loader, device)
        val_f1  = metrics['f1']

        print(f"Epoch {epoch}/{cfg.NUM_EPOCHS} — "
              f"train loss: {avg_loss:.4f}  "
              f"val acc: {metrics['accuracy']:.4f}  "
              f"val prec: {metrics['precision']:.4f}  "
              f"val rec: {metrics['recall']:.4f}  "
              f"val F1: {val_f1:.4f}")

        # LR scheduler
        scheduler.step(val_f1)

        # save checkpoint
        checkpoint_path = os.path.join(cfg.CHECKPOINT_ROOT, f"combined_model_epoch{epoch}.pt")
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':                 avg_loss,
            'val_f1':               val_f1
        }, checkpoint_path)

        # save best model separately
        if val_f1 > best_f1:
            best_f1       = val_f1
            epochs_no_imp = 0
            shutil.copy(checkpoint_path, os.path.join(cfg.CHECKPOINT_ROOT, 'best_model.pt'))
            print(f"  New best model saved (F1: {best_f1:.4f}) ✓")
        else:
            epochs_no_imp += 1
            print(f"  No improvement for {epochs_no_imp}/{patience} epochs")
            if epochs_no_imp >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Training complete. Best val F1: {best_f1:.4f}")


if __name__ == "__main__":
    train()