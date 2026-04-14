import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from preprocessing.dataset import MRIDataset
from models.combined_model import combined_model
from training.losses import BinaryClassificationLoss 
import os
from utils.config import DATA_ROOT, JSON_ROOT, SEQ_LEN, BATCH_SIZE, CHECKPOINT_ROOT
LABEL_JSON = os.path.join(JSON_ROOT, "labels.json")


dataset = MRIDataset(DATA_ROOT, LABEL_JSON, seq_len=SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# 2. Model
model = combined_model()
model.train()  # enable gradients

# 3. Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Ensure checkpoint folder exists
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

# 4. Mini training loop (tiny overfit)
num_epochs = 3  # small for test
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i, (x_batch, y_batch) in enumerate(loader):
        optimizer.zero_grad()

        # y_batch is already a tensor of shape (B,1), dtype=torch.float32
        out, hidden = model(x_batch)

        # Compute loss
        loss = BinaryClassificationLoss()(out, y_batch)

        # Backward
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Batch {i+1}, loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} average loss: {epoch_loss / len(loader):.4f}")

    # 5. Save checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_ROOT, f"combined_model_epoch{epoch+1}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

# 6. Final forward sanity check
with torch.no_grad():
    x_sample, y_sample = next(iter(loader))
    out, _ = model(x_sample)
    print("\nFinal batch outputs:")
    print(out)
    print("Target labels:")
    print(y_sample)