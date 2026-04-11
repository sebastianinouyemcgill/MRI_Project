import sys
from torch.utils.data import Dataset
import os
import json
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.sliding_window import create_sliding_windows
import utils.config as cfg
import importlib
importlib.reload(cfg)


class MRIDataset(Dataset):
    """
    PyTorch Dataset for 3D MRI tumor sequences.

    Ensures that only valid sequences are included, skips missing files,
    and prints a summary per patient.

    Returns:
        x_tensor: (T, 1, H, W, D)
        y_label: torch tensor float32, shape (1,)
    """
    def __init__(self, data_root=cfg.DATA_ROOT, label_json_path=cfg.JSON_ROOT, seq_len=cfg.SEQ_LEN, split=None):
        self.data_root = data_root
        self.seq_len = seq_len

        self.sequence_paths = []
        self.y = []

        # Load labels
        with open(label_json_path, "r") as f:
            labels = json.load(f)

        # Apply split if provided
        if split is not None:
            with open(split, "r") as f:
                patient_ids = set(line.strip() for line in f)
            labels = {pid: info for pid, info in labels.items() if pid in patient_ids}

        # Generate sliding window metadata (NO tensors)
        X_meta, y = create_sliding_windows(labels, seq_len=self.seq_len)

        for (pid, date_list), label in zip(X_meta, y):
            seq_paths = []

            for date in date_list:
                folder = os.path.join(self.data_root, pid, date)

                if not os.path.exists(folder):
                    continue

                files = [f for f in os.listdir(folder) if f.endswith(".pt") and not f.startswith("._")]

                if len(files) == 0:
                    continue

                full_path = os.path.join(folder, files[0])
                seq_paths.append(full_path)

            # only keep full valid sequences
            if len(seq_paths) == len(date_list):
                self.sequence_paths.append(seq_paths)
                self.y.append(label)

        print(f"Total valid sequences: {len(self.sequence_paths)}")

    def __len__(self):
        return len(self.sequence_paths)

    def __getitem__(self, idx):
        paths = self.sequence_paths[idx]

        tensors = [torch.load(p, weights_only=False).float() for p in paths]
        x_seq = torch.stack(tensors)

        if x_seq.ndim == 4:
            x_seq = x_seq.unsqueeze(1)

        y_label = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)

        return x_seq, y_label


# Example usage
if __name__ == "__main__":

    LABEL_JSON = os.path.join(cfg.JSON_ROOT, "labels.json")
    SEQ_LEN = 2  # you can also move this to config if you want

    dataset = MRIDataset(cfg.DATA_ROOT, LABEL_JSON, seq_len=cfg.SEQ_LEN)
    print(f"Dataset length: {len(dataset)}")

    # inspect first sequence
    x, y = dataset[0]
    print(f"First sequence shape: {x.shape}, label: {y}")