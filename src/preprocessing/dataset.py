import sys
from torch.utils.data import Dataset
import os
import json
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.sliding_window import create_sliding_windows
import utils.config as cfg


class MRIDataset(Dataset):
    """
    PyTorch Dataset for 3D MRI tumor sequences.

    Returns:
        x_seq:    (T, 1, H, W, D)
        days:     (T,) log-normalized days_elapsed per timestep
        y_label:  float32 tensor, shape (1,)
    """
    def __init__(self, data_root=None, label_json_path=None, seq_len=None, split=None):
        self.data_root = data_root or cfg.DATA_ROOT
        self.seq_len   = seq_len or cfg.SEQ_LEN

        self.sequence_paths = []
        self.days_elapsed   = []
        self.y              = []

        # Load labels
        with open(label_json_path or cfg.JSON_ROOT, "r") as f:
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

                files = [f for f in os.listdir(folder)
                         if f.endswith(".pt") and not f.startswith("._")]

                if len(files) == 0:
                    continue

                seq_paths.append(os.path.join(folder, files[0]))

            # only keep full valid sequences
            if len(seq_paths) == len(date_list):
                # extract days_elapsed for each transition in this sequence
                # labels[pid] keys are like "t1_t2", "t2_t3", etc. sorted by transition
                transitions = sorted(labels[pid].keys())

                # for SEQ_LEN=2 there is 1 transition (t1_t2)
                # we pad to length T by prepending 0 for the first timepoint
                days = [0.0] + [
                    float(labels[pid][t]['days_elapsed'])
                    for t in transitions[:len(date_list) - 1]
                ]

                self.sequence_paths.append(seq_paths)
                self.days_elapsed.append(days)
                self.y.append(label)

        print(f"Total valid sequences: {len(self.sequence_paths)}")

    def __len__(self):
        return len(self.sequence_paths)

    def __getitem__(self, idx):
        paths = self.sequence_paths[idx]

        tensors = [torch.load(p, weights_only=False).float() for p in paths]
        x_seq   = torch.stack(tensors)

        if x_seq.ndim == 4:
            x_seq = x_seq.unsqueeze(1)

        # log-normalize days (compresses large gaps, e.g. 365 → 5.9, 7 → 2.1)
        days_tensor = torch.log1p(
            torch.tensor(self.days_elapsed[idx], dtype=torch.float32)
        )  # (T,)

        y_label = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)

        return x_seq, days_tensor, y_label


if __name__ == "__main__":
    LABEL_JSON = os.path.join(cfg.JSON_ROOT, "labels.json")

    dataset = MRIDataset(cfg.DATA_ROOT, LABEL_JSON, seq_len=cfg.SEQ_LEN)
    print(f"Dataset length: {len(dataset)}")

    x, days, y = dataset[0]
    print(f"x shape:    {x.shape}")       # (T, 1, 128, 128, 128)
    print(f"days shape: {days.shape}")    # (T,)
    print(f"days:       {days}")          # log-normalized values
    print(f"y:          {y}")