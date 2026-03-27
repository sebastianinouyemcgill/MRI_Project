import sys
from torch.utils.data import Dataset
from sliding_window import create_sliding_windows
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import DATA_ROOT, JSON_ROOT, SEQ_LEN

class MRIDataset(Dataset):
    """
    PyTorch Dataset for 3D MRI tumor sequences.
    
    Each item is a sequence of tensors (B, T, 1, 128, 128, 128)
    and the corresponding label dictionary from labels.json.
    
    Args:
        data_root (str): Root folder containing patient subfolders with preprocessed tensors.
        label_json_path (str): Path to labels.json from generate_labels().
        seq_len (int): Number of consecutive timepoints per sequence.
    """
    def __init__(self, data_root, label_json_path, seq_len=SEQ_LEN):
        self.data_root = data_root
        self.seq_len = seq_len

        # Load labels
        with open(label_json_path, "r") as f:
            self.labels = json.load(f)

        # Generate sequences
        print("Generating sliding window sequences...")
        self.X, self.y = create_sliding_windows(self.labels, seq_len=self.seq_len)
        print(f"Total sequences: {len(self.X)}")

        # Optional: convert all sequences to float32 and add channel dim if missing
        for i in range(len(self.X)):
            seq = self.X[i].float()  # ensure float
            # Add channel dim if tensor shape is (T, H, W, D)
            if seq.ndim == 4:
                seq = seq.unsqueeze(2)  # shape -> (T, 1, H, W, D)
            self.X[i] = seq

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns:
            tuple:
                x_tensor: torch.Tensor of shape (T, 1, H, W, D)
                y_label: dict with growth/label info for next transition
        """
        return self.X[idx], self.y[idx]


# Example usage
if __name__ == "__main__":

    LABEL_JSON = os.path.join(JSON_ROOT, "labels.json")  # full path to labels.json
    SEQ_LEN = 2  # you can also move this to config if you want

    dataset = MRIDataset(DATA_ROOT, LABEL_JSON, seq_len=SEQ_LEN)
    print(f"Dataset length: {len(dataset)}")

    # inspect first sequence
    x, y = dataset[0]
    print(f"First sequence shape: {x.shape}, label: {y}")