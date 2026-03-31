import os
import sys
import torch
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import DATA_ROOT, JSON_ROOT, SEQ_LEN, MODALITY 

def load_patient_tensors(patient_path):
    """
    Loads only POST .pt tensors for a patient.
    Handles variable modalities and variable timepoints.
    Returns: list of tensors, one per timepoint (stacked if multiple modalities)
    """
    patient_id = os.path.basename(patient_path)
    # print(f"\nLoading tensors for patient: {patient_id}")

    timepoints = sorted(os.listdir(patient_path))
    tensors = []

    for tp in timepoints:
        tp_path = os.path.join(patient_path, tp)
        if not os.path.isdir(tp_path):
            continue

        # Only load POST .pt files
        post_files = [f for f in os.listdir(tp_path) if f.endswith(".pt") and "POST" in f.upper()]
        if len(post_files) == 0:
            # print(f"  Skipping timepoint {tp} (no POST .pt files)")
            continue

        modality_tensors = []
        for f in sorted(post_files):
            tensor_path = os.path.join(tp_path, f)
            try:
                data = torch.load(tensor_path)
                if isinstance(data, dict):
                    tensor = data.get("image", None)
                    if tensor is None:
                        # print(f"    Skipping {f} (no 'image' key)")
                        continue
                else:
                    tensor = data
                modality_tensors.append(tensor)
            except Exception as e:
                # print(f"    Failed to load {f}: {e}")
                pass
        if len(modality_tensors) == 0:
            # print(f"  No valid POST tensors found in {tp}")
            continue

        try:
            if len(modality_tensors) > 1:
                tp_tensor = torch.stack(modality_tensors)
            else:
                tp_tensor = modality_tensors[0]
            tensors.append(tp_tensor)
            # print(f"  Loaded {tp}: {len(modality_tensors)} POST modality(ies), shape {tp_tensor.shape}")
        except Exception as e:
            # print(f"  Failed to stack tensors at {tp}: {e}")
            pass

    # print(f"Finished patient {patient_id}, total timepoints loaded: {len(tensors)}")
    return tensors


def create_sliding_windows(label_json, seq_len, modality=MODALITY):
    """
    Returns:
        X: list of (patient_id, [date strings])
        y: list of labels
    """

    X = []
    y = []

    for pid, transitions in label_json.items():
        # Sort transitions by start date
        sorted_items = sorted(
            transitions.items(),
            key=lambda x: x[1]["date_ti"]
        )

        # Build ordered list of dates
        dates = []
        for key, info in sorted_items:
            dates.append(info["date_ti"])
        # add final date
        dates.append(sorted_items[-1][1]["date_tf"])

        if len(dates) < seq_len + 1:
            continue

        for i in range(len(dates) - seq_len):
            input_dates = dates[i:i + seq_len]

            # label from NEXT transition
            transition = sorted_items[i + seq_len - 1][1]
            label = transition["label"]

            X.append((pid, input_dates))
            y.append(label)

    return X, y


if __name__ == "__main__":
    label_path = os.path.join(JSON_ROOT, "labels.json")
    with open(label_path, "r") as f:
        labels = json.load(f)

    # Generate sequences
    X, y = create_sliding_windows(labels, seq_len=SEQ_LEN, modality=MODALITY)

    print(f"\nDone. X sequences: {len(X)}, y labels: {len(y)}")