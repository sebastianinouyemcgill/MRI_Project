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
    print(f"\nLoading tensors for patient: {patient_id}")

    timepoints = sorted(os.listdir(patient_path))
    tensors = []

    for tp in timepoints:
        tp_path = os.path.join(patient_path, tp)
        if not os.path.isdir(tp_path):
            continue

        # Only load POST .pt files
        post_files = [f for f in os.listdir(tp_path) if f.endswith(".pt") and "POST" in f.upper()]
        if len(post_files) == 0:
            print(f"  Skipping timepoint {tp} (no POST .pt files)")
            continue

        modality_tensors = []
        for f in sorted(post_files):
            tensor_path = os.path.join(tp_path, f)
            try:
                data = torch.load(tensor_path)
                if isinstance(data, dict):
                    tensor = data.get("image", None)
                    if tensor is None:
                        print(f"    Skipping {f} (no 'image' key)")
                        continue
                else:
                    tensor = data
                modality_tensors.append(tensor)
            except Exception as e:
                print(f"    Failed to load {f}: {e}")

        if len(modality_tensors) == 0:
            print(f"  No valid POST tensors found in {tp}")
            continue

        try:
            if len(modality_tensors) > 1:
                tp_tensor = torch.stack(modality_tensors)
            else:
                tp_tensor = modality_tensors[0]
            tensors.append(tp_tensor)
            print(f"  Loaded {tp}: {len(modality_tensors)} POST modality(ies), shape {tp_tensor.shape}")
        except Exception as e:
            print(f"  Failed to stack tensors at {tp}: {e}")

    print(f"Finished patient {patient_id}, total timepoints loaded: {len(tensors)}")
    return tensors


def create_sliding_windows(label_json, seq_len=SEQ_LEN, modality=MODALITY):
    """
    Args:
        label_json: dict from labels.json
        seq_len: int
        modality: int or str (used for future modality handling)

    Returns:
        X: list of torch tensors [sequence_len x modality x H x W x D]
        y: list of labels (int)
    """
    X = []
    y = []

    total_patients = 0
    total_sequences = 0

    for patient_id in os.listdir(DATA_ROOT):
        patient_path = os.path.join(DATA_ROOT, patient_id)
        if not os.path.isdir(patient_path):
            continue
        if patient_id not in label_json:
            print(f"Skipping {patient_id} (no labels)")
            continue

        timepoint_tensors = load_patient_tensors(patient_path)
        labels = label_json[patient_id]

        num_scans = len(timepoint_tensors)
        if num_scans < seq_len + 1:
            print(f"{patient_id}: not enough scans for seq_len={seq_len}, has {num_scans}")
            continue

        total_patients += 1
        sequences_for_patient = 0

        for i in range(num_scans - seq_len):
            seq = timepoint_tensors[i:i+seq_len]

            # Corresponding label = next transition
            label_key = f"t{i+1}_t{i+2}"
            if label_key not in labels:
                print(f"{patient_id}: missing label for {label_key}, skipping")
                continue

            label = labels[label_key]["label"]

            try:
                seq_tensor = torch.stack(seq)
                X.append(seq_tensor)
                y.append(label)
                sequences_for_patient += 1
            except Exception as e:
                print(f"{patient_id}: failed to stack seq {i}-{i+seq_len}: {e}")

        if sequences_for_patient > 0:
            print(f"{patient_id}: {sequences_for_patient} sequences generated")
        total_sequences += sequences_for_patient

    print(f"\nTotal patients with sequences: {total_patients}")
    print(f"Total sequences generated: {total_sequences}")
    return X, y


if __name__ == "__main__":
    label_path = os.path.join(JSON_ROOT, "labels.json")
    with open(label_path, "r") as f:
        labels = json.load(f)

    # Generate sequences
    X, y = create_sliding_windows(labels, seq_len=SEQ_LEN, modality=MODALITY)

    print(f"\nDone. X sequences: {len(X)}, y labels: {len(y)}")