import os
import sys
import torch
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import DATA_ROOT, JSON_ROOT, SEQ_LEN 

from datetime import datetime

def parse_date(d):
    return datetime.strptime(d, "%Y-%m-%d")


def create_sliding_windows(label_json, seq_len):
    """
    Returns:
        X: list of dicts with:
            {
                "pid": str,
                "dates": [t1, t2, ..., tT],
                "start_idx": int   # index into sorted transitions
            }
        y: list of labels
    """

    X = []
    y = []

    for pid, transitions in label_json.items():
        # ---- robust sort ----
        sorted_items = sorted(
            transitions.items(),
            key=lambda x: parse_date(x[1]["date_ti"])
        )

        # ---- continuity check (hard fail = skip patient) ----
        valid = True
        for i in range(len(sorted_items) - 1):
            if sorted_items[i][1]["date_tf"] != sorted_items[i + 1][1]["date_ti"]:
                print(f"[SKIP] {pid} has non-contiguous dates")
                valid = False
                break
        if not valid:
            continue

        # ---- build date timeline ----
        dates = [item[1]["date_ti"] for item in sorted_items]
        dates.append(sorted_items[-1][1]["date_tf"])

        if len(dates) < seq_len + 1:
            continue

        # ---- sliding windows ----
        for i in range(len(dates) - seq_len):
            input_dates = dates[i:i + seq_len]

            # label = transition from last input → next
            transition = sorted_items[i + seq_len - 1][1]
            label = transition["label"]

            X.append({
                "pid": pid,
                "dates": input_dates,
                "start_idx": i  # 🔥 CRITICAL: enables correct alignment later
            })
            y.append(label)

    return X, y

if __name__ == "__main__":
    import os
    import json

    DATA_ROOT = "../../data/mini/processed"
    LABEL_JSON = "../../data/json/mini/labels.json"
    SEQ_LEN = 3

    with open(LABEL_JSON, "r") as f:
        labels = json.load(f)

    X, y = create_sliding_windows(labels, SEQ_LEN)

    print(f"Total sequences: {len(X)}")

    issues = {
        "missing_folder": 0,
        "bad_days_len": 0,
        "date_order": 0,
    }

    for i in range(min(50, len(X))):  # sample first 50
        meta = X[i]
        pid = meta["pid"]
        dates = meta["dates"]
        start_idx = meta["start_idx"]

        # ---- check ordering ----
        parsed = [parse_date(d) for d in dates]
        if parsed != sorted(parsed):
            print(f"[ORDER ERROR] {pid} {dates}")
            issues["date_order"] += 1

        # ---- check folders ----
        for d in dates:
            folder = os.path.join(DATA_ROOT, pid, d)
            if not os.path.exists(folder):
                print(f"[MISSING] {pid} {d}")
                issues["missing_folder"] += 1

        # ---- check days alignment ----
        sorted_items = sorted(
            labels[pid].items(),
            key=lambda x: parse_date(x[1]["date_ti"])
        )

        days = [0.0] + [
            float(sorted_items[j][1]["days_elapsed"])
            for j in range(start_idx, start_idx + len(dates) - 1)
        ]

        if len(days) != len(dates):
            print(f"[DAYS LEN ERROR] {pid}")
            issues["bad_days_len"] += 1

        # ---- debug print one clean sample ----
        if i == 0:
            print("\n=== SAMPLE ===")
            print("PID:", pid)
            print("Dates:", dates)
            print("Start idx:", start_idx)
            print("Days:", days)
            print("Label:", y[i])

    print("\n===== SUMMARY =====")
    for k, v in issues.items():
        print(f"{k}: {v}")


"""
def load_patient_tensors(patient_path):
    
    Loads only POST .pt tensors for a patient.
    Handles variable modalities and variable timepoints.
    Returns: list of tensors, one per timepoint (stacked if multiple modalities)
    
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
"""

"""
def create_sliding_windows(label_json, seq_len, modality=MODALITY):
    
    Returns:
        X: list of (patient_id, [date strings])
        y: list of labels
    

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
"""
