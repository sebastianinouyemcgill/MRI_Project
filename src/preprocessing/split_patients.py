import sys
import os
import json
import math
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import JSON_ROOT, SPLIT_ROOT, RANDOM_SEED

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

labels_path = os.path.join(JSON_ROOT, "labels.json")
if not os.path.exists(labels_path):
    raise FileNotFoundError(f"labels.json not found at {labels_path}")

with open(labels_path, "r") as f:
    labels = json.load(f)

# Stratify by (bin, label) so each split has both classes
bins = {
    ("short",  0): [], ("short",  1): [],
    ("medium", 0): [], ("medium", 1): [],
    ("long",   0): [], ("long",   1): [],
}

for pid, patient_data in labels.items():
    num_timepoints = len(patient_data) + 1

    if num_timepoints <= 3:
        b = "short"
    elif num_timepoints <= 10:
        b = "medium"
    else:
        b = "long"

    # get patient-level label: 1 if any transition is positive
    has_progression = int(any(entry.get("label", 0) == 1 for entry in patient_data.values()))
    bins[(b, has_progression)].append(pid)

print("Patients per (bin, label):")
for key, p_list in bins.items():
    print(f"  {key}: {len(p_list)} patients")

random.seed(RANDOM_SEED)

splits = {"train": [], "val": [], "test": []}

for (b, lbl), p_list in bins.items():
    random.shuffle(p_list)
    n       = len(p_list)
    n_train = math.floor(n * TRAIN_RATIO)
    n_val   = math.floor(n * VAL_RATIO)

    # guarantee at least 1 per split if enough patients exist
    if n >= 3:
        n_train = max(1, n_train)
        n_val   = max(1, n_val)
    n_test = n - n_train - n_val

    splits["train"].extend(p_list[:n_train])
    splits["val"].extend(p_list[n_train:n_train + n_val])
    splits["test"].extend(p_list[n_train + n_val:])

for split_name, p_list in splits.items():
    print(f"{split_name.upper()}: {len(p_list)} patients")

os.makedirs(SPLIT_ROOT, exist_ok=True)
for split_name, p_list in splits.items():
    out_file = os.path.join(SPLIT_ROOT, f"{split_name}.txt")
    with open(out_file, "w") as f:
        for pid in p_list:
            f.write(f"{pid}\n")
    print(f"Saved {split_name} split → {out_file}")