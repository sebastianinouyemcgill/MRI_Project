import sys
import os
import json
import math
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import JSON_ROOT, SPLIT_ROOT, RANDOM_SEED

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

labels_path = os.path.join(JSON_ROOT, "labels.json")
if not os.path.exists(labels_path):
    raise FileNotFoundError(f"labels.json not found at {labels_path}")

with open(labels_path, "r") as f:
    labels = json.load(f)


# Define bins for number of timepoints
bins = {
    "short": [],   # 2-3 timepoints
    "medium": [],  # 4-10 timepoints
    "long": []     # >10 timepoints
}

for patient_id, patient_data in labels.items():
    num_timepoints = len(patient_data) + 1  # number of scans = transitions + 1
    if num_timepoints <= 3:
        bins["short"].append(patient_id)
    elif num_timepoints <= 10:
        bins["medium"].append(patient_id)
    else:
        bins["long"].append(patient_id)

print("Patients per bin:")
for b, p_list in bins.items():
    print(f"  {b}: {len(p_list)} patients")

random.seed(RANDOM_SEED)

splits = {"train": [], "val": [], "test": []}

for b, p_list in bins.items():
    random.shuffle(p_list)
    n = len(p_list)
    n_train = math.floor(n * TRAIN_RATIO)
    n_val = math.floor(n * VAL_RATIO)
    n_test = n - n_train - n_val  # remainder goes to test

    # Assign to splits
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
    print(f"Saved {split_name} split to {out_file}")

# If implementing k-fold later:
# 1. Keep bins for patient length as above
# 2. Assign each patient to 1 of k folds (ensure each fold has a mix of bins)
# 3. For each iteration:
#    - train = all folds except fold_i
#    - test  = fold_i
# 4. You can repeat similar printing and writing logic for fold_i