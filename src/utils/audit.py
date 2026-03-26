import os
from collections import defaultdict

DATASET_PATH = "/Volumes/SSD 2/Projects/MRI Project/Yale-Brain-Mets-Longitudinal"

scan_count_distribution = defaultdict(int)

for patient in os.listdir(DATASET_PATH):
    patient_path = os.path.join(DATASET_PATH, patient)

    if not os.path.isdir(patient_path):
        continue

    # count timepoint folders
    timepoints = [
        t for t in os.listdir(patient_path)
        if os.path.isdir(os.path.join(patient_path, t))
    ]

    num_scans = len(timepoints)

    if num_scans > 0:
        scan_count_distribution[num_scans] += 1

# print results
print("Scan count distribution:\n")
for num_scans in sorted(scan_count_distribution):
    print(f"{num_scans} scans: {scan_count_distribution[num_scans]} patients")