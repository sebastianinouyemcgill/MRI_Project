import os
import sys
import json
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import DATA_ROOT, JSON_ROOT

def compute_tumor_volume(tensor, threshold_sigma): 
    brain_voxels = tensor[tensor > 0]
    if brain_voxels.numel() == 0:
        return 0.0

    mean_val = brain_voxels.mean()
    std_val = brain_voxels.std()
    threshold = mean_val + (threshold_sigma * std_val)

    tumor_voxels = (tensor > threshold).sum().item()
    return float(tumor_voxels)

def process_pt_file(file_path, threshold_sigma):
    """Helper for parallel processing of a single .pt file."""
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception as e:
        return 0.0
    
    if isinstance(data, dict):
        tensor = data.get("image", None)
        if tensor is None:
            return 0.0
    else:
        tensor = data
    return compute_tumor_volume(tensor, threshold_sigma)

def compute_patient_volumes(data_path=DATA_ROOT, output_path=JSON_ROOT, threshold_sigma=2.0, max_workers=4):
    volumes = {}

    patients = [p for p in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, p))]
    for patient_id in tqdm(patients, desc="Patients"):
        patient_path = os.path.join(data_path, patient_id)
        volumes[patient_id] = {}
        scans = sorted(os.listdir(patient_path))

        for scan_date in tqdm(scans, desc=f"Scans for {patient_id}", leave=False):
            scan_path = os.path.join(patient_path, scan_date)
            if not os.path.isdir(scan_path):
                continue

            pt_files = [f for f in os.listdir(scan_path) if f.endswith(".pt") and "POST" in f.upper()]
            if not pt_files:
                continue

            tumor_volume = 0.0
            # Parallel processing per .pt file
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for pt_file in pt_files:
                    file_path = os.path.join(scan_path, pt_file)
                    futures.append(executor.submit(process_pt_file, file_path, threshold_sigma))
                for f in as_completed(futures):
                    tumor_volume += f.result()

            volumes[patient_id][scan_date] = tumor_volume

        num_timepoints = len(volumes[patient_id])
        if num_timepoints <= 1:
            # Remove patients with <=1 valid scan
            del volumes[patient_id]

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "volumes.json")
    with open(output_file, "w") as f:
        json.dump(volumes, f, indent=4)

    print(f"Saved volumes to {output_file}")
    return volumes

if __name__ == "__main__":
    threshold_sigma = 2.0
    JSON_output = compute_patient_volumes(data_path=DATA_ROOT, output_path=JSON_ROOT,
                                          threshold_sigma=threshold_sigma, max_workers=4)
    print("Done. Patients processed:", len(JSON_output))