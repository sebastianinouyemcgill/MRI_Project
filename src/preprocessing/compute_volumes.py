import os
import json
import torch

def compute_tumor_volume(tensor, threshold_sigma): # Still don't know how to define tumour threshold for intensities
  """
  Compute tumor volume as intensity above threshold. Assume the tumor is 'brighter'
  than the average brain tissue (common in T1-weighted post-contrast).
  Use Standard Deviations (Sigma) above the mean to find 'hotspots'.
  """
  # 1. Ignore the black background (0s) to get brain-only stats
  brain_voxels = tensor[tensor > 0]
  if brain_voxels.numel() == 0:
    return 0.0

  mean_val = brain_voxels.mean()
  std_val = brain_voxels.std()

  # 2. Define threshold as Mean + N * StdDev
  # Typically, tumor tissue is significantly brighter than healthy gray matter
  threshold = mean_val + (threshold_sigma * std_val)

  tumor_voxels = (tensor > threshold).sum().item()
  tumor_volume = float(tumor_voxels)
  return tumor_volume

def compute_patient_volumes(data_path, output_path):
  """
  Loop through all patients and compute tumor volumes.
  """
  volumes = {}

  # Loop over patient IDs
  for patient_id in os.listdir(data_path):
    patient_path = os.path.join(data_path, patient_id)

    if not os.path.isdir(patient_path):
      continue

    volumes[patient_id] = {}

    scans = sorted(os.listdir(patient_path))

    # Loop over scan date in each patient
    for scan_date in scans:
      scan_path = os.path.join(patient_path, scan_date)

      if not os.path.isdir(scan_path):
        continue

      # Find .pt files inside each scan date folder
      pt_files = [f for f in os.listdir(scan_path) if f.endswith(".pt") and "POST" in f.upper()]

      if len(pt_files) == 0:
        print(f"No POST .pt files in {scan_path}")
        continue

      # For multiple .pt files
      for pt_file in pt_files:
        file_path = os.path.join(scan_path, pt_file)

      data = torch.load(file_path)

      # Handle dict case
      if isinstance(data, dict):
        tensor = data.get("image", None)
        if tensor is None:
            print(f"Skipping {file_path}, no 'image' key")
            continue
      else:
        tensor = data

      # Debug
      print(f"{patient_id} {scan_date}: min={tensor.min()} max={tensor.max()}")

      tumor_volume = compute_tumor_volume(tensor, 2.0)

      # Use scan_date as timepoint
      volumes[patient_id][scan_date] = tumor_volume

    num_timepoints = len(volumes[patient_id])

    if num_timepoints <= 1:
      print(f"Removing {patient_id} (only {num_timepoints} valid scan)")
      del volumes[patient_id]

  output_file = os.path.join(output_path, "volumes.json")

  with open(output_file, "w") as f:
    json.dump(volumes, f, indent=4)

  print(f"Saved volumes to {output_file}")
  return volumes

if __name__ == "__main__":
  data_path = "/data/processed"
  output_path = "/data/json"
  
  JSON_output = compute_patient_volumes(data_path, output_path)
  print("Done. Patients processed:", len(JSON_output))
  print(f"JSON: {JSON_output}")
