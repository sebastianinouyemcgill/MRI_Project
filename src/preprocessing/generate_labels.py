import sys
import os
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import DATA_ROOT, JSON_ROOT, GROWTH_THRESHOLD


def compute_growth(v1, v2):
    '''
    relative tumour growth
    '''
    if v1 == 0:
        return float('inf') if v2 > 0 else 0
    return (v2 - v1) / v1


def compute_time_delta_days(date1, date2):
  """
  Compute time difference in days between two scan dates.
  """
  d1 = datetime.strptime(date1, "%Y-%m-%d")
  d2 = datetime.strptime(date2, "%Y-%m-%d")
  diff = (d2 - d1).days
  return diff


def generate_labels(volume_json_path=JSON_ROOT, output_path=JSON_ROOT, growth_threshold=GROWTH_THRESHOLD):
  """
  Generate progression labels from computed tumor volumes.
  Output format:
  {
    "patient_id": {
      "t1_t2": {
        "growth": float,
        "label": int,
        "delta_days": int
      }
    }
  }
  """
  if not os.path.exists(volume_json_path):
        raise FileNotFoundError(f"Error: {volume_json_path} not found.")

  # Load volumes
  with open(volume_json_path, "r") as f:
    volumes = json.load(f)

  labels = {}

  for patient_id, time_scan in volumes.items():

    # Sort dates chronologically
    timepoints = sorted(time_scan.keys(), key=lambda x: datetime.strptime(x, "%Y-%m-%d"))

    # Filter out scans with less than 2 timepoints
    if len(timepoints) < 2:
      print(f"Skipping {patient_id} (not enough timepoints)")
      continue

    labels[patient_id] = {}
    print(f"\nProcessing {patient_id} with {len(timepoints)} timepoints")

    # Assign labels for each timepoint
    for i in range(len(timepoints) - 1):
      ti = timepoints[i]
      tf = timepoints[i + 1]

      vi = time_scan[ti]
      vf = time_scan[tf]

      growth = compute_growth(vi, vf)
      label = 1 if growth > growth_threshold else 0
      delta_days = compute_time_delta_days(ti, tf)

      key = f"t{i+1}_t{i+2}"

      labels[patient_id][key] = {
         "date_ti": ti,
         "date_tf": tf,
         "volume_ti": vi,
         "volume_tf": vf,
         "growth": growth,
         "label": label,
         "days_elapsed": delta_days
         }

      print(f"{key}: growth={growth:.3f}, label={label}, Δt={delta_days} days")

  # Save output
  os.makedirs(output_path, exist_ok=True)
  output_file = os.path.join(output_path, "labels.json")

  with open(output_file, "w") as f:
    json.dump(labels, f, indent=4)

  print(f"\nSaved labels to {output_file}")
  print(f"Total patients with labels: {len(labels)}")

  return labels

# RUN
if __name__ == "__main__":
    volume_json_path = os.path.join(JSON_ROOT, "volumes.json")  

    full_labels = generate_labels(volume_json_path, output_path=JSON_ROOT, growth_threshold=GROWTH_THRESHOLD)

    print("Done. Patients processed:", len(full_labels))
    print(f"JSON: {full_labels}")
