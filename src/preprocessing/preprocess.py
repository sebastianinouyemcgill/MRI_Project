import os
import numpy as np
import torch
import nibabel as nib
import torch.nn.functional as F
from scipy.ndimage import zoom
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Inspect single file
def inspect_file(file_path):
    print("\nInspecting file:", file_path)
    img = nib.load(file_path)
    data = img.get_fdata()
    header = img.header

    print("Shape:", data.shape)
    print("Voxel spacing:", header.get_zooms())
    print("Datatype:", data.dtype)
    print("Min intensity:", np.min(data))
    print("Max intensity:", np.max(data))
    print("Mean intensity:", np.mean(data))

# Preprocess single NIfTI file
def preprocess_nifti(file_path, target_size=(128,128,128), target_spacing=(1.0,1.0,1.0)):
    img = nib.load(file_path)
    data = img.get_fdata()
    original_shape = data.shape
    original_spacing = img.header.get_zooms()

    # Remove NaNs
    data = np.nan_to_num(data)

    # Normalize intensity
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val > min_val:
        data = (data - min_val) / (max_val - min_val)

    # Resample to isotropic spacing
    zoom_factors = [original_spacing[i]/target_spacing[i] for i in range(3)]
    data = zoom(data, zoom=zoom_factors, order=1)  # linear interpolation

    # Convert to torch tensor (float16 to save space)
    tensor = torch.tensor(data).half()

    # Add channel dimension
    tensor = tensor.unsqueeze(0)

    # Add batch dimension for interpolation
    tensor = tensor.unsqueeze(0)

    # Resize to target size
    tensor = F.interpolate(tensor, size=target_size, mode="trilinear", align_corners=False)

    # Remove batch dimension
    tensor = tensor.squeeze(0)
    return tensor

# Worker function for parallel processing
def process_file(args):
    input_path, output_path, target_size = args

    # Skip if already processed (silent)
    if os.path.exists(output_path):
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        tensor = preprocess_nifti(input_path, target_size=target_size)
        torch.save(tensor, output_path)
        return "Saved"
    except Exception as e:
        return f"Failed {input_path}: {e}"

# Find first NIfTI file (for inspection)
def find_first_nifti(data_root):
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                return os.path.join(root, file)
    return None

# Preprocess entire dataset (parallel)
def preprocess_dataset(input_root, output_root, target_size=(128,128,128),
                       max_files=None, num_workers=None, verbose=True):
    """
    input_root: root folder of raw NIfTI files
    output_root: root folder to save .pt tensors
    target_size: final 3D size for network
    max_files: optional, only process first N files (for testing)
    num_workers: optional, number of parallel workers (defaults to os.cpu_count())
    verbose: prints progress
    """

    # Gather all files and corresponding output paths
    file_list = []
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if not (file.endswith(".nii") or file.endswith(".nii.gz")):
                continue
            input_path = os.path.join(root, file)

            relative_path = os.path.relpath(root, input_root)
            output_folder = os.path.join(output_root, relative_path)
            output_file = file.replace(".nii.gz", ".pt").replace(".nii", ".pt")
            output_path = os.path.join(output_folder, output_file)

            file_list.append((input_path, output_path, target_size))

    if max_files is not None:
        file_list = file_list[:max_files]

    total_files = len(file_list)
    print(f"Total files to process: {total_files}")

    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)  # default: 8 or all cores

    # Parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(process_file, file_list), total=total_files):
            if result is not None and result.startswith("Failed"):
                print(result)
    
    """
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i, result in enumerate(executor.map(process_file, file_list), 1):
            if verbose and (i % 100 == 0 or result is not None and result.startswith("Failed")):
                print(f"[{i}/{total_files}] {result}")
    """

# Main block
if __name__ == "__main__":
    DATA_DIR = "/Volumes/SSD 2/Projects/MRI Project/Yale-Brain-Mets-Longitudinal"
    OUTPUT_DIR = "/Volumes/SSD 2/Projects/MRI Project/Processed Data"

    # Check folders
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Preprocess (test with 10 files)
    preprocess_dataset(DATA_DIR, OUTPUT_DIR)

    print("\nFinished preprocessing test batch.")