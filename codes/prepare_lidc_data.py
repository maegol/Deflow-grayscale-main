# Script to prepare the LIDC dataset for CycleGAN
# Remove duplicates
# Obtain the high-dose and low-dose cases
# Split data into train and test
# window clipping -1200;600
# normalize to [0,1]
# resize to 256x256 (maybe)
# save as .png or .npy files
# Save the data in the following structure:
# - /path/to/data/trainA/*.npy
# - /path/to/data/trainB/*.npy
# - /path/to/data/testA/*.npy
# - /path/to/data/testB/*.npy
import os
import numpy as np
import pylidc as pl
import random
import shutil
import pydicom
from collections import defaultdict
import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split
from PIL import Image

DOSE_THRESHOLD = 80       # mA
WINDOW_MIN, WINDOW_MAX = -1200, 600
IMAGE_SIZE = 512
TEST_SIZE = 0.2
RANDOM_STATE = 42
OUTPUT_DIR = "/mnt/home_liu/degradation/method_4/dataset"

# 1. Query all scans and group by patient

all_scans = pl.query(pl.Scan).all()
patient_scans = defaultdict(list)

for scan in all_scans:
    patient_scans[scan.patient_id].append(scan)


print(f"Total scans: {len(all_scans)}")
print(f"First scan sample: ID={all_scans[0].id}, Patient={all_scans[0].patient_id}")

removed_duplicates = 0
high_scans, low_scans = [], []
unique_scans = []
high_dose_count = 0
low_dose_count = 0

# 2. Deduplicate & classify by dose

for pid, scans in patient_scans.items():
    
    if len(scans) > 1:
        removed_duplicates += 1

    dicom_dir = scans[0].get_path_to_dicom_files()
    dcm_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    dcm_first = sorted(dcm_files)[0] if dcm_files else None
    ds = pydicom.dcmread(os.path.join(dicom_dir, dcm_first))
    tube_current = ds.XRayTubeCurrent
    unique_scans.append(scans[0])
    if tube_current > DOSE_THRESHOLD:
        high_dose_count += 1
        high_scans.append(scans[0])
    else:
        low_dose_count += 1
        low_scans.append(scans[0])

# Results
print("\n=== Final Counts ===")
print(f"Unique patients: {len(unique_scans)}")
print(f"High-dose scans: {high_dose_count}")
print(f"Low-dose scans: {low_dose_count}")
print(f"Total duplicates removed: {removed_duplicates}")


# 3. Split into train/test
high_train, high_test = train_test_split(
    high_scans, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
low_train, low_test = train_test_split(
    low_scans,  test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print("\n=== Train/Test Split ===")
print(f"High-dose train: {len(high_train)}, test: {len(high_test)}")
print(f"Low-dose train: {len(low_train)}, test: {len(low_test)}")

# 4. Prepare output folders
folders = {
    "trainA": high_train,
    "testA":  high_test,
    "trainB": low_train,
    "testB":  low_test
}
for folder in folders:
    os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)


# 5. Preprocessing function
def process_and_save(scans, domain_folder):
    out_dir = os.path.join(OUTPUT_DIR, domain_folder)
    for scan in scans:
        if scan.patient_id != "LIDC-IDRI-0085":
            continue
        dicom_dir = scan.get_path_to_dicom_files()
        files = sorted(glob.glob(os.path.join(dicom_dir, "*.dcm")))
        print(f"Processing scan {scan.id} with {len(files)} DICOM files...")
        f=0
        for dcm_file in files:
            f=f+1
            ds = pydicom.dcmread(dcm_file, force=True)
            # HU conversion
            arr = ds.pixel_array.astype(np.float32)
            arr = arr * ds.RescaleSlope + ds.RescaleIntercept
            # print(f"Rescale {ds.RescaleSlope}, Intercept {ds.RescaleIntercept}")
            # Windowing
            arr = np.clip(arr, WINDOW_MIN, WINDOW_MAX)
            arr = ((arr - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN) * 255.0
                  ).astype(np.uint8)

            # print(f"window interval: [{WINDOW_MIN}, {WINDOW_MAX}]")
            # print(f"image max: {arr.max()}, min: {arr.min()}, shape: {arr.shape}")
            # Resize & save
            if arr.shape[0] != IMAGE_SIZE or arr.shape[1] != IMAGE_SIZE:
                print(f"Resizing image from {arr.shape} to ({IMAGE_SIZE}, {IMAGE_SIZE})")
                img = Image.fromarray(arr)
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            else:
                img = Image.fromarray(arr)
                # print(f" Image size is {arr.shape}, no resize needed.")
            # Construct filename
            base = f"{scan.patient_id}_{os.path.basename(dcm_file)}"
            print(f"dcm_file: {dcm_file}")
            print(f"basename: {os.path.basename(dcm_file)}")
            print(f"base: {base}")
            # print(f"scan.id: {scan.id}, path base: {os.path.basename(dcm_file)}")
            # print(f"base: {base}")
            png_path = os.path.join(out_dir, base.replace(".dcm", ".png"))
            # print(f"Saving to {png_path}")
            img.save(png_path)
            print(f"Saved {f}/{len(files)}: {png_path}")
# 6. Run preprocessing for each split
count = 0
for domain, scans in folders.items():
    count +=1
    if count == 2:
        break
    print(f"Processing {len(scans)} scans for {domain}...")
    process_and_save(scans, domain)
    print(f"{count}/{len(folders)}")

print("Data preparation complete.")