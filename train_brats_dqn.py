import os
from pathlib import Path

# Your actual BraTS data folders
base_path = "/Users/arunyahooda/Desktop/BT2O23"

print("Exploring your BraTS 2023 data...")
print("="*60)

# Check the Training Data folder
training_path = os.path.join(base_path, "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
if os.path.exists(training_path):
    print(f"✅ Found Training Data folder")
    contents = os.listdir(training_path)
    
    # Find patient folders (they usually start with BraTS-GLI-)
    patient_folders = [f for f in contents if os.path.isdir(os.path.join(training_path, f)) and 'BraTS' in f]
    
    print(f"   Found {len(patient_folders)} patient folders")
    print(f"   First 5 patient IDs: {patient_folders[:5]}")
    
    # Check what's in the first patient folder
    if patient_folders:
        first_patient = patient_folders[0]
        patient_path = os.path.join(training_path, first_patient)
        files = os.listdir(patient_path)
        print(f"\n   Contents of {first_patient}:")
        for f in files:
            print(f"      - {f}")

print("\n" + "="*60)

# Check the actual_train folder
actual_train_path = os.path.join(base_path, "actual_train")
if os.path.exists(actual_train_path):
    print(f"✅ Found actual_train folder")
    contents = os.listdir(actual_train_path)
    print(f"   Contents: {contents}")
    
    # If there are subfolders, check them
    for item in contents:
        item_path = os.path.join(actual_train_path, item)
        if os.path.isdir(item_path):
            sub_contents = os.listdir(item_path)
            print(f"   📁 {item}/: {sub_contents[:5]}")

print("\n" + "="*60)

# Check Training_png folder (converted images)
png_path = os.path.join(base_path, "Training_png")
if os.path.exists(png_path):
    print(f"✅ Found Training_png folder (converted images)")
    contents = os.listdir(png_path)[:10]
    print(f"   Sample contents: {contents}")

print("\n" + "="*60)

# Check CSV files for data organization
csv_files = [f for f in os.listdir(base_path) if f.endswith('.csv')]
print(f"Found CSV files: {csv_files}")

if 'rl_train_t1c_30.csv' in csv_files:
    print("\n✅ Found rl_train_t1c_30.csv - this might contain your 30 training samples!")
    import csv
    csv_path = os.path.join(base_path, 'rl_train_t1c_30.csv')
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        print(f"   CSV has {len(rows)} rows")
        if rows:
            print(f"   Header: {rows[0]}")
            if len(rows) > 1:
                print(f"   First data row: {rows[1]}")

print("\n" + "="*60)
print("📊 SUMMARY:")
print("-" * 60)

# Determine the best approach
if patient_folders:
    print(f"1. You have BraTS 2023 data with {len(patient_folders)} patients")
    print("   These use different naming (BraTS-GLI-XXXXX) than the paper (73, 76, 80)")
    print("\n   RECOMMENDED APPROACH:")
    print("   - Use the first 30 patients from your BraTS 2023 data")
    print("   - Or use the patients listed in rl_train_t1c_30.csv")
    
elif os.path.exists(png_path) and os.listdir(png_path):
    print("2. You have PNG converted images in Training_png/")
    print("\n   RECOMMENDED APPROACH:")
    print("   - Modify the code to load PNG images instead of NIfTI")
    
print("\n" + "="*60)
print("Next steps:")
print("1. The paper used BraTS 2014 data (patients 73, 76, 80)")
print("2. You have BraTS 2023 data (different patient IDs)")
print("3. We need to adapt the code to use your actual data")