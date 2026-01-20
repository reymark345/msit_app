"""
Apply severity labels from the CSV template to organize images.
"""

import csv
import shutil
from pathlib import Path

PUBLIC_DATASET = Path("dataset/public_dataset")
SEVERITY_DATASET = Path("dataset/severity")
csv_file = Path("severity_labeling_template.csv")

if not csv_file.exists():
    print(f"Error: {csv_file} not found!")
    print("Please run: python scripts/create_severity_template.py first")
    exit(1)

print("Reading labels from CSV...")

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Check if CSV is filled
unfilled = [r for r in rows if not r['Severity (Empty/Early/Intermediate/Final)'].strip() or not r['Split (train/val/test)'].strip()]
if unfilled:
    print(f"Warning: {len(unfilled)} rows are not filled in the CSV!")
    print(f"Please fill in Severity and Split for all images.")
    exit(1)

print(f"Processing {len(rows)} images...")

copied = 0
errors = 0

for row in rows:
    disease = row['Disease']
    filename = row['Filename']
    severity = row['Severity (Empty/Early/Intermediate/Final)'].strip()
    split = row['Split (train/val/test)'].strip()
    
    # Validate severity
    if severity not in ['Early', 'Intermediate', 'Final']:
        print(f"Warning: Invalid severity '{severity}' for {filename}, skipping...")
        errors += 1
        continue
    
    # Validate split
    if split not in ['train', 'val', 'test']:
        print(f"Warning: Invalid split '{split}' for {filename}, skipping...")
        errors += 1
        continue
    
    source = PUBLIC_DATASET / disease / filename
    dest = SEVERITY_DATASET / split / severity / filename
    
    if not source.exists():
        print(f"Warning: {source} not found, skipping...")
        errors += 1
        continue
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    copied += 1

print(f"\n[OK] Organized {copied} images")
if errors > 0:
    print(f"[WARNING] {errors} errors occurred")

# Show summary
print("\nSEVERITY DATASET SUMMARY:")
for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}:")
    for severity in ["Healthy", "Early", "Intermediate", "Final"]:
        severity_path = SEVERITY_DATASET / split / severity
        if severity_path.exists():
            count = len(list(severity_path.glob("*.jpg"))) + len(list(severity_path.glob("*.png")))
            print(f"  {severity:15s}: {count:4d} images")

print("\nReady to train! Run:")
print("python scripts/train_severity.py")
