"""
Create a CSV template to help organize diseased images by severity level.
"""

import csv
from pathlib import Path

PUBLIC_DATASET = Path("dataset/public_dataset")
output_file = Path("severity_labeling_template.csv")

print("Creating severity labeling template...")

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Disease', 'Filename', 'Severity (Empty/Early/Intermediate/Final)', 'Split (train/val/test)'])
    
    diseases = ["Anthracnose", "Bacterial_Canker", "Scab", "Stem_End_Rot"]
    
    for disease in diseases:
        disease_path = PUBLIC_DATASET / disease
        if disease_path.exists():
            for img in sorted(disease_path.glob("*.jpg")):
                # Leave severity and split empty for user to fill
                writer.writerow([disease, img.name, '', ''])

print(f"\n[OK] Template created: {output_file}")
print(f"     Total images to label: {sum(len(list((PUBLIC_DATASET / d).glob('*.jpg'))) for d in diseases if (PUBLIC_DATASET / d).exists())}")
print("\nNEXT STEPS:")
print("1. Open 'severity_labeling_template.csv' in Excel")
print("2. For each image, fill in:")
print("   - Severity: Early / Intermediate / Final")
print("   - Split: train / val / test")
print("   (Suggested split: 70% train, 15% val, 15% test)")
print("3. Save the file")
print("4. Run: python scripts/apply_severity_labels.py")
