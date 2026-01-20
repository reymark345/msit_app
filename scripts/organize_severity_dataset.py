"""
Script to help organize images from disease-type folders into severity-level folders.

This script will:
1. Automatically copy Healthy images to severity dataset
2. Help you manually categorize diseased images by severity level
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

# Paths
PUBLIC_DATASET = Path("dataset/public_dataset")
SEVERITY_DATASET = Path("dataset/severity")

# Severity class definitions
SEVERITY_INFO = """
SEVERITY LEVELS:
- Healthy: 0% disease coverage (no symptoms)
- Early: 0.1-3% disease coverage (few small spots)
- Intermediate: 3-12% disease coverage (multiple spots, visible damage)
- Final: >=12% disease coverage (large lesions, severe damage)
"""


def copy_healthy_images():
    """Copy all healthy images to severity dataset."""
    
    print("="*60)
    print("STEP 1: COPYING HEALTHY IMAGES")
    print("="*60)
    
    source_healthy = PUBLIC_DATASET / "Healthy"
    
    if not source_healthy.exists():
        print(f"Error: {source_healthy} not found!")
        return
    
    healthy_images = list(source_healthy.glob("*.jpg")) + list(source_healthy.glob("*.png"))
    print(f"Found {len(healthy_images)} healthy images")
    
    # Split: 70% train, 15% val, 15% test
    random.shuffle(healthy_images)
    n_train = int(len(healthy_images) * 0.70)
    n_val = int(len(healthy_images) * 0.15)
    
    train_images = healthy_images[:n_train]
    val_images = healthy_images[n_train:n_train+n_val]
    test_images = healthy_images[n_train+n_val:]
    
    # Copy to train
    dest_train = SEVERITY_DATASET / "train" / "Healthy"
    dest_train.mkdir(parents=True, exist_ok=True)
    for img in train_images:
        shutil.copy2(img, dest_train / img.name)
    print(f"Copied {len(train_images)} images to train/Healthy/")
    
    # Copy to val
    dest_val = SEVERITY_DATASET / "val" / "Healthy"
    dest_val.mkdir(parents=True, exist_ok=True)
    for img in val_images:
        shutil.copy2(img, dest_val / img.name)
    print(f"Copied {len(val_images)} images to val/Healthy/")
    
    # Copy to test
    dest_test = SEVERITY_DATASET / "test" / "Healthy"
    dest_test.mkdir(parents=True, exist_ok=True)
    for img in test_images:
        shutil.copy2(img, dest_test / img.name)
    print(f"Copied {len(test_images)} images to test/Healthy/")
    
    print(f"\n[OK] Healthy images organized successfully!")


def show_sample_images():
    """Show sample images from each disease type to help with categorization."""
    
    print("\n" + "="*60)
    print("STEP 2: SAMPLE IMAGES FROM EACH DISEASE")
    print("="*60)
    print("This will help you understand the severity levels in your dataset.")
    
    diseases = ["Anthracnose", "Bacterial_Canker", "Scab", "Stem_End_Rot"]
    
    for disease in diseases:
        disease_path = PUBLIC_DATASET / disease
        if not disease_path.exists():
            continue
        
        images = list(disease_path.glob("*.jpg"))[:6]  # Show 6 samples
        
        if not images:
            continue
        
        print(f"\n{disease}: {len(list(disease_path.glob('*.jpg')))} total images")
        
        # Display images
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(f'{disease} - Sample Images', fontsize=14, fontweight='bold')
        
        for idx, img_path in enumerate(images):
            img = Image.open(img_path)
            ax = axes[idx // 3, idx % 3]
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(img_path.name, fontsize=8)
        
        plt.tight_layout()
        plt.show()


def create_manual_organization_guide():
    """Create a guide for manually organizing images."""
    
    print("\n" + "="*60)
    print("STEP 3: MANUAL ORGANIZATION GUIDE")
    print("="*60)
    
    print(SEVERITY_INFO)
    
    print("\nMANUAL STEPS TO ORGANIZE YOUR IMAGES:")
    print("-" * 60)
    
    diseases = ["Anthracnose", "Bacterial_Canker", "Scab", "Stem_End_Rot"]
    
    for disease in diseases:
        disease_path = PUBLIC_DATASET / disease
        if disease_path.exists():
            count = len(list(disease_path.glob("*.jpg")))
            print(f"\n{disease} ({count} images):")
            print(f"  1. Open folder: {disease_path}")
            print(f"  2. Look at each image and estimate disease coverage:")
            print(f"     - Early (0.1-3%): Few small spots")
            print(f"     - Intermediate (3-12%): Multiple spots, visible damage")
            print(f"     - Final (>=12%%): Large lesions, severe damage")
            print(f"  3. Copy images to:")
            print(f"     - dataset/severity/train/Early/")
            print(f"     - dataset/severity/train/Intermediate/")
            print(f"     - dataset/severity/train/Final/")
            print(f"  4. Split: 70% train, 15% val, 15% test")


def count_severity_dataset():
    """Count images in severity dataset."""
    
    print("\n" + "="*60)
    print("CURRENT SEVERITY DATASET STATUS")
    print("="*60)
    
    for split in ["train", "val", "test"]:
        split_path = SEVERITY_DATASET / split
        print(f"\n{split.upper()}:")
        
        for severity in ["Healthy", "Early", "Intermediate", "Final"]:
            severity_path = split_path / severity
            if severity_path.exists():
                count = len(list(severity_path.glob("*.jpg"))) + len(list(severity_path.glob("*.png")))
                print(f"  {severity:15s}: {count:4d} images")


def create_organization_template():
    """Create a CSV template for organizing images."""
    
    print("\n" + "="*60)
    print("CREATING ORGANIZATION TEMPLATE")
    print("="*60)
    
    import csv
    
    output_file = Path("severity_organization_template.csv")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Disease', 'Filename', 'Severity', 'Split'])
        
        diseases = ["Anthracnose", "Bacterial_Canker", "Scab", "Stem_End_Rot"]
        
        for disease in diseases:
            disease_path = PUBLIC_DATASET / disease
            if disease_path.exists():
                for img in disease_path.glob("*.jpg"):
                    # Leave severity and split empty for user to fill
                    writer.writerow([disease, img.name, '', ''])
    
    print(f"[OK] Created template: {output_file}")
    print(f"\nYou can fill in the 'Severity' and 'Split' columns, then use")
    print(f"this CSV to automatically organize images.")


def organize_from_csv(csv_file):
    """Organize images based on a filled CSV template."""
    
    import csv
    import pandas as pd
    
    print("\n" + "="*60)
    print("ORGANIZING FROM CSV")
    print("="*60)
    
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found!")
        return
    
    df = pd.read_csv(csv_file)
    
    # Validate
    if df['Severity'].isna().any() or df['Split'].isna().any():
        print("Error: CSV contains empty Severity or Split values!")
        print("Please fill in all rows before running this.")
        return
    
    copied = 0
    for _, row in df.iterrows():
        disease = row['Disease']
        filename = row['Filename']
        severity = row['Severity']
        split = row['Split']
        
        source = PUBLIC_DATASET / disease / filename
        dest = SEVERITY_DATASET / split / severity / filename
        
        if source.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            copied += 1
    
    print(f"[OK] Organized {copied} images based on CSV")


if __name__ == "__main__":
    print("="*60)
    print("MANGO SEVERITY DATASET ORGANIZATION TOOL")
    print("="*60)
    print(SEVERITY_INFO)
    
    # Step 1: Copy healthy images
    copy_healthy_images()
    
    # Step 2: Show current status
    count_severity_dataset()
    
    # Step 3: Create guide
    print("\n" + "="*60)
    print("NEXT STEPS FOR DISEASED IMAGES")
    print("="*60)
    print("\nOption 1: MANUAL ORGANIZATION (Recommended)")
    print("  - Look at each diseased image")
    print("  - Estimate disease coverage percentage")
    print("  - Copy to appropriate severity folder")
    print("  - This gives you the most control and accuracy")
    
    print("\nOption 2: USE CSV TEMPLATE")
    print("  - Run with --create-template flag")
    print("  - Fill in severity and split in Excel/CSV editor")
    print("  - Run with --from-csv flag to auto-organize")
    
    print("\n" + "="*60)
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. Show sample images from each disease")
    print("2. Create CSV template for organization")
    print("3. Exit (organize manually)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        show_sample_images()
        create_manual_organization_guide()
    elif choice == "2":
        create_organization_template()
        print("\nFill in the CSV, then run:")
        print("python scripts/organize_severity_dataset.py --from-csv severity_organization_template.csv")
    else:
        print("\n[OK] Please organize diseased images manually.")
        create_manual_organization_guide()
    
    print("\n" + "="*60)
    print("Once images are organized, train with:")
    print("python scripts/train_severity.py")
    print("="*60)
