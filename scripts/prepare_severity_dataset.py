"""
Utility script to prepare severity dataset structure.
This creates the folder structure needed for severity classification.

Folder structure:
dataset/severity/
  ├── train/
  │   ├── Healthy/
  │   ├── Early/
  │   ├── Intermediate/
  │   └── Final/
  ├── val/
  │   ├── Healthy/
  │   ├── Early/
  │   ├── Intermediate/
  │   └── Final/
  └── test/
      ├── Healthy/
      ├── Early/
      ├── Intermediate/
      └── Final/

Severity Classes:
- Healthy: No disease symptoms
- Early: Severity 0.1-3% (first symptoms of disease)
- Intermediate: Severity 3-12% (advanced symptoms)
- Final: Severity ≥12% (very/extremely advanced symptoms)
"""

import os
from pathlib import Path
import shutil


def create_severity_structure():
    """Create the directory structure for severity classification."""
    
    base_dir = Path("dataset/severity")
    splits = ["train", "val", "test"]
    classes = ["Healthy", "Early", "Intermediate", "Final"]
    
    print("="*60)
    print("CREATING SEVERITY DATASET STRUCTURE")
    print("="*60)
    
    for split in splits:
        for cls in classes:
            dir_path = base_dir / split / cls
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"[OK] Created: {dir_path}")
    
    print("\n" + "="*60)
    print("DIRECTORY STRUCTURE CREATED SUCCESSFULLY")
    print("="*60)
    print("\nSeverity Class Definitions:")
    print("  - Healthy: No disease symptoms")
    print("  - Early: Severity 0.1-3% (first symptoms)")
    print("  - Intermediate: Severity 3-12% (advanced symptoms)")
    print("  - Final: Severity >=12% (very/extremely advanced symptoms)")
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Organize your mango images into the appropriate folders")
    print("2. Place images in train/val/test splits (e.g., 70%/15%/15%)")
    print("3. Ensure each severity class has balanced samples if possible")
    print("4. Run: python scripts/train_severity.py")
    print("="*60)


def show_dataset_info():
    """Display information about the current dataset."""
    
    base_dir = Path("dataset/severity")
    
    if not base_dir.exists():
        print("Severity dataset directory does not exist yet.")
        return
    
    print("\n" + "="*60)
    print("CURRENT DATASET STATISTICS")
    print("="*60)
    
    for split in ["train", "val", "test"]:
        split_path = base_dir / split
        if not split_path.exists():
            continue
            
        print(f"\n{split.upper()}:")
        total = 0
        
        for cls in ["Healthy", "Early", "Intermediate", "Final"]:
            cls_path = split_path / cls
            if cls_path.exists():
                count = len([f for f in cls_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                total += count
                print(f"  {cls:15s}: {count:4d} images")
        
        print(f"  {'Total':15s}: {total:4d} images")
    
    print("="*60)


if __name__ == "__main__":
    # Create directory structure
    create_severity_structure()
    
    # Show current dataset info
    show_dataset_info()
