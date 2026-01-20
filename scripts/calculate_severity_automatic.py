"""
Automatic severity calculation using image segmentation algorithms.
Based on the paper: "Mango fruit diseases severity estimation based on image segmentation and deep learning"

Algorithm:
1. Color space segmentation → Get total fruit area (Sfruit)
2. Image thresholding → Get lesion area (Slesion)  
3. Calculate severity = (Slesion / Sfruit) * 100
4. Classify into Early (0.1-3%), Intermediate (3-12%), Final (≥12%)
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
import csv
from tqdm import tqdm

# Paths
PUBLIC_DATASET = Path("dataset/public_dataset")
SEVERITY_DATASET = Path("dataset/severity")

# Threshold value from the paper
THRESHOLD_VALUE = 120

# Severity class boundaries (from the paper)
EARLY_MIN = 0.1
EARLY_MAX = 3.0
INTERMEDIATE_MAX = 12.0


def segment_fruit_by_color_space(image):
    """
    Step 1: Color space segmentation to extract fruit area.
    Returns binary image where fruit=black, background=white
    """
    # Convert to different color spaces for better segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for mango colors (yellow/orange/red tones)
    # Adjust these ranges based on your images
    lower_mango = np.array([10, 30, 30])
    upper_mango = np.array([180, 255, 255])
    
    # Create mask
    mask = cv2.inRange(hsv, lower_mango, upper_mango)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Invert: fruit should be black (0), background white (255)
    fruit_binary = cv2.bitwise_not(mask)
    
    return fruit_binary


def calculate_lesion_area(image, fruit_mask):
    """
    Step 2: Image thresholding to calculate lesion area.
    Uses threshold value of 120 as per the paper.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding (threshold = 120 from paper)
    # Pixels < 120 = black (lesions + background)
    # Pixels >= 120 = white (healthy parts)
    _, thresh = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    
    # Apply fruit mask to exclude background
    # Only consider pixels within the fruit area
    thresh_masked = cv2.bitwise_and(thresh, thresh, mask=cv2.bitwise_not(fruit_mask))
    
    return thresh_masked


def calculate_severity(image_path, debug=False):
    """
    Calculate disease severity for a single image.
    
    Returns:
        severity_percent: Disease coverage percentage
        severity_class: 'Early', 'Intermediate', or 'Final'
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None
    
    # Step 1: Segment fruit to get total fruit area
    fruit_binary = segment_fruit_by_color_space(image)
    
    # Step 2: Count total fruit pixels (Sfruit)
    Sfruit = np.sum(fruit_binary == 0)  # Black pixels = fruit
    
    if Sfruit == 0:
        return None, None  # No fruit detected
    
    # Step 3: Calculate lesion area using thresholding
    thresh_result = calculate_lesion_area(image, fruit_binary)
    
    # Step 4: Count white pixels (healthy parts) within fruit area
    SFwhite = np.sum(thresh_result == 255)
    
    # Step 5: Calculate lesion pixels
    # Slesion = Sfruit - SFwhite
    Slesion = Sfruit - SFwhite
    
    # Step 6: Calculate severity percentage
    # Severity = (Slesion / Sfruit) * 100
    severity_percent = (Slesion / Sfruit) * 100
    
    # Step 7: Classify into severity class
    if severity_percent < EARLY_MIN:
        severity_class = 'Healthy'  # Essentially no disease
    elif EARLY_MIN <= severity_percent < EARLY_MAX:
        severity_class = 'Early'
    elif EARLY_MAX <= severity_percent < INTERMEDIATE_MAX:
        severity_class = 'Intermediate'
    else:  # >= 12%
        severity_class = 'Final'
    
    # Debug visualization
    if debug:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(fruit_binary, cmap='gray')
        axes[0, 1].set_title(f'Fruit Segmentation\nSfruit={Sfruit} pixels')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(thresh_result, cmap='gray')
        axes[1, 0].set_title(f'Lesion Detection\nSlesion={Slesion} pixels')
        axes[1, 0].axis('off')
        
        axes[1, 1].text(0.5, 0.5, 
                       f'Severity: {severity_percent:.2f}%\nClass: {severity_class}',
                       ha='center', va='center', fontsize=16, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return severity_percent, severity_class


def process_all_images(create_csv=True):
    """
    Process all diseased images and automatically calculate severity.
    """
    print("="*60)
    print("AUTOMATIC SEVERITY CALCULATION")
    print("="*60)
    print("Using image segmentation algorithms from the paper:")
    print("1. Color space segmentation for fruit area")
    print("2. Image thresholding for lesion area")
    print("3. Severity = (Lesion area / Fruit area) * 100")
    print("="*60 + "\n")
    
    diseases = ["Anthracnose", "Bacterial_Canker", "Scab", "Stem_End_Rot"]
    
    results = []
    failed = []
    
    # Process all diseased images
    for disease in diseases:
        disease_path = PUBLIC_DATASET / disease
        if not disease_path.exists():
            continue
        
        print(f"\nProcessing {disease}...")
        images = list(disease_path.glob("*.jpg"))
        
        for img_path in tqdm(images, desc=disease):
            severity_percent, severity_class = calculate_severity(img_path)
            
            if severity_percent is None:
                failed.append(str(img_path))
                continue
            
            results.append({
                'disease': disease,
                'filename': img_path.name,
                'severity_percent': severity_percent,
                'severity_class': severity_class
            })
    
    print(f"\n[OK] Processed {len(results)} images successfully")
    if failed:
        print(f"[WARNING] Failed to process {len(failed)} images")
    
    # Show severity distribution
    print("\n" + "="*60)
    print("SEVERITY DISTRIBUTION")
    print("="*60)
    
    from collections import Counter
    severity_counts = Counter([r['severity_class'] for r in results])
    
    for severity in ['Healthy', 'Early', 'Intermediate', 'Final']:
        count = severity_counts.get(severity, 0)
        percent = (count / len(results) * 100) if results else 0
        print(f"{severity:15s}: {count:4d} images ({percent:.1f}%)")
    
    # Save to CSV
    if create_csv:
        csv_file = Path("severity_calculated.csv")
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['disease', 'filename', 'severity_percent', 'severity_class'])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n[OK] Results saved to: {csv_file}")
    
    return results


def organize_by_calculated_severity(results, split_ratio=(0.70, 0.15, 0.15)):
    """
    Organize images into severity dataset based on calculated severity.
    """
    print("\n" + "="*60)
    print("ORGANIZING IMAGES BY CALCULATED SEVERITY")
    print("="*60)
    
    from collections import defaultdict
    import random
    
    # Group by severity class
    severity_groups = defaultdict(list)
    for result in results:
        severity_groups[result['severity_class']].append(result)
    
    # Split each severity class
    for severity_class, images in severity_groups.items():
        random.shuffle(images)
        
        n = len(images)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]
        
        # Copy to train
        for img_data in train_imgs:
            src = PUBLIC_DATASET / img_data['disease'] / img_data['filename']
            dst = SEVERITY_DATASET / 'train' / severity_class / img_data['filename']
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        
        # Copy to val
        for img_data in val_imgs:
            src = PUBLIC_DATASET / img_data['disease'] / img_data['filename']
            dst = SEVERITY_DATASET / 'val' / severity_class / img_data['filename']
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        
        # Copy to test
        for img_data in test_imgs:
            src = PUBLIC_DATASET / img_data['disease'] / img_data['filename']
            dst = SEVERITY_DATASET / 'test' / severity_class / img_data['filename']
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        
        print(f"{severity_class:15s}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
    
    print("\n[OK] Images organized!")


def show_final_summary():
    """Show final dataset summary."""
    print("\n" + "="*60)
    print("FINAL SEVERITY DATASET SUMMARY")
    print("="*60)
    
    for split in ["train", "val", "test"]:
        split_path = SEVERITY_DATASET / split
        print(f"\n{split.upper()}:")
        
        for severity in ["Healthy", "Early", "Intermediate", "Final"]:
            severity_path = split_path / severity
            if severity_path.exists():
                count = len(list(severity_path.glob("*.jpg"))) + len(list(severity_path.glob("*.png")))
                print(f"  {severity:15s}: {count:4d} images")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Step 1: Calculate severity for all images
    results = process_all_images(create_csv=True)
    
    # Step 2: Organize images by calculated severity
    if results:
        organize_by_calculated_severity(results)
        
        # Step 3: Show final summary
        show_final_summary()
        
        print("\n" + "="*60)
        print("READY TO TRAIN!")
        print("="*60)
        print("Run: python scripts/train_severity.py")
        print("="*60)
