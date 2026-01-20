"""
Test multiple images at once and save results to CSV.
"""

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import csv

# Configuration
MODEL_PATH = "models/mango_severity_resnet50.pth"
TEST_FOLDER = "dataset/severity/test"  # Or any folder with your test images
OUTPUT_CSV = "severity_test_results.csv"

# Load model
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device)

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()

# Get class names
class_to_idx = checkpoint['class_to_idx']
idx_to_class = {v: k for k, v in class_to_idx.items()}

print(f"Model loaded! Classes: {list(class_to_idx.keys())}\n")

# Prepare transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Find all images
test_path = Path(TEST_FOLDER)
image_files = []
for ext in ['*.jpg', '*.jpeg', '*.png']:
    image_files.extend(test_path.rglob(ext))

print(f"Found {len(image_files)} images to test\n")

# Test all images
results = []
correct = 0
total = 0

for img_path in image_files:
    # Get true label from folder name (if organized by severity)
    true_label = img_path.parent.name if img_path.parent.name in class_to_idx else "Unknown"
    
    # Load and predict
    try:
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
        
        predicted_class = idx_to_class[pred_idx]
        
        # Check if correct
        is_correct = (predicted_class == true_label)
        if true_label != "Unknown":
            total += 1
            if is_correct:
                correct += 1
        
        results.append({
            'filename': img_path.name,
            'true_label': true_label,
            'predicted': predicted_class,
            'confidence': f"{confidence*100:.2f}%",
            'correct': 'Yes' if is_correct else 'No'
        })
        
        print(f"✓ {img_path.name}: {predicted_class} ({confidence*100:.1f}%)")
        
    except Exception as e:
        print(f"✗ Failed to process {img_path.name}: {e}")

# Save results to CSV
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['filename', 'true_label', 'predicted', 'confidence', 'correct'])
    writer.writeheader()
    writer.writerows(results)

print(f"\n[OK] Results saved to: {OUTPUT_CSV}")

# Show accuracy if we have true labels
if total > 0:
    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2f}%")
