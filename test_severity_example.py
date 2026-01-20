"""
Simple example script to test severity model on your images.
"""

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# Configuration
MODEL_PATH = "models/mango_severity_resnet50.pth"
# IMAGE_PATH = "images_sample/test_image.jpg"  # Change this to your image path
IMAGE_PATH = "dataset/collected_dataset/output_jpg/Anthracnose/IMG_7869.jpg"  # Change this to your image path

# C:\laragon\www\msit_app\dataset\collected_dataset\output_jpg\Anthracnose\IMG_7703.jpg


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

# Prepare image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and predict
print(f"Testing image: {IMAGE_PATH}")
image = Image.open(IMAGE_PATH).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    confidence = probs[pred_idx].item()

predicted_class = idx_to_class[pred_idx]

print("\n" + "="*60)
print("RESULT")
print("="*60)
print(f"Severity Stage: {predicted_class}")
print(f"Confidence: {confidence*100:.2f}%")
print("\nAll Probabilities:")
for i in range(len(probs)):
    class_name = idx_to_class[i]
    prob = probs[i].item()
    bar = '█' * int(prob * 50)
    print(f"  {class_name:15s} [{bar:<50}] {prob*100:5.1f}%")
print("="*60)
