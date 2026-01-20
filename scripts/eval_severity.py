import os
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_fscore_support,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns


# =====================
# CONFIG
# =====================
DATA_DIR = Path("dataset/severity")
TEST_DIR = DATA_DIR / "test"
MODEL_PATH = Path("models/mango_severity_resnet50.pth")
BATCH_SIZE = 16
NUM_CLASSES = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# =====================
# TRANSFORMS
# =====================
test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# =====================
# LOAD TEST DATA
# =====================
test_ds = datasets.ImageFolder(TEST_DIR, transform=test_tfms)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Test dataset size: {len(test_ds)} images")
print(f"Classes: {test_ds.classes}")
print(f"Class to index: {test_ds.class_to_idx}\n")


# =====================
# LOAD MODEL
# =====================
print(f"Loading model from: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Build model architecture
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, NUM_CLASSES)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

print(f"Model architecture: {checkpoint.get('arch', 'resnet50')}")
print(f"Training accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
print(f"Trained at epoch: {checkpoint.get('epoch', 'N/A')}\n")

# Get severity class definitions
severity_defs = checkpoint.get('severity_classes', {
    "Healthy": "No disease symptoms",
    "Early": "Severity 0.1-3%",
    "Intermediate": "Severity 3-12%",
    "Final": "Severity ≥12%"
})

print("="*60)
print("SEVERITY CLASS DEFINITIONS")
print("="*60)
for cls, desc in severity_defs.items():
    print(f"  {cls}: {desc}")
print("="*60 + "\n")


# =====================
# EVALUATE
# =====================
all_preds = []
all_labels = []
all_probs = []

print("Evaluating on test set...")
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)


# =====================
# METRICS
# =====================
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

# Overall metrics
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, support = precision_recall_fscore_support(
    all_labels, all_preds, average='weighted', zero_division=0
)

print(f"\nOverall Metrics:")
print(f"  Accuracy:  {accuracy * 100:.2f}%")
print(f"  Precision: {precision * 100:.2f}%")
print(f"  Recall:    {recall * 100:.2f}%")
print(f"  F1-Score:  {f1 * 100:.2f}%")

# Per-class metrics
print(f"\nPer-Class Metrics:")
print("="*60)
precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
    all_labels, all_preds, average=None, zero_division=0
)

for i, class_name in enumerate(test_ds.classes):
    print(f"\n{class_name}:")
    print(f"  Samples:   {support_per_class[i]}")
    print(f"  Precision: {precision_per_class[i] * 100:.2f}%")
    print(f"  Recall:    {recall_per_class[i] * 100:.2f}%")
    print(f"  F1-Score:  {f1_per_class[i] * 100:.2f}%")


# =====================
# CLASSIFICATION REPORT
# =====================
print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT")
print("="*60)
print(classification_report(
    all_labels, 
    all_preds, 
    target_names=test_ds.classes,
    digits=4,
    zero_division=0
))


# =====================
# CONFUSION MATRIX
# =====================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=test_ds.classes,
    yticklabels=test_ds.classes
)
plt.title('Confusion Matrix - Severity Classification', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()

# Save confusion matrix
cm_path = Path("models/severity_confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Confusion matrix saved to: {cm_path}")
plt.close()


# =====================
# PER-CLASS ACCURACY VISUALIZATION
# =====================
plt.figure(figsize=(12, 6))

# Plot 1: Accuracy per class
plt.subplot(1, 2, 1)
accuracies = []
for i in range(len(test_ds.classes)):
    class_mask = all_labels == i
    class_acc = (all_preds[class_mask] == all_labels[class_mask]).mean() * 100
    accuracies.append(class_acc)

colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
bars = plt.bar(test_ds.classes, accuracies, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('Accuracy (%)', fontsize=11)
plt.title('Per-Class Accuracy', fontsize=12, fontweight='bold')
plt.ylim([0, 105])
plt.xticks(rotation=15, ha='right')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: F1-Score per class
plt.subplot(1, 2, 2)
bars = plt.bar(test_ds.classes, f1_per_class * 100, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('F1-Score (%)', fontsize=11)
plt.title('Per-Class F1-Score', fontsize=12, fontweight='bold')
plt.ylim([0, 105])
plt.xticks(rotation=15, ha='right')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
metrics_path = Path("models/severity_metrics.png")
plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
print(f"✅ Metrics visualization saved to: {metrics_path}")
plt.close()


# =====================
# SUMMARY
# =====================
print("\n" + "="*60)
print("EVALUATION SUMMARY")
print("="*60)
print(f"Total test samples: {len(all_labels)}")
print(f"Correctly classified: {(all_preds == all_labels).sum()}")
print(f"Misclassified: {(all_preds != all_labels).sum()}")
print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
print(f"Final Test Precision: {precision * 100:.2f}%")
print(f"Final Test F1-Score: {f1 * 100:.2f}%")
print("="*60)

# Compare with study results
print("\n" + "="*60)
print("COMPARISON WITH REFERENCE STUDY")
print("="*60)
print("Reference Study (Mango Disease Severity Estimation):")
print("  - Accuracy:  97.82%")
print("  - Precision: 97.09%")
print("  - F1-Score:  97.79%")
print(f"\nYour Model:")
print(f"  - Accuracy:  {accuracy * 100:.2f}%")
print(f"  - Precision: {precision * 100:.2f}%")
print(f"  - F1-Score:  {f1 * 100:.2f}%")
print("="*60)
