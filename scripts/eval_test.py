import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

DATA_DIR = Path("dataset/classification")
WEIGHTS = Path("models/mango_classifier.pth")
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=test_tfms)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

ckpt = torch.load(WEIGHTS, map_location=device)
class_to_idx = ckpt["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(ckpt["model_state"])
model.eval().to(device)

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

acc = (y_true == y_pred).mean()
print(f"\n✅ TEST accuracy: {acc:.4f} ({acc*100:.2f}%)\n")

target_names = [idx_to_class[i] for i in range(num_classes)]
print("Classification report:\n")
print(classification_report(y_true, y_pred, target_names=target_names))

print("Confusion matrix (rows=true, cols=pred):\n")
print(confusion_matrix(y_true, y_pred))
