import os
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# =====================
# CONFIG
# =====================
DATA_DIR = Path("dataset/classification")   # has train/val/test
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4
SEED = 42
NUM_CLASSES = 5

# Save as a new file so you don't overwrite older experiments
WEIGHTS_OUT = Path("models/mango_classifier_70_15_15_resnet18.pth")

# Freeze backbone for a few epochs, then unfreeze to fine-tune
FREEZE_EPOCHS = 5

# Early stopping
PATIENCE = 6

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =====================
# REPRODUCIBILITY
# =====================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # for determinism (esp. if you use CUDA later)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


set_seed(SEED)


# =====================
# DEVICE
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =====================
# TRANSFORMS
# =====================
# train_tfms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(0.1, 0.1, 0.1),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.RandomRotation(12)], p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.25, 0.25, 0.15, 0.05)], p=0.8),
    transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.25, p=1.0)], p=0.25),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
])
# The train_tfms include image augmentation techniques such as:
# RandomResizedCrop
# RandomHorizontalFlip
# RandomRotation
# ColorJitter


val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# =====================
# DATASETS
# =====================
train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfms)
val_ds   = datasets.ImageFolder(DATA_DIR / "val", transform=val_tfms)

print("Class mapping:", train_ds.class_to_idx)

# class counts for weighting
train_labels = [y for _, y in train_ds.samples]
counts = Counter(train_labels)
print("Train class counts:", counts)

# compute weights: inverse frequency
class_weights = torch.tensor(
    [1.0 / counts[i] for i in range(NUM_CLASSES)],
    dtype=torch.float
).to(device)

print("Class weights:", class_weights.detach().cpu().numpy())


# =====================
# DATALOADERS
# =====================
g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    worker_init_fn=seed_worker,
    generator=g
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =====================
# MODEL
# =====================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.to(device)


# Freeze backbone initially
def set_backbone_trainable(trainable: bool):
    for name, param in model.named_parameters():
        # Keep classifier trainable always
        if name.startswith("fc."):
            param.requires_grad = True
        else:
            param.requires_grad = trainable


set_backbone_trainable(False)
print(f"Backbone frozen for first {FREEZE_EPOCHS} epochs.")


# =====================
# LOSS / OPTIM / SCHED
# =====================
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)


# =====================
# TRAIN LOOP
# =====================
Path("models").mkdir(exist_ok=True)

best_acc = 0.0
best_epoch = 0
no_improve = 0

for epoch in range(1, EPOCHS + 1):

    # Unfreeze backbone after FREEZE_EPOCHS
    if epoch == FREEZE_EPOCHS + 1:
        set_backbone_trainable(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )
        print("✅ Backbone unfrozen (fine-tuning all layers).")

    # ---- TRAIN ----
    model.train()
    train_correct, train_total = 0, 0
    train_loss_sum = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        train_correct += (preds == y).sum().item()
        train_total += y.size(0)

    train_acc = train_correct / train_total
    train_loss = train_loss_sum / train_total

    # ---- VAL ----
    model.eval()
    val_correct, val_total = 0, 0
    val_loss_sum = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            val_loss_sum += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    val_acc = val_correct / val_total
    val_loss = val_loss_sum / val_total

    # step scheduler on val accuracy
    scheduler.step(val_acc)

    lr_now = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch:02d} | lr={lr_now:.2e} | "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    # ---- SAVE BEST ----
    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch
        no_improve = 0

        torch.save({
            "model_state": model.state_dict(),
            "class_to_idx": train_ds.class_to_idx,
            "arch": "resnet18",
            "img_size": 224,
            "split": "70_15_15",
            "seed": SEED
        }, WEIGHTS_OUT)

        print(f"✅ Model saved -> {WEIGHTS_OUT}")
    else:
        no_improve += 1

    # ---- EARLY STOP ----
    if no_improve >= PATIENCE:
        print(f"⏹ Early stopping (no improvement for {PATIENCE} epochs).")
        break


print(f"\nBest validation accuracy: {best_acc:.4f} (epoch {best_epoch})")
print(f"Saved best model at: {WEIGHTS_OUT}")
