import os
import shutil
import random
from pathlib import Path

SOURCE_DIR = Path("dataset/public_dataset")
TARGET_DIR = Path("dataset/classification")
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)

for class_name in os.listdir(SOURCE_DIR):
    class_path = SOURCE_DIR / class_name
    if not class_path.is_dir():
        continue

    images = [p for p in class_path.iterdir() if p.suffix.lower() in IMG_EXTS]
    random.shuffle(images)

    n_total = len(images)
    if n_total == 0:
        print(f"⚠️ {class_name}: no images found, skipping")
        continue

    n_train = int(n_total * SPLIT["train"])
    n_val   = int(n_total * SPLIT["val"])

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split_name, split_images in splits.items():
        dest_dir = TARGET_DIR / split_name / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for img_path in split_images:
            shutil.copy2(img_path, dest_dir / img_path.name)

    print(f"{class_name}: {n_total} → "
          f"{len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")

print("\n✅ Split complete.")
