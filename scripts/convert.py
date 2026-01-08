from pathlib import Path
from PIL import Image
import pillow_heif

# Register HEIF/HEIC support
pillow_heif.register_heif_opener()

# Paths
INPUT_ROOT = Path("dataset/collected_dataset/raw_images")
OUTPUT_ROOT = Path("dataset/collected_dataset/output_jpg")

# JPG quality (good for disease texture)
JPG_QUALITY = 95

# Find all HEIC files recursively
heic_files = list(INPUT_ROOT.rglob("*.heic")) + list(INPUT_ROOT.rglob("*.HEIC"))

if not heic_files:
    print("❌ No HEIC files found. Check your input directory.")
    exit()

for heic_path in heic_files:
    # Relative path from raw_images
    rel_path = heic_path.relative_to(INPUT_ROOT)

    # Create corresponding output folder
    out_dir = OUTPUT_ROOT / rel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output JPG path
    out_path = out_dir / (heic_path.stem + ".jpg")

    # Open and convert
    img = Image.open(heic_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    img.save(
        out_path,
        format="JPEG",
        quality=JPG_QUALITY,
        subsampling=0,
        optimize=True
    )

    print(f"✅ Converted: {rel_path} → {out_path.relative_to(OUTPUT_ROOT)}")

print(f"\n🎉 Done. Converted {len(heic_files)} file(s).")
