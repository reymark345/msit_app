from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

model.predict(
    source="source/collected_dataset/output_jpg/Occluded/IMG_7728.jpg",
    conf=0.25,
    save=True
)
