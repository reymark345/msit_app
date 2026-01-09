from ultralytics import YOLO

# Load YOLOv8 nano (lightweight, good for CPU)
model = YOLO("yolov8n.pt")

# Train on your Roboflow dataset
model.train(
    data="dataset/detection/data.yaml",
    imgsz=640,
    epochs=100,
    batch=8,
    device="cpu"  # change to 0 if you have CUDA GPU
)
