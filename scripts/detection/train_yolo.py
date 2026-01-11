from ultralytics import YOLO

# Load YOLOv8 nano with CBAM from custom config
model = YOLO("yolov8_cbam.yaml")  # Custom YAML with CBAM attention modules

# Optionally load pretrained weights from yolov8n.pt for transfer learning
# model.load("yolov8n.pt")

# Train on your Roboflow dataset
model.train(
    data="dataset/detection/data.yaml",
    imgsz=640,
    epochs=100,
    batch=8,
    device="cpu"  # change to 0 if you have CUDA GPU
)
