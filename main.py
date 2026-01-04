from flask import Flask, request, jsonify
from ultralytics import YOLO
import torch
import torchvision.transforms as T
from PIL import Image
import io

app = Flask(__name__)

# --- Load models once (startup) ---
yolo_model = YOLO("weights/yolov8_cbam.pt")

# Example: PyTorch CNN classifier
cnn_model = torch.jit.load("weights/cnn_disease.pt")  # or torch.load(...) depending on how you saved it
cnn_model.eval()

CLASS_NAMES = ["Healthy", "Anthracnose", "Bacterial Canker", "Mango Scab", "Stem-End Rot"]

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use multipart form field name 'image'."}), 400

    img_bytes = request.files["image"].read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # --- Stage 1: YOLO detection ---
    yolo_results = yolo_model.predict(img, conf=0.25, iou=0.5, verbose=False)
    r = yolo_results[0]

    detections = []
    if r.boxes is None or len(r.boxes) == 0:
        return jsonify({"detections": [], "message": "No mango detected."})

    # Each box: xyxy
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])

        # Crop fruit region
        crop = img.crop((x1, y1, x2, y2))

        # --- Stage 2: CNN disease classification ---
        x = transform(crop).unsqueeze(0)
        with torch.no_grad():
            logits = cnn_model(x)
            probs = torch.softmax(logits, dim=1)[0]
            cls_id = int(torch.argmax(probs).item())
            cls_conf = float(probs[cls_id].item())

        detections.append({
            "bbox_xyxy": [x1, y1, x2, y2],
            "det_conf": conf,
            "disease": CLASS_NAMES[cls_id],
            "disease_conf": cls_conf
        })

    return jsonify({"detections": detections})
