"""
Flask API for Mango Disease Detection + Severity Estimation
Combines:
1. YOLO detection (detect mango fruit)
2. Disease classification (classify disease type)
3. Severity estimation (estimate disease severity level)
"""

from flask import Flask, request, jsonify
from ultralytics import YOLO
import torch
import torchvision.transforms as T
from torchvision import models
from torch import nn
from PIL import Image
import io
from pathlib import Path

app = Flask(__name__)

# =====================
# LOAD MODELS
# =====================

# 1. YOLO Detection Model
yolo_model = YOLO("models/object/yolov8_cbam.pt")  # Adjust path as needed

# 2. Disease Classification Model (ResNet18)
disease_model_path = Path("models/mango_classifier_70_15_15_resnet18.pth")
if disease_model_path.exists():
    checkpoint = torch.load(disease_model_path, map_location='cpu')
    disease_model = models.resnet18(weights=None)
    disease_model.fc = nn.Linear(disease_model.fc.in_features, 5)  # 5 disease classes
    disease_model.load_state_dict(checkpoint['model_state'])
    disease_model.eval()
    disease_class_to_idx = checkpoint.get('class_to_idx', {})
    disease_idx_to_class = {v: k for k, v in disease_class_to_idx.items()}
    DISEASE_ENABLED = True
    print("✅ Disease classification model loaded")
else:
    disease_model = None
    disease_idx_to_class = {}
    DISEASE_ENABLED = False
    print("⚠️  Disease classification model not found")

# 3. Severity Estimation Model (ResNet50)
severity_model_path = Path("models/mango_severity_resnet50.pth")
if severity_model_path.exists():
    checkpoint = torch.load(severity_model_path, map_location='cpu')
    severity_model = models.resnet50(weights=None)
    num_features = severity_model.fc.in_features
    num_classes = checkpoint.get('num_classes', 4)
    severity_model.fc = nn.Linear(num_features, num_classes)
    severity_model.load_state_dict(checkpoint['model_state'])
    severity_model.eval()
    severity_class_to_idx = checkpoint.get('class_to_idx', {})
    severity_idx_to_class = {v: k for k, v in severity_class_to_idx.items()}
    severity_descriptions = checkpoint.get('severity_classes', {})
    SEVERITY_ENABLED = True
    print("✅ Severity estimation model loaded")
else:
    severity_model = None
    severity_idx_to_class = {}
    severity_descriptions = {}
    SEVERITY_ENABLED = False
    print("⚠️  Severity estimation model not found")


# =====================
# TRANSFORMS
# =====================
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# =====================
# API ENDPOINTS
# =====================

@app.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "models": {
            "detection": "loaded",
            "disease_classification": "loaded" if DISEASE_ENABLED else "not_found",
            "severity_estimation": "loaded" if SEVERITY_ENABLED else "not_found"
        }
    })


@app.post("/predict")
def predict():
    """
    Main prediction endpoint.
    Performs detection, disease classification, and severity estimation.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use multipart form field name 'image'."}), 400

    img_bytes = request.files["image"].read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # =====================
    # STAGE 1: YOLO Detection
    # =====================
    yolo_results = yolo_model.predict(img, conf=0.25, iou=0.5, verbose=False)
    r = yolo_results[0]

    detections = []
    if r.boxes is None or len(r.boxes) == 0:
        return jsonify({"detections": [], "message": "No mango detected."})

    # Process each detected mango
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        det_conf = float(box.conf[0])

        # Crop fruit region
        crop = img.crop((x1, y1, x2, y2))
        x_tensor = transform(crop).unsqueeze(0)

        detection_result = {
            "bbox_xyxy": [x1, y1, x2, y2],
            "det_conf": det_conf
        }

        # =====================
        # STAGE 2: Disease Classification
        # =====================
        if DISEASE_ENABLED and disease_model is not None:
            with torch.no_grad():
                logits = disease_model(x_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                disease_cls_id = int(torch.argmax(probs).item())
                disease_conf = float(probs[disease_cls_id].item())

            detection_result["disease"] = disease_idx_to_class.get(disease_cls_id, "Unknown")
            detection_result["disease_conf"] = disease_conf
        else:
            detection_result["disease"] = "Model not loaded"
            detection_result["disease_conf"] = 0.0

        # =====================
        # STAGE 3: Severity Estimation
        # =====================
        if SEVERITY_ENABLED and severity_model is not None:
            with torch.no_grad():
                logits = severity_model(x_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                severity_cls_id = int(torch.argmax(probs).item())
                severity_conf = float(probs[severity_cls_id].item())

            severity_class = severity_idx_to_class.get(severity_cls_id, "Unknown")
            severity_desc = severity_descriptions.get(severity_class, "N/A")

            detection_result["severity"] = severity_class
            detection_result["severity_conf"] = severity_conf
            detection_result["severity_description"] = severity_desc
        else:
            detection_result["severity"] = "Model not loaded"
            detection_result["severity_conf"] = 0.0
            detection_result["severity_description"] = "N/A"

        detections.append(detection_result)

    return jsonify({"detections": detections})


@app.post("/predict_severity_only")
def predict_severity_only():
    """
    Endpoint for severity estimation only (no detection).
    Useful when you already have a cropped mango image.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use multipart form field name 'image'."}), 400

    if not SEVERITY_ENABLED or severity_model is None:
        return jsonify({"error": "Severity model not loaded"}), 503

    img_bytes = request.files["image"].read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Transform and predict
    x_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = severity_model(x_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        severity_cls_id = int(torch.argmax(probs).item())
        severity_conf = float(probs[severity_cls_id].item())

    severity_class = severity_idx_to_class.get(severity_cls_id, "Unknown")
    severity_desc = severity_descriptions.get(severity_class, "N/A")

    # Get all class probabilities
    all_probs = {severity_idx_to_class.get(i, f"Class_{i}"): float(probs[i].item()) 
                 for i in range(len(probs))}

    return jsonify({
        "severity": severity_class,
        "confidence": severity_conf,
        "description": severity_desc,
        "all_probabilities": all_probs
    })


# =====================
# RUN SERVER
# =====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("MANGO DISEASE DETECTION + SEVERITY ESTIMATION API")
    print("="*60)
    print("Available endpoints:")
    print("  GET  /health                 - Health check")
    print("  POST /predict                - Full pipeline (detection + disease + severity)")
    print("  POST /predict_severity_only  - Severity estimation only")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=True)
