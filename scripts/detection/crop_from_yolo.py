from ultralytics import YOLO
import cv2
import os

model = YOLO("runs/detect/train/weights/best.pt")

img_path = "path/to/occluded.jpg"
results = model(img_path)

img = cv2.imread(img_path)
os.makedirs("cropped_fruits", exist_ok=True)

i = 0
for box in results[0].boxes.xyxy:
    x1, y1, x2, y2 = map(int, box)
    crop = img[y1:y2, x1:x2]
    cv2.imwrite(f"cropped_fruits/fruit_{i}.jpg", crop)
    i += 1
