# Mango Disease Severity Estimation

Complete implementation of severity estimation for mango diseases based on:

**"Mango fruit diseases severity estimation based on image segmentation and deep learning"**  
[Research Paper Link](https://link.springer.com/article/10.1007/s42452-025-06550-z)

## Quick Start

### 1. Setup Dataset Structure
```bash
python scripts/prepare_severity_dataset.py
```

Then organize your images into:
```
dataset/severity/
├── train/{Healthy, Early, Intermediate, Final}/
├── val/{Healthy, Early, Intermediate, Final}/
└── test/{Healthy, Early, Intermediate, Final}/
```

### 2. Train Model
```bash
python scripts/train_severity.py
```

### 3. Evaluate
```bash
python scripts/eval_severity.py
```

### 4. Predict
```bash
# Single image
python scripts/predict_severity.py --image mango.jpg --visualize

# Via API
python main_severity.py
curl -X POST http://localhost:5000/predict -F "image=@mango.jpg"
```

## Severity Classes

| Class | Coverage | Description |
|-------|----------|-------------|
| **Healthy** | 0% | No disease symptoms |
| **Early** | 0.1-3% | First symptoms |
| **Intermediate** | 3-12% | Advanced symptoms |
| **Final** | ≥12% | Very advanced symptoms |

## System Architecture

```
Image → [YOLO Detection] → Crop → ┬→ [Disease Model (ResNet18)] → Disease Type
                                  └→ [Severity Model (ResNet50)] → Severity Level
```

## API Endpoints

### Full Pipeline
```bash
POST /predict
```
Returns detection + disease + severity for all mangoes in image.

### Severity Only
```bash
POST /predict_severity_only
```
Returns severity estimation for a pre-cropped mango image.

## Model Details

- **Architecture**: ResNet50 (pre-trained on ImageNet)
- **Input**: 224×224 RGB images
- **Output**: 4 severity classes
- **Target Performance**: 97.82% accuracy (from research paper)
- **Training Time**: ~20-30 minutes on GPU

## Files Created

**Scripts:**
- `scripts/train_severity.py` - Train the model
- `scripts/eval_severity.py` - Evaluate performance
- `scripts/predict_severity.py` - Make predictions
- `scripts/prepare_severity_dataset.py` - Setup folders

**API:**
- `main_severity.py` - Flask API with full pipeline

**Models:**
- `models/mango_severity_resnet50.pth` - Trained model (after training)

## Requirements

```bash
pip install -r requirements_severity.txt
```

Main dependencies: `torch`, `torchvision`, `flask`, `ultralytics`, `scikit-learn`, `matplotlib`, `seaborn`

## Training Configuration

- **Epochs**: 30 (with early stopping)
- **Batch Size**: 16
- **Learning Rate**: 1e-4
- **Optimizer**: AdamW
- **Data Augmentation**: Flips, rotation, color jitter, affine transforms
- **Strategy**: Freeze backbone (5 epochs) → Fine-tune all layers

## Tips for Best Results

1. **Balanced Dataset**: Aim for 200+ images per severity class
2. **Quality Images**: Clear, well-lit photos of mango fruits
3. **Consistent Labeling**: Use the severity percentages as guidelines
4. **Monitor Training**: Watch for overfitting on validation set
5. **Iterative Improvement**: Start small, evaluate, collect more data, retrain

## Reference

Touré, M. L., Faye, Y., Gueye, A., & Diarra, M. (2025). Mango fruit diseases severity estimation based on image segmentation and deep learning. *SN Applied Sciences*, 7, 40.
