"""
Inference script for mango disease severity estimation.
Usage:
    python scripts/predict_severity.py --image path/to/image.jpg
    python scripts/predict_severity.py --image path/to/image.jpg --visualize
"""

import argparse
from pathlib import Path
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


# =====================
# CONFIG
# =====================
MODEL_PATH = Path("models/mango_severity_resnet50.pth")
IMG_SIZE = 224


# =====================
# TRANSFORMS
# =====================
inference_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(model_path: Path, device):
    """Load the severity classification model."""
    
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Build model
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    num_classes = checkpoint.get('num_classes', 4)
    model.fc = nn.Linear(num_features, num_classes)
    
    # Load weights
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    # Get class names and severity info
    class_to_idx = checkpoint.get('class_to_idx', {})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    severity_classes = checkpoint.get('severity_classes', {})
    
    print(f"Model loaded successfully!")
    print(f"Architecture: {checkpoint.get('arch', 'resnet50')}")
    print(f"Classes: {list(class_to_idx.keys())}\n")
    
    return model, idx_to_class, severity_classes


def predict_severity(image_path: Path, model, idx_to_class, severity_classes, device):
    """Predict severity class for a single image."""
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    original_image = image.copy()
    
    # Transform for model
    input_tensor = inference_tfms(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
    
    # Get prediction details
    pred_class = idx_to_class[pred_idx]
    severity_desc = severity_classes.get(pred_class, "N/A")
    
    # Get all class probabilities
    all_probs = {idx_to_class[i]: probs[i].item() for i in range(len(probs))}
    
    return {
        'image': original_image,
        'predicted_class': pred_class,
        'confidence': confidence,
        'severity_description': severity_desc,
        'all_probabilities': all_probs
    }


def visualize_prediction(result, save_path=None):
    """Visualize the prediction results."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Display image
    ax1.imshow(result['image'])
    ax1.axis('off')
    title = f"Predicted: {result['predicted_class']}\n"
    title += f"Confidence: {result['confidence']*100:.2f}%\n"
    title += f"{result['severity_description']}"
    ax1.set_title(title, fontsize=12, fontweight='bold')
    
    # Display probability distribution
    classes = list(result['all_probabilities'].keys())
    probs = [result['all_probabilities'][c] * 100 for c in classes]
    
    # Color code based on severity
    colors = []
    for cls in classes:
        if cls == 'Healthy':
            colors.append('#2ecc71')
        elif cls == 'Early':
            colors.append('#f39c12')
        elif cls == 'Intermediate':
            colors.append('#e67e22')
        else:  # Final
            colors.append('#e74c3c')
    
    bars = ax2.barh(classes, probs, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Probability (%)', fontsize=11)
    ax2.set_title('Severity Class Probabilities', fontsize=12, fontweight='bold')
    ax2.set_xlim([0, 100])
    
    # Add value labels
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax2.text(prob + 1, i, f'{prob:.1f}%', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_prediction_details(result):
    """Print detailed prediction information."""
    
    print("\n" + "="*60)
    print("SEVERITY PREDICTION RESULTS")
    print("="*60)
    print(f"\nPredicted Severity: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print(f"Description: {result['severity_description']}")
    
    print(f"\nAll Class Probabilities:")
    print("-" * 60)
    for cls, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
        bar_length = int(prob * 50)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"  {cls:15s} [{bar}] {prob*100:5.2f}%")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Mango Disease Severity Prediction")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default=str(MODEL_PATH), help='Path to model weights')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    parser.add_argument('--save-viz', type=str, default=None, help='Save visualization to path')
    
    args = parser.parse_args()
    
    # Check if image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using: python scripts/train_severity.py")
        return
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load model
    model, idx_to_class, severity_classes = load_model(model_path, device)
    
    # Predict
    print(f"Analyzing image: {image_path}")
    result = predict_severity(image_path, model, idx_to_class, severity_classes, device)
    
    # Print results
    print_prediction_details(result)
    
    # Visualize if requested
    if args.visualize or args.save_viz:
        visualize_prediction(result, save_path=args.save_viz)


if __name__ == "__main__":
    main()
