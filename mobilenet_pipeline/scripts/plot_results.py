import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

DATASET_DIR = "../dataset"
MODEL_PATH = "../models/mobilenet_v3_food.pt"
CLASS_MAP_PATH = "../models/class_names.json"
OUTPUT_DIR = "../models/results"
BATCH_SIZE = 64
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

class LetterboxPad:
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        hp = (max_dim - w) // 2
        vp = (max_dim - h) // 2
        return F.pad(img, (hp, vp, max_dim - w - hp, max_dim - h - vp), fill=self.fill)

# Evaluation Data Pipeline
val_transforms = transforms.Compose([
    LetterboxPad(fill=0),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def generate_yolo_style_reports():
    print(f"Loading evaluation data on {DEVICE}...")
    val_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"), val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    with open(CLASS_MAP_PATH, "r") as f:
        class_names = json.load(f)

    num_classes = len(class_names)
    print(f"Loading saved model weights from {MODEL_PATH}...")

    # Reconstruct MobileNetV3 Architecture
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[2] = torch.nn.Dropout(p=0.1, inplace=True)
    model.classifier[3] = torch.nn.Linear(in_features, num_classes)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    all_preds = []
    all_targets = []
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    print("Evaluating validation set for Top-1 & Top-5 Accuracy...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation Progress"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            batch_size = labels.size(0)

            outputs = model(inputs)
            
            # --- TOP-1 & TOP-5 ACCURACY CALCULATION ---
            # Get the top 5 predicted class indices for each sample in batch
            _, top5_preds = outputs.topk(5, 1, True, True)
            
            # Top-1 predictions (for confusion matrix & report)
            top1_preds = top5_preds[:, 0]
            
            # Check if true target label exists in the top 5 predictions
            correct_k = top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds))
            
            correct_top1 += correct_k[:, :1].sum().item()
            correct_top5 += correct_k[:, :5].sum().item()
            total_samples += batch_size

            all_preds.extend(top1_preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    top1_acc = (correct_top1 / total_samples) * 100
    top5_acc = (correct_top5 / total_samples) * 100

    print("\n" + "="*40)
    print(f" Top-1 Validation Accuracy: {top1_acc:.2f}%")
    print(f" Top-5 Validation Accuracy: {top5_acc:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    generate_yolo_style_reports()