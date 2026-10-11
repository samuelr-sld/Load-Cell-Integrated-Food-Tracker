import os
import json
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as F
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# --- Configuration & Hyperparameters ---
DATASET_DIR = "../dataset"
OUTPUT_DIR = "../models"
MODEL_NAME = "mobilenet_v3_large"

# Execution & Hardware Settings
BATCH_SIZE = 64          
NUM_WORKERS = 4               # Lowered to prevent Windows multiprocessing freezes
PIN_MEMORY = True
PERSISTENT_WORKERS = False    # Disabled to prevent memory deadlocks on Windows
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training & Regularization Settings
NUM_EPOCHS = 150         
PATIENCE = 20            
LEARNING_RATE = 1e-3     
FINE_TUNE_LR = 1e-4      
IMAGE_SIZE = 224         
DROPOUT_RATE = 0.2       

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Custom Transform: Aspect-Ratio Preserving Letterbox Padding ---
class LetterboxPad:
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        hp = (max_dim - w) // 2
        vp = (max_dim - h) // 2
        return F.pad(img, (hp, vp, max_dim - w - hp, max_dim - h - vp), fill=self.fill)

# --- Data Augmentation Pipeline ---
data_transforms = {
    'train': transforms.Compose([
        LetterboxPad(fill=0),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15.0),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        LetterboxPad(fill=0),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def train_model():
    print(f"Using Device: {DEVICE}")
    print(f"Loading datasets from {DATASET_DIR} using PyTorch ImageFolder...")

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATASET_DIR, x), data_transforms[x])
        for x in ['train', 'val']
    }

    dataloaders = {
        x: DataLoader(
            image_datasets[x], 
            batch_size=BATCH_SIZE, 
            shuffle=(x == 'train'), 
            num_workers=NUM_WORKERS, 
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            drop_last=(x == 'train')  # CRITICAL FIX: Prevents BatchNorm crash on last batch
        )
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    print(f"Dataset Loaded! Total Classes: {num_classes}")
    print(f"Training samples: {dataset_sizes['train']} | Validation samples: {dataset_sizes['val']}")

    class_map_path = os.path.join(OUTPUT_DIR, "class_names.json")
    with open(class_map_path, "w") as f:
        json.dump(class_names, f, indent=4)

    # --- Compute Balanced Class Weights ---
    print("Computing class weights to compensate for dataset imbalance...")
    train_targets = image_datasets['train'].targets
    classes_present = np.unique(train_targets)
    weights = compute_class_weight(class_weight='balanced', classes=classes_present, y=train_targets)
    
    full_weights = np.ones(num_classes, dtype=np.float32)
    for idx, cls_id in enumerate(classes_present):
        full_weights[cls_id] = weights[idx]
    
    class_weights_tensor = torch.tensor(full_weights, dtype=torch.float).to(DEVICE)

    # --- Initialize MobileNetV3-Large ---
    print(f"Initializing {MODEL_NAME} architecture...")
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

    # Freeze backbone initially
    for param in model.features.parameters():
        param.requires_grad = False

    # Modify Classifier Head
    in_features = model.classifier[-1].in_features
    model.classifier[2] = nn.Dropout(p=DROPOUT_RATE, inplace=False)
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    
    optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_top1 = 0.0
    patience_counter = 0
    backbone_unfrozen = False

    print("\n--- Starting MobileNetV3-Large Training ---")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 35)

        # Unfreeze the backbone after 3 epochs
        if epoch == 3 and not backbone_unfrozen:
            print(">>> Unfreezing Backbone Features for End-to-End Fine-tuning <<<")
            for param in model.features.parameters():
                param.requires_grad = True
            
            optimizer = optim.AdamW(model.parameters(), lr=FINE_TUNE_LR, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(NUM_EPOCHS - 3), eta_min=1e-6)
            backbone_unfrozen = True

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects_top1 = 0
            running_corrects_top5 = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Phase"):
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, top5_preds = outputs.topk(min(5, num_classes), 1, True, True)
                    correct_k = top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds))

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects_top1 += correct_k[:, :1].reshape(-1).float().sum().item()
                running_corrects_top5 += correct_k[:, :min(5, num_classes)].reshape(-1).float().sum().item()

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_top1 = running_corrects_top1 / dataset_sizes[phase]
            epoch_top5 = running_corrects_top5 / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | Top-1 Acc: {epoch_top1:.4f} | Top-5 Acc: {epoch_top5:.4f}")

            # NEW: Immediate Disk-Saving Mechanism
            if phase == 'val':
                if epoch_top1 > best_acc_top1:
                    best_acc_top1 = epoch_top1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    
                    checkpoint_path = os.path.join(OUTPUT_DIR, "mobilenet_v3_food101_best.pt")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'class_names': class_names,
                        'num_classes': num_classes,
                        'best_acc_top1': best_acc_top1
                    }, checkpoint_path)
                    print(f"--> Validation Top-1 Acc improved to {best_acc_top1:.4f}. Saved to disk immediately!")
                else:
                    patience_counter += 1
                    print(f"--> No improvement in Top-1 Acc. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at Epoch {epoch + 1}!")
            break

    time_elapsed = time.time() - start_time
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Top-1 Validation Accuracy: {best_acc_top1:.4f}")

    # Load best weights for ONNX export
    model.load_state_dict(best_model_wts)
    
    onnx_path = os.path.join(OUTPUT_DIR, "mobilenet_v3_food101.onnx")
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    model.eval()
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=17,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Exported ONNX model to {onnx_path}")

if __name__ == "__main__":
    train_model()