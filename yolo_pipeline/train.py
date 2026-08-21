from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11s-seg.pt")
    
    model.train(
        data="data.yaml",
        epochs=120,
        patience=25,             # Prevents post-epoch-60 overfitting
        imgsz=640,
        batch=6,                 # Lowered to 6 to safely accommodate retina_masks
        workers=2,
        device=0,
        cache='disk',
        
        # Safe Regularization & Loss Tuning
        cls=1.2,                 # Focuses learning on fixing class confusion
        dropout=0.1,             # Mild regularization
        
        # Moderate Augmentations
        mixup=0.10,              # Conservative mixup to prevent overly noisy images
        degrees=10.0,            # Subtle rotation handling
        mosaic=1.0,              # Standard multi-item handling
        
        # Segmentation Improvement
        retina_masks=True,       # Sharper mask evaluation
        name="food_seg_11s_v2"
    )