"""
Converts EduardoPacheco/FoodSeg103 (Hugging Face) from semantic segmentation masks
into YOLO single-class instance-segmentation format (images/ + labels/ with polygon .txt files).

Target Directory Layout:
    yolo_pipeline/
    ├── data/
    │   ├── images/ (train/val)
    │   └── labels/ (train/val)
    └── scripts/
        └── convert_foodseg.py
"""

import os
import numpy as np
import cv2
from datasets import load_dataset

# Write to yolo_pipeline/data/ relative to yolo_pipeline/scripts/
OUT_DIR = "../data"
MIN_CONTOUR_AREA = 50  # Skip tiny noise blobs from mask edges

def ensure_dirs():
    for split in ["train", "val"]:
        os.makedirs(f"{OUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUT_DIR}/labels/{split}", exist_ok=True)

def convert_split(dataset_split, out_name):
    print(f"Converting {out_name}: {len(dataset_split)} images")
    for i, example in enumerate(dataset_split):
        image = example["image"].convert("RGB")
        label = np.array(example["label"])  # HxW, pixel value = class id

        img_path = f"{OUT_DIR}/images/{out_name}/{i:06d}.jpg"
        image.save(img_path, quality=95)

        h, w = label.shape
        lines = []
        class_ids = np.unique(label)

        for cid in class_ids:
            if cid == 0:
                continue  # 0 = background, skip

            mask = (label == cid).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                    continue
                contour = contour.reshape(-1, 2).astype(np.float32)
                contour[:, 0] /= w  # normalize x to 0-1
                contour[:, 1] /= h  # normalize y to 0-1
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in contour)
                
                # Class 0 = generic food_item (single-class pipeline setup)
                lines.append(f"0 {coords}")

        label_path = f"{OUT_DIR}/labels/{out_name}/{i:06d}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        if i % 500 == 0:
            print(f"  {i}/{len(dataset_split)}")

if __name__ == "__main__":
    ensure_dirs()
    ds = load_dataset("EduardoPacheco/FoodSeg103")
    convert_split(ds["train"], "train")
    convert_split(ds["validation"], "val")
    print("Done converting dataset to single-class YOLO segmentation format.")