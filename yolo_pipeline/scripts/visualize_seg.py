"""
Visualizes YOLO segmentation polygon labels drawn over the source image.
Saves an output image so you can open and inspect it in VS Code.

Usage:
    python visualize_seg.py
"""

import cv2
import numpy as np
import yaml
IMAGE_PATH = "data/images/train/000007.jpg"
LABEL_PATH = "data/labels/train/000007.txt"
OUTPUT_PATH = "check_000002.jpg"

with open("data.yaml") as f:
    names = yaml.safe_load(f)["names"]

img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]

with open(LABEL_PATH) as f:
    lines = [l.strip() for l in f if l.strip()]

print(f"Found {len(lines)} annotated region(s) in this image")

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
]

for i, line in enumerate(lines):
    parts = list(map(float, line.split()))
    class_id = int(parts[0])
    coords = parts[1:]

    points = np.array(coords, dtype=np.float32).reshape(-1, 2)
    points[:, 0] *= w
    points[:, 1] *= h
    points = points.astype(np.int32)

    color = colors[i % len(colors)]
    cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)

    label = names.get(class_id, f"id_{class_id}")
    x, y = points[0]
    cv2.putText(img, label, (int(x), int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    print(f"  region {i}: class_id={class_id} -> '{label}', {len(points)} points")

cv2.imwrite(OUTPUT_PATH, img)
print(f"Saved visualization to {OUTPUT_PATH}")