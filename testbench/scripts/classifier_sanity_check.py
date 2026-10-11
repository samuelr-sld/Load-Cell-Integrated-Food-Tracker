"""Isolates the classifier from YOLO: runs held-out Food-101 validation images through the testbench's own loader,
preprocessing and ClassificationService, and reports top-1/top-5.

If the class-index mapping or preprocessing were wrong this would collapse to ~1% (chance). A healthy integration
lands near the validation accuracy stored in the checkpoint (best_acc_top1).

    python -m testbench.scripts.classifier_sanity_check --per-class 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from ..config import REPO_ROOT, Settings
from ..model_loader import load_classifier, resolve_device
from ..classification import ClassificationService
from ..preprocessing import build_val_transform

VAL_DIR = REPO_ROOT / "mobilenet_pipeline" / "dataset" / "val"


def run(per_class: int = 5, device_pref: str = "auto", val_dir: Path = VAL_DIR) -> dict:
    settings = Settings()
    device, _ = resolve_device(device_pref)
    model, class_names, arch, meta = load_classifier(settings.mobilenet_weights, settings.class_names_json, device)
    svc = ClassificationService(model, class_names, device)

    folders = sorted(p.name for p in val_dir.iterdir() if p.is_dir())
    if folders != class_names:
        raise SystemExit("validation folder names do not match the checkpoint class list")

    transform = build_val_transform()
    top1 = top5 = n = 0
    for ci, name in enumerate(class_names):
        files = sorted((val_dir / name).glob("*.jpg"))[:per_class]
        tensors = [transform(Image.open(f).convert("RGB")) for f in files]  # same loader as torchvision ImageFolder
        for preds in svc.classify(tensors):
            n += 1
            top1 += preds[0].class_index == ci
            top5 += any(p.class_index == ci for p in preds)
    return {"device": str(device), "architecture": arch, "images": n, "top1": top1 / n, "top5": top5 / n,
            "checkpoint_best_val_top1": meta["checkpoint_best_val_top1"], "chance_top1": 1 / len(class_names)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    print(json.dumps(run(args.per_class, args.device), indent=2))
    sys.exit(0)
