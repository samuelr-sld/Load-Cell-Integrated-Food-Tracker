"""Can YOLO's own class prediction serve as the food identity? Runs the testbench's segmentation path over FoodSeg103
validation photos (ground-truth polygon masks) and separates two questions:

  * localisation  - did YOLO draw a segment where food really is?      (mask IoU >= 0.5, class ignored)
  * identity      - given such a segment, is its class name correct?

It also validates the per-instance output schema (class id range, name lookup, confidence range, mask shape) and reports
class accuracy weighted by pixel area, which is what matters for portion-based nutrition.

    python -m testbench.scripts.yolo_reliability_check --limit 300
    python -m testbench.scripts.yolo_reliability_check --out report.json      # full validation split
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from ..config import REPO_ROOT, Settings
from ..ingest import decode_image
from ..model_loader import load_yolo, resolve_device
from ..segmentation import SegmentationService

DATA_DIR = REPO_ROOT / "yolo_pipeline" / "data"
THRESHOLDS = (0.25, 0.40, 0.50, 0.60, 0.70)
MATCH_IOU = 0.5
DUP_IOU = 0.7          # two segments this similar are "the same region"


def load_gt(label_path: Path, h: int, w: int) -> list[tuple[int, np.ndarray]]:
    """YOLO-seg label file -> [(class_id, bool mask HxW)]. Polygons are normalised, so they scale to any decode size."""
    out = []
    if not label_path.is_file():
        return out
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2:
            continue
        pts = np.array(parts[1:], dtype=np.float32).reshape(-1, 2) * np.array([w, h], dtype=np.float32)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        out.append((int(parts[0]), m.astype(bool)))
    return out


def schema_problems(dets, names: dict[int, str], h: int, w: int) -> Counter:
    """Every YOLO detection must carry a valid id, a name that is the checkpoint's name for that id, a sane confidence."""
    bad = Counter()
    valid_ids = {k for k in names if k >= 0}
    for d in dets:
        if not isinstance(d.class_id, int):
            bad["class_id_not_int"] += 1
        if d.class_id not in valid_ids:
            bad["class_id_outside_named_classes"] += 1
        if d.class_name != names.get(d.class_id) or d.class_name.startswith("class_"):
            bad["class_name_does_not_match_checkpoint_names"] += 1
        if not (0.0 < d.confidence <= 1.0):
            bad["confidence_outside_(0,1]"] += 1
        if d.mask.shape != (h, w):
            bad["mask_shape_differs_from_image"] += 1
        if not d.mask.any():
            bad["empty_mask"] += 1
        x1, y1, x2, y2 = d.box_xyxy
        if not (x2 > x1 and y2 > y1):
            bad["degenerate_box"] += 1
    return bad


def _flat(masks: list[np.ndarray]) -> np.ndarray:
    return np.stack([m.reshape(-1) for m in masks]).astype(np.float32) if masks else np.zeros((0, 0), np.float32)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = a @ b.T
    return inter / np.maximum(a.sum(1)[:, None] + b.sum(1)[None, :] - inter, 1.0)


def match_image(dets, gts) -> list[dict]:
    """Greedy by descending confidence; class-agnostic. One row per prediction."""
    rows = [{"conf": d.confidence, "pred": d.class_id, "gt": None, "iou": 0.0} for d in dets]
    if not dets or not gts:
        return rows
    iou = _iou(_flat([d.mask for d in dets]), _flat([m for _, m in gts]))
    taken: set[int] = set()
    for i in np.argsort([-d.confidence for d in dets]):
        for j in np.argsort(-iou[i]):
            if iou[i, j] < MATCH_IOU:
                break
            if j not in taken:
                taken.add(int(j))
                rows[i].update(gt=gts[j][0], iou=float(iou[i, j]))
                break
    return rows


def pixel_stats(dets, gts, h: int, w: int, thr: float) -> tuple[int, int, int]:
    """(gt food pixels, covered by some prediction, covered AND same class). Higher confidence paints last (wins)."""
    gt_map = np.full((h, w), -1, np.int16)
    for c, m in gts:
        gt_map[m] = c
    pred_map = np.full((h, w), -1, np.int16)
    for d in sorted((d for d in dets if d.confidence >= thr), key=lambda d: d.confidence):
        pred_map[d.mask] = d.class_id
    food = gt_map >= 0
    covered = food & (pred_map >= 0)
    return int(food.sum()), int(covered.sum()), int((covered & (pred_map == gt_map)).sum())


def run(limit: int | None = 300, split: str = "val", device_pref: str = "auto", conf: float | None = None) -> dict:
    settings = replace(Settings(), **({"yolo_conf": conf} if conf is not None else {}))
    device, _ = resolve_device(device_pref)
    yolo, names = load_yolo(settings.yolo_weights)
    seg = SegmentationService(yolo, names, device, settings)
    images = sorted((DATA_DIR / "images" / split).glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images under {DATA_DIR / 'images' / split}")
    if limit:
        step = max(1, len(images) // limit)         # evenly spaced, not just the first N
        images = images[::step][:limit]

    schema_bad: Counter = Counter()
    rows_all: list[dict] = []
    gt_class_count: Counter = Counter()
    confusions: Counter = Counter()
    pix = {t: [0, 0, 0] for t in THRESHOLDS}
    n_pred = n_dup_pairs = n_dup_pairs_diffcls = 0
    ms: list[float] = []
    out_of_taxonomy_ids: Counter = Counter()

    for path in images:
        decoded = decode_image(path.read_bytes(), path.name, settings)
        h, w = decoded.rgb.shape[:2]
        t0 = time.perf_counter()
        out = seg.segment(decoded.rgb)
        ms.append((time.perf_counter() - t0) * 1000)
        dets = out.detections
        schema_bad += schema_problems(dets, names, h, w)
        out_of_taxonomy_ids.update(d.class_id for d in dets if d.class_id not in names or d.class_id < 0)

        gts = load_gt(DATA_DIR / "labels" / split / f"{path.stem}.txt", h, w)
        gt_class_count.update(c for c, _ in gts)
        rows = match_image(dets, gts)
        for r in rows:
            r["ok"] = r["gt"] is not None and r["gt"] == r["pred"]
            if r["gt"] is not None and not r["ok"]:
                confusions[(r["gt"], r["pred"])] += 1
        rows_all.extend(rows)
        n_pred += len(dets)

        if len(dets) > 1:
            m = _iou(_flat([d.mask for d in dets]), _flat([d.mask for d in dets]))
            for i in range(len(dets)):
                for j in range(i + 1, len(dets)):
                    if m[i, j] >= DUP_IOU:
                        n_dup_pairs += 1
                        n_dup_pairs_diffcls += dets[i].class_id != dets[j].class_id
        for t in THRESHOLDS:
            for k, v in enumerate(pixel_stats(dets, gts, h, w, t)):
                pix[t][k] += v

    total_gt = sum(gt_class_count.values())
    sweep = {}
    for t in THRESHOLDS:
        sel = [r for r in rows_all if r["conf"] >= t]
        loc = [r for r in sel if r["gt"] is not None]
        ok = [r for r in sel if r["ok"]]
        food, covered, correct = pix[t]
        sweep[f"{t:.2f}"] = {
            "predictions": len(sel),
            "localised_pct": round(100 * len(loc) / len(sel), 1) if sel else None,
            "class_correct_given_localised_pct": round(100 * len(ok) / len(loc), 1) if loc else None,
            "strict_precision_pct": round(100 * len(ok) / len(sel), 1) if sel else None,
            "strict_recall_pct": round(100 * len(ok) / total_gt, 1) if total_gt else None,
            "pixel_coverage_pct": round(100 * covered / food, 1) if food else None,
            "pixel_class_acc_of_all_food_pct": round(100 * correct / food, 1) if food else None,
            "pixel_class_acc_of_covered_pct": round(100 * correct / covered, 1) if covered else None,
        }

    per_class = {}
    base = [r for r in rows_all if r["conf"] >= THRESHOLDS[0]]
    by_pred, by_gt_ok = defaultdict(list), Counter()
    for r in base:
        by_pred[r["pred"]].append(r)
    for r in base:
        if r["ok"]:
            by_gt_ok[r["pred"]] += 1
    for cid in sorted(k for k in names if k >= 0):
        preds = by_pred.get(cid, [])
        tp, n_gt = by_gt_ok.get(cid, 0), gt_class_count.get(cid, 0)
        per_class[names[cid]] = {
            "class_id": cid, "gt_instances": n_gt, "predictions": len(preds), "true_positive": tp,
            "precision": round(tp / len(preds), 3) if preds else None, "recall": round(tp / n_gt, 3) if n_gt else None,
        }

    top_conf = [(f"{names[g]} -> {names.get(p, p)}", n) for (g, p), n in confusions.most_common(12)]
    return {
        "split": split, "images": len(images), "device": str(device), "conf_used": settings.yolo_conf,
        "gt_instances": total_gt, "predictions": n_pred,
        "yolo_ms_per_image": {"mean": round(float(np.mean(ms)), 1), "p95": round(float(np.percentile(ms, 95)), 1)},
        "schema_problems": dict(schema_bad) or "none",
        "class_ids_outside_named_classes": dict(out_of_taxonomy_ids) or "none",
        "duplicates": {"pairs_with_mask_iou_ge_0.7": n_dup_pairs, "of_which_different_class": n_dup_pairs_diffcls},
        "threshold_sweep": sweep,
        "top_confusions_gt_to_pred": top_conf,
        "per_class_at_base_conf": per_class,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300, help="images to evaluate, evenly spaced (0 = whole split)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--conf", type=float, default=None, help="override the segmentation confidence floor (default: Settings)")
    ap.add_argument("--out", type=Path, default=None, help="write the full JSON report here")
    args = ap.parse_args()
    rep = run(args.limit or None, args.split, args.device, args.conf)
    if args.out:
        args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    brief = {k: v for k, v in rep.items() if k != "per_class_at_base_conf"}
    print(json.dumps(brief, indent=2))
    sys.exit(0)
