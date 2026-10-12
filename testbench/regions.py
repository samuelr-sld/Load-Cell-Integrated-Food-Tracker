"""Same-region duplicate handling and per-region identity state.

YOLO is run with per-class NMS, so one physical food region can come back as several segments with different classes.
Segments are grouped into one *region* only when their MASKS overlap strongly (IoU). Box overlap is deliberately ignored
(adjacent portions have overlapping boxes) and so is containment (a small topping inside a large dish is a separate item).
Grouping is greedy against each group's highest-confidence segment, so chains A~B~C never merge A with C.
All original segments stay in the result; regions only say which of them describe the same food.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .nutrition import NUTRIENT_FIELDS, NutritionService, norm_label, serving_nutrition

MAX_WEIGHT_G = 5000.0
TOP_DISH_SUGGESTIONS = 3
DISH_NOTE = "Food-101 dish suggestion. The classifier was trained on whole dishes, so it is unreliable on ingredient crops."


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    return inter / float(int(a.sum()) + int(b.sum()) - inter)


def group_by_mask_overlap(items: list[tuple[int, float, np.ndarray]], iou_threshold: float) -> list[list[tuple[int, float]]]:
    """items: (segment_id, confidence, bool mask). Returns groups of (segment_id, iou_with_primary), primary first.
    Groups are ordered by their primary's position in the input so region ids are stable."""
    order = sorted(range(len(items)), key=lambda i: -items[i][1])
    groups: list[list[int]] = []
    ious: dict[int, float] = {}
    for i in order:
        for g in groups:
            iou = mask_iou(items[g[0]][2], items[i][2])
            if iou >= iou_threshold:
                g.append(i)
                ious[i] = iou
                break
        else:
            groups.append([i])
            ious[i] = 1.0
    groups.sort(key=lambda g: min(g))
    return [[(items[i][0], round(ious[i], 4)) for i in g] for g in groups]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RegionRecord:
    id: int
    primary_segment_id: int
    segment_ids: list[int]                      # primary first
    candidates: list[dict] = field(default_factory=list)
    status: str = "needs_confirmation"          # "needs_confirmation" | "confirmed"
    identity: Optional[dict] = None             # {label, taxonomy, from_suggestion, confirmed_at}
    usda_fdc_id: Optional[int] = None           # user's pick when the label has several USDA records
    weight_g: Optional[float] = None            # user-entered only; never inferred from mask area. Legacy: the total-weight workflow never writes it.
    user_entity_count: Optional[int] = None     # physical items per the user; kept apart from len(segment_ids), which is what YOLO detected
    excluded: bool = False                      # user marked this region "not food" (e.g. the plate itself): left out of area and weight
    source: str = "detected"                    # "detected" | "manual" (user-drawn fallback for a food YOLO missed entirely)
    manual_polygon: Optional[list] = None       # [[x, y], ...] in inference-image pixels, only for source == "manual"

    def confirm(self, label: str, taxonomy: str, from_suggestion: bool, fdc_id: Optional[int] = None) -> None:
        self.identity = {"label": norm_label(label), "taxonomy": taxonomy, "from_suggestion": from_suggestion, "confirmed_at": _now()}
        self.usda_fdc_id, self.status = fdc_id, "confirmed"

    def clear(self) -> None:
        self.identity, self.usda_fdc_id, self.status = None, None, "needs_confirmation"

    def nutrition(self, svc: NutritionService) -> dict:
        if self.identity is None:
            return {"status": "needs_confirmation", "message": "Confirm the food identity first. Nothing is looked up for unconfirmed regions.",
                    "per_100g": None, "serving": None}
        look = svc.lookup(self.identity["taxonomy"], self.identity["label"], self.usda_fdc_id)
        selected = look["selected"]
        out = {"status": look["status"], "message": look["message"], "usda_candidates": look["candidates"],
               "usda_record": ({k: selected[k] for k in ("fdc_id", "description", "data_source", "preparation", "energy_basis")} if selected else None),
               "per_100g": selected["per_100g"] if selected else None, "weight_g": self.weight_g, "serving": None,
               "serving_message": None}
        if selected and self.weight_g is not None:
            out["serving"] = serving_nutrition(selected["per_100g"], self.weight_g)
        elif selected:
            out["serving_message"] = "Enter a portion weight in grams to calculate serving nutrition. It is not estimated from the image."
        return out

    def to_dict(self, svc: NutritionService) -> dict:
        cands = [{**c, "usda_mapped": bool(svc.candidates(c["taxonomy"], c["label"]))} for c in self.candidates]
        return {"region_id": self.id, "primary_segment_id": self.primary_segment_id, "segment_ids": self.segment_ids,
                "duplicate_segment_ids": self.segment_ids[1:], "status": self.status, "identity": self.identity,
                "candidates": cands, "nutrition": self.nutrition(svc), "source": self.source, "excluded": self.excluded,
                "user_entity_count": self.user_entity_count, "detected_segment_count": len(self.segment_ids)}


def build_candidates(segments: list, predictions_by_segment: dict[int, list[dict]]) -> list[dict]:
    """YOLO classes of every member segment (best confidence per class), then MobileNet's top dishes for the primary."""
    best: dict[str, dict] = {}
    for s in segments:
        label = norm_label(s.yolo_class_name)
        if label not in best or s.yolo_confidence > best[label]["confidence"]:
            best[label] = {"source": "yolo", "taxonomy": "foodseg103", "label": label, "class_id": s.yolo_class_id,
                           "confidence": s.yolo_confidence, "segment_id": s.id}
    out = sorted(best.values(), key=lambda c: -c["confidence"])
    for p in predictions_by_segment.get(segments[0].id, [])[:TOP_DISH_SUGGESTIONS]:
        out.append({"source": "mobilenet", "taxonomy": "food101", "label": norm_label(p["label"]), "class_id": p["class_index"],
                    "confidence": p["softmax_score"], "segment_id": segments[0].id, "note": DISH_NOTE})
    return out


def valid_weight(value) -> float:
    w = float(value)
    if not math.isfinite(w) or w <= 0 or w > MAX_WEIGHT_G:
        raise ValueError(f"weight_g must be greater than 0 and at most {MAX_WEIGHT_G:g}")
    return w


def meal_totals(regions: list[RegionRecord], svc: NutritionService) -> dict:
    """Sum over REGIONS (not segments), so duplicate detections of one food are never counted twice."""
    included, excluded = [], []
    totals = {k: 0.0 for k in NUTRIENT_FIELDS}
    for r in regions:
        if r.excluded:
            continue
        n = r.nutrition(svc)
        label = r.identity["label"] if r.identity else None
        if n["serving"] is None:
            reason = {"needs_confirmation": "identity not confirmed"}.get(n["status"], n["status"] if n["status"] != "ok" else "no portion weight entered")
            excluded.append({"region_id": r.id, "label": label, "reason": reason})
            continue
        included.append({"region_id": r.id, "label": label, "weight_g": r.weight_g, "fdc_id": n["usda_record"]["fdc_id"], "serving": n["serving"]})
        for k in totals:
            totals[k] = None if totals[k] is None or n["serving"][k] is None else totals[k] + n["serving"][k]
    return {"totals": {k: (None if v is None else round(v, 2)) for k, v in totals.items()} if included else None,
            "included_regions": included, "excluded_regions": excluded,
            "note": "Totals cover only confirmed regions with a USDA record and a user-entered weight."}
