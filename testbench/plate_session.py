"""Glue between an InferenceRecord and the Phase-1 services (PlateAnalysisService, WeightAllocationService).

Nothing here estimates anything itself: it rebuilds full-frame masks, maps the record's confirmed identities into
RegionIdentity, calls the existing services, then applies USDA per-100 g values (nutrition.serving_nutrition) to the
ESTIMATED weights. The legacy per-region `weight_g` is never read or written by this workflow.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from .config import Settings
from .nutrition import NUTRIENT_FIELDS, NutritionService, serving_nutrition
from .plate_analysis import FoodRegion, PlateAnalysisConfig, PlateAnalysisService, PlateGeometry, region_mask
from .records import InferenceRecord
from .regions import RegionRecord
from .weight_allocation import (DISCLAIMER, RegionIdentity, TotalWeightMeasurement, WeightAllocationConfig,
                                WeightAllocationService)

# API weight source -> WeightAllocationService source. "manual_simulation" is a typed-in number standing in for the future
# ESP32 load cell; when the sensor exists it becomes "load_cell" (already a valid service source) with no other change.
WEIGHT_SOURCE_MAP = {"manual_simulation": "manual"}
WEIGHT_SOURCES = tuple(WEIGHT_SOURCE_MAP)

MISSED_FOOD_WARNING = ("High plate coverage does not prove every food was detected. A food YOLO missed entirely has no mask, so it "
                       "cannot be accounted for automatically: its weight would be wrongly shared among the detected foods. "
                       "Add it manually (draw its outline) or re-shoot the photo.")
UNVALIDATED_WARNING = ("estimate_unvalidated: area-proportional weights and the nutrition derived from them have no empirically "
                       "established accuracy margin.")
SIMULATION_WARNING = ("The total weight is a manual simulation, not a live load-cell reading.")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ masks
def unpack_mask(packed: bytes, height: int, width: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=height * width).reshape(height, width).astype(bool)


def polygon_mask(polygon: list, height: int, width: int) -> np.ndarray:
    m = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(polygon, dtype=float)).astype(np.int32)], 1)
    return m.astype(bool)


def full_region_mask(record: InferenceRecord, region: RegionRecord) -> np.ndarray:
    h, w = record.image["height"], record.image["width"]
    if region.source == "manual":
        return polygon_mask(region.manual_polygon, h, w)
    return region_mask([unpack_mask(record.segment_masks[sid], h, w) for sid in region.segment_ids])


# ------------------------------------------------------------------ calibration
def build_geometry(record: InferenceRecord, *, center_x=None, center_y=None, radius_px=None, rim_points=None,
                   image_width=None, image_height=None, plate_diameter_mm=None) -> PlateGeometry:
    """Calibration may come from a different resolution than the one the masks were made at (e.g. the original photo);
    PlateGeometry.scaled_to maps it onto the inference image. Raises ValueError on anything unusable."""
    cw, ch = image_width or record.image["width"], image_height or record.image["height"]
    if rim_points:
        geo = PlateGeometry.from_rim_points(rim_points, cw, ch, plate_diameter_mm)
    elif None in (center_x, center_y, radius_px):
        raise ValueError("give either rim_points (three points on the plate rim) or center_x, center_y and radius_px")
    else:
        geo = PlateGeometry(center_x, center_y, radius_px, cw, ch, plate_diameter_mm)
    if (geo.image_width, geo.image_height) != (record.image["width"], record.image["height"]):
        geo = geo.scaled_to(record.image["width"], record.image["height"])
    return geo


# ------------------------------------------------------------------ analysis
def _issue_warning(i: dict) -> dict:
    return {"code": i["code"], "severity": i["severity"], "message": i["message"], "region_id": i.get("region_id")}


def compute(record: InferenceRecord, settings: Settings, nutri: NutritionService) -> dict:
    """Everything the UI and the meal log need for one result, derived fresh from the record (no cached numbers)."""
    active = [r for r in record.regions if not r.excluded]
    seg_by_id = {s.id: s for s in record.segments}
    missing = []
    if record.plate is None:
        missing.append("plate_calibration")
    if record.total_weight is None:
        missing.append("total_weight")
    if not active:
        missing.append("food_regions")

    def confidence(r: RegionRecord) -> float:
        return 2.0 if r.source == "manual" else seg_by_id[r.primary_segment_id].yolo_confidence

    analysis = alloc = None
    if record.plate is not None and active:
        cfg = PlateAnalysisConfig(settings.overlap_policy, settings.off_plate_flag_fraction, settings.shared_flag_fraction, settings.min_region_px)
        priority = [r.id for r in sorted(active, key=lambda r: -confidence(r))] if settings.overlap_policy == "priority" else None
        regions = [FoodRegion(r.id, full_region_mask(record, r), list(r.segment_ids)) for r in active]
        analysis = PlateAnalysisService(cfg).analyze(record.plate["geometry"], regions, priority)
        if record.total_weight is not None:
            tw = record.total_weight
            meas = TotalWeightMeasurement(tw["net_weight_g"], WEIGHT_SOURCE_MAP[tw["source"]], tw.get("set_at"), tw.get("tare_g"))
            idents = [RegionIdentity(r.id, "confirmed" if r.status == "confirmed" else "needs_confirmation",
                                     r.identity["label"] if r.identity else None, r.identity["taxonomy"] if r.identity else None,
                                     r.user_entity_count) for r in active]
            alloc = WeightAllocationService(WeightAllocationConfig(settings.strict_coverage)).allocate(analysis, meas, idents, settings.food_coefficients)

    area = {a.region_id: a for a in analysis.regions} if analysis else {}
    al = {a.region_id: a for a in alloc.allocations} if alloc else {}

    rows, included, excluded_nut = [], [], []
    totals = {k: 0.0 for k in NUTRIENT_FIELDS}
    for r in active:
        a, w = area.get(r.id), al.get(r.id)
        look = nutri.lookup(r.identity["taxonomy"], r.identity["label"], r.usda_fdc_id) if r.identity else None
        usda = ({k: look["selected"][k] for k in ("fdc_id", "description", "data_source", "preparation", "energy_basis")}
                if look and look["selected"] else None)
        serving = None
        if w is not None and w.status == "allocated" and usda:
            serving = serving_nutrition(look["selected"]["per_100g"], w.estimated_weight_g)
            included.append(r.id)
            for k in totals:
                totals[k] = None if totals[k] is None or serving[k] is None else totals[k] + serving[k]
        elif w is not None and w.status == "allocated":
            excluded_nut.append({"region_id": r.id, "label": w.label,
                                 "reason": look["message"] if look else "no USDA record"})
        rows.append({
            "region_id": r.id, "source": r.source, "color_rgb": seg_by_id[r.primary_segment_id].color_rgb if r.source == "detected" else [120, 120, 120],
            "identity_status": r.status, "label": r.identity["label"] if r.identity else None,
            "taxonomy": r.identity["taxonomy"] if r.identity else None,
            "entity_count": {"detected_segments": len(r.segment_ids), "user_count": r.user_entity_count},
            "legacy_manual_weight_g": r.weight_g,
            "area": ({k: getattr(a, k) for k in ("mask_area_px", "on_plate_px", "attributed_area_px", "food_area_pct", "plate_coverage_pct",
                                                  "shared_px", "attributed_area_cm2", "flags")} if a else None),
            "allocation": ({"status": w.status, "estimated_weight_g": w.estimated_weight_g, "unresolved_weight_g": w.unresolved_weight_g,
                            "weight_share": w.weight_share, "weight_per_entity_g": w.weight_per_entity_g, "coefficient": w.coefficient} if w else None),
            "usda": usda, "usda_status": look["status"] if look else None, "usda_message": look["message"] if look else None,
            "nutrition": serving,
        })

    # --- states -------------------------------------------------------------------------------------------------
    status = f"allocation_{alloc.status}" if alloc else None          # None = inputs still missing, nothing was attempted
    nutrition_complete = bool(alloc) and alloc.status != "rejected" and bool(included) and not excluded_nut
    complete = status == "allocation_complete" and nutrition_complete
    reasons = []
    if status is None:
        reasons.append("inputs_missing: " + ", ".join(missing))
    elif status == "allocation_rejected":
        reasons += [i.code for i in alloc.issues if i.severity == "blocking"]
    elif status == "allocation_partial":
        reasons.append("unresolved_regions")
    if status in ("allocation_complete", "allocation_partial") and (not included or excluded_nut):
        reasons.append("nutrition_incomplete")

    warnings = [{"code": "missed_food_limitation", "severity": "warning", "message": MISSED_FOOD_WARNING, "region_id": None}]
    if record.total_weight is not None:
        warnings.append({"code": "weight_is_simulation", "severity": "info", "message": SIMULATION_WARNING, "region_id": None})
    warnings.append({"code": "estimate_unvalidated", "severity": "warning", "message": UNVALIDATED_WARNING, "region_id": None})
    if alloc:
        warnings += [_issue_warning(i.to_dict()) for i in alloc.issues]
        unresolved_ids = [a.region_id for a in alloc.allocations if a.status == "unresolved"]
        if unresolved_ids:
            warnings.append({"code": "unresolved_regions", "severity": "warning", "region_id": None,
                             "message": f"{alloc.unresolved_weight_g:g} g is held back for unconfirmed region(s) {', '.join(map(str, unresolved_ids))}; "
                                        "it is NOT given to the confirmed foods. Confirm, correct or exclude them."})
    elif analysis:
        warnings += [{"code": "uncertain_coverage", "severity": "warning", "message": f"Plate coverage is uncertain: {f.replace('_', ' ')}.", "region_id": None}
                     for f in analysis.coverage_flags]
    for e in excluded_nut:
        warnings.append({"code": "no_usda_data", "severity": "warning", "region_id": e["region_id"],
                         "message": f"Region {e['region_id']} ({e['label']}) has no nutrition yet: {e['reason']}"})
    if any(r.source == "manual" for r in active):
        warnings.append({"code": "manual_region", "severity": "info", "region_id": None,
                         "message": "At least one region is a user-drawn outline, not a YOLO mask; its area is only as accurate as the drawing."})

    pie = [{"region_id": row["region_id"], "label": row["label"], "color_rgb": row["color_rgb"],
            "attributed_area_px": row["area"]["attributed_area_px"], "food_area_pct": row["area"]["food_area_pct"]}
           for row in rows if row["area"] and row["area"]["attributed_area_px"] > 0]

    return {
        "result_id": record.result_id, "ready": status is not None, "missing_inputs": missing,
        "allocation_status": status, "validation_status": "estimate_unvalidated",
        "completeness_status": "complete" if complete else "incomplete", "incomplete_reasons": reasons,
        "plate": record.plate["geometry"].to_dict() if record.plate else None,
        "plate_source": record.plate["input"].get("source", "manual") if record.plate else None,   # manual | saved_calibration | automatic_detection
        "plate_analysis": analysis.to_dict() if analysis else None,
        "food_area": ({"unique_food_area_px": analysis.food_area_px, "plate_coverage_pct": analysis.plate_coverage_pct,
                       "overlap_px": analysis.overlap_px, "overlap_policy": analysis.overlap_policy,
                       "distribution": pie, "note": "Computed from mask pixels; the pie chart only draws these numbers."} if analysis else None),
        "regions": rows,
        "total_weight": record.total_weight,
        "allocation": ({"status": alloc.status, "method": alloc.method, "allocated_weight_g": alloc.allocated_weight_g,
                        "unresolved_weight_g": alloc.unresolved_weight_g, "mass_conserved": alloc.mass_conserved,
                        "difference_g": round(record.total_weight["net_weight_g"] - alloc.allocated_weight_g - alloc.unresolved_weight_g, 3)
                        if alloc.status != "rejected" else None,
                        "disclaimer": alloc.disclaimer, "coverage_flags": alloc.coverage_flags} if alloc else None),
        "nutrition": {"totals": ({k: (None if v is None else round(v, 2)) for k, v in totals.items()} if included else None),
                      "included_regions": included, "excluded_regions": excluded_nut, "complete": nutrition_complete,
                      "note": "USDA per-100 g values scaled by the ESTIMATED weights; covers confirmed foods with a selected USDA record only."},
        "warnings": warnings, "disclaimer": DISCLAIMER,
    }


# ------------------------------------------------------------------ meal snapshot
def meal_snapshot(record: InferenceRecord, analysis: dict, nutri: NutritionService) -> dict:
    """Self-contained copy of a finalized analysis: nothing in it depends on the in-memory record or on later USDA edits."""
    items = []
    for row in analysis["regions"]:
        al = row["allocation"]
        if not al or al["status"] != "allocated":
            continue
        per100 = nutri.food(row["usda"]["fdc_id"])["per_100g"] if row["usda"] else None
        items.append({"region_id": row["region_id"], "region_source": row["source"], "label": row["label"], "taxonomy": row["taxonomy"],
                      "detected_segments": row["entity_count"]["detected_segments"], "user_entity_count": row["entity_count"]["user_count"],
                      "estimated_weight_g": al["estimated_weight_g"], "weight_per_entity_g": al["weight_per_entity_g"],
                      "area_px": row["area"]["attributed_area_px"], "food_area_pct": row["area"]["food_area_pct"],
                      "usda": row["usda"], "per_100g": per100, "nutrition": row["nutrition"]})
    unresolved = [{"region_id": r["region_id"], "unresolved_weight_g": r["allocation"]["unresolved_weight_g"]}
                  for r in analysis["regions"] if r["allocation"] and r["allocation"]["status"] == "unresolved"]
    geo, pa = analysis["plate"], analysis["plate_analysis"]
    return {
        "result_id": record.result_id, "filename": record.filename, "completeness_status": analysis["completeness_status"],
        "incomplete_reasons": analysis["incomplete_reasons"], "allocation_status": analysis["allocation_status"],
        "validation_status": analysis["validation_status"], "allocation_method": analysis["allocation"]["method"],
        "weight_source": record.total_weight["source"], "total_weight_g": record.total_weight["net_weight_g"],
        "allocated_weight_g": analysis["allocation"]["allocated_weight_g"], "unresolved_weight_g": analysis["allocation"]["unresolved_weight_g"],
        "plate_summary": {"center_x": geo["center_x"], "center_y": geo["center_y"], "radius_px": geo["radius_px"],
                          "image_width": geo["image_width"], "image_height": geo["image_height"],
                          "plate_diameter_mm": geo["plate_diameter_mm"], "plate_area_px": pa["plate_area_px"],
                          "unique_food_area_px": pa["food_area_px"], "plate_coverage_pct": pa["plate_coverage_pct"],
                          "overlap_policy": pa["overlap_policy"], "coverage_flags": pa["coverage_flags"]},
        "items": items, "unresolved": unresolved, "totals": analysis["nutrition"]["totals"],
        "warnings": [w["code"] for w in analysis["warnings"]], "disclaimer": analysis["disclaimer"],
    }
