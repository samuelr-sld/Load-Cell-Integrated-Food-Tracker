"""Plate analysis + total-weight allocation + meal log through the real FastAPI app, fake networks, REAL cached USDA database.
The two demonstration meals run end to end. Expected numbers are derived from the mask pixels / USDA per-100 g values, not hard-coded UI data."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from testbench.config import Settings
from testbench.crops import CropProcessor
from testbench.main import create_app
from testbench.meal_log import MealLog
from testbench.nutrition import NutritionService
from testbench.orchestrator import InferencePipeline
from testbench.plate_session import unpack_mask
from .conftest import FakeClassifier, FakeSegmenter, detection, jpeg_bytes, rect_mask

DB = Settings().nutrition_db
pytestmark = pytest.mark.skipif(not Path(DB).exists() or NutritionService(DB).stats()["usda_records"] < 14, reason="USDA cache not imported")

SHAPE = (200, 300)
PLATE = {"center_x": 150, "center_y": 100, "radius_px": 95}
YOLO = {-1: "background", 23: "egg", 47: "chicken duck", 48: "sausage", 65: "rice"}
RECTS = [(70, 70, 110, 130), (115, 70, 155, 130), (160, 70, 230, 130)]    # 2400, 2400, 4200 px, all inside the plate circle
HOTDOG, EGG_BOILED, RICE_COOKED, CHICKEN_BREAST = 174614, 173424, 168878, 171477
P100 = {HOTDOG: (322.0, 11.69, 2.66, 29.36), EGG_BOILED: (155.0, 12.58, 1.12, 10.61), RICE_COOKED: (130.0, 2.69, 28.17, 0.28),
        CHICKEN_BREAST: (165.0, 31.02, 0.0, 3.57)}
KEYS = ("energy_kcal", "protein_g", "carbs_g", "fat_g")


def make_client(labels, tmp_path, rects=RECTS, **overrides):
    img = np.full((200, 300, 3), 128, dtype=np.uint8)
    dets, masks = [], []
    for i, ((cid, name), rc) in enumerate(zip(labels, rects)):
        m = rect_mask(SHAPE, *rc)
        img[m] = (60 * (i + 1), 90, 200 - 50 * i)
        dets.append(detection(i, m, class_id=cid, name=name, conf=0.9 - 0.1 * i))
        masks.append(m)
    s = replace(Settings(), meals_db=tmp_path / "meals.sqlite", **overrides)
    pipe = InferencePipeline(FakeSegmenter(dets), CropProcessor(s.min_mask_pixels, s.min_crop_side), FakeClassifier(["hot_dog", "pizza"] + ["x"] * 99),
                             s, torch.device("cpu"), {"device": {"resolved": "cpu"}}, YOLO)
    c = TestClient(create_app(settings=s, pipeline=pipe, nutrition=NutritionService(DB)))
    c.__enter__()
    res = c.post("/api/inference", files=[("files", ("meal.jpg", jpeg_bytes(img, 100), "image/jpeg"))]).json()["items"][0]["result"]
    return c, res["result_id"], masks, s


def confirm(c, rid, region, label, fdc=None):
    return c.put(f"/api/results/{rid}/regions/{region}/identity", json={"label": label, "taxonomy": "foodseg103", **({"fdc_id": fdc} if fdc else {})})


def setup_meal(c, rid, total=300.0):
    assert c.put(f"/api/results/{rid}/plate", json=PLATE).status_code == 200
    assert c.put(f"/api/results/{rid}/total-weight", json={"net_weight_g": total, "source": "manual_simulation"}).status_code == 200


def analysis(c, rid):
    return c.get(f"/api/results/{rid}/plate-analysis").json()


def serving(fdc, g):
    return [v * g / 100 for v in P100[fdc]]


# ------------------------------------------------------------------ demonstration meals
def test_meal_1_hotdogs_egg_rice_end_to_end(tmp_path):
    c, rid, masks, s = make_client([(48, "sausage"), (23, "egg"), (65, "rice")], tmp_path)
    assert analysis(c, rid)["allocation_status"] is None and set(analysis(c, rid)["missing_inputs"]) == {"plate_calibration", "total_weight"}
    setup_meal(c, rid, 300.0)
    # nothing confirmed yet: the weight is held back, never handed to anyone
    a = analysis(c, rid)
    assert a["allocation_status"] == "allocation_partial" and a["allocation"]["unresolved_weight_g"] == 300.0
    assert a["nutrition"]["totals"] is None and a["completeness_status"] == "incomplete"

    confirm(c, rid, 1, "sausage")
    confirm(c, rid, 2, "egg", fdc=EGG_BOILED)
    confirm(c, rid, 3, "rice", fdc=RICE_COOKED)
    assert c.put(f"/api/results/{rid}/regions/1/entity-count", json={"user_entity_count": 2}).json()["user_entity_count"] == 2
    a = analysis(c, rid)
    assert a["allocation_status"] == "allocation_complete" and a["validation_status"] == "estimate_unvalidated"
    assert a["completeness_status"] == "complete" and a["incomplete_reasons"] == []

    # areas come from the real mask pixels (rectangles fully on the plate)
    px = [int(m.sum()) for m in masks]
    assert px == [2400, 2400, 4200]
    rows = {r["region_id"]: r for r in a["regions"]}
    assert [rows[i]["area"]["attributed_area_px"] for i in (1, 2, 3)] == px
    assert a["food_area"]["unique_food_area_px"] == sum(px)
    pie = {d["region_id"]: d["food_area_pct"] for d in a["food_area"]["distribution"]}
    assert pie == pytest.approx({i: 100 * p / sum(px) for i, p in zip((1, 2, 3), px)}, abs=1e-3) and sum(pie.values()) == pytest.approx(100, abs=1e-2)

    # weights = W * area share; reconcile with the total
    w = {i: rows[i]["allocation"]["estimated_weight_g"] for i in (1, 2, 3)}
    assert w == pytest.approx({1: 80.0, 2: 80.0, 3: 140.0}, abs=0.05)
    assert sum(w.values()) == pytest.approx(300.0, abs=1e-6) and a["allocation"]["mass_conserved"] and a["allocation"]["difference_g"] == pytest.approx(0, abs=1e-6)
    # detected segments and the user's count are separate; per-item weight uses the user's count only
    assert rows[1]["entity_count"] == {"detected_segments": 1, "user_count": 2} and rows[1]["allocation"]["weight_per_entity_g"] == 40.0
    assert rows[2]["entity_count"]["user_count"] is None and rows[2]["allocation"]["weight_per_entity_g"] is None

    # USDA nutrition from ESTIMATED weights; the legacy per-region weight was never touched
    assert rows[1]["usda"]["fdc_id"] == HOTDOG
    want = np.sum([serving(HOTDOG, w[1]), serving(EGG_BOILED, w[2]), serving(RICE_COOKED, w[3])], axis=0)
    assert [a["nutrition"]["totals"][k] for k in KEYS] == pytest.approx(want, abs=0.05)
    assert all(rows[i]["legacy_manual_weight_g"] is None for i in rows)
    assert c.get(f"/api/results/{rid}/nutrition").json()["totals"] is None          # legacy meal total: no manual weights entered

    # finalize -> snapshot persisted, duplicate save is a no-op
    r = c.post(f"/api/results/{rid}/meal", json={})
    assert r.status_code == 201
    meal = r.json()["meal"]
    assert meal["completeness_status"] == "complete" and meal["weight_source"] == "manual_simulation" and meal["allocation_method"] == "area_proportional"
    assert meal["total_weight_g"] == 300.0 and meal["validation_status"] == "estimate_unvalidated"
    assert [i["label"] for i in meal["items"]] == ["sausage", "egg", "rice"] and [i["fdc_id"] for i in meal["items"]] == [HOTDOG, EGG_BOILED, RICE_COOKED]
    assert sum(i["estimated_weight_g"] for i in meal["items"]) == pytest.approx(300.0) and meal["items"][0]["user_entity_count"] == 2
    assert meal["plate_summary"]["unique_food_area_px"] == sum(px) and meal["totals"] == a["nutrition"]["totals"]
    again = c.post(f"/api/results/{rid}/meal", json={})
    assert again.status_code == 200 and again.json()["duplicate"] and again.json()["meal"]["meal_id"] == meal["meal_id"]
    assert len(c.get("/api/meals").json()["meals"]) == 1

    # survives a server restart (new app, same file) and no longer needs the in-memory result
    c2 = TestClient(create_app(settings=s, pipeline=None, nutrition=NutritionService(DB)))
    saved = c2.get(f"/api/meals/{meal['meal_id']}").json()
    assert saved == meal and c2.get(f"/api/results/{rid}").status_code == 404
    assert c2.get("/api/meals/nope").status_code == 404


def test_meal_2_chicken_and_rice(tmp_path):
    c, rid, _, _ = make_client([(47, "chicken duck"), (65, "rice")], tmp_path, rects=[RECTS[0], RECTS[2]])     # 2400 : 4200
    setup_meal(c, rid, 270.0)
    confirm(c, rid, 1, "chicken duck", fdc=CHICKEN_BREAST)
    # chicken has two USDA records: without a pick the allocation is complete but the NUTRITION is not, so the meal is incomplete
    confirm(c, rid, 2, "rice")
    a = analysis(c, rid)
    assert a["allocation_status"] == "allocation_complete" and a["completeness_status"] == "incomplete" and "nutrition_incomplete" in a["incomplete_reasons"]
    assert c.post(f"/api/results/{rid}/meal", json={}).status_code == 409
    confirm(c, rid, 2, "rice", fdc=RICE_COOKED)
    a = analysis(c, rid)
    w = {r["region_id"]: r["allocation"]["estimated_weight_g"] for r in a["regions"]}
    assert w == pytest.approx({1: 270 * 2400 / 6600, 2: 270 * 4200 / 6600}, abs=0.06) and sum(w.values()) == pytest.approx(270.0, abs=1e-6)
    want = np.sum([serving(CHICKEN_BREAST, w[1]), serving(RICE_COOKED, w[2])], axis=0)
    assert [a["nutrition"]["totals"][k] for k in KEYS] == pytest.approx(want, abs=0.05)
    assert a["completeness_status"] == "complete"
    assert c.post(f"/api/results/{rid}/meal", json={}).json()["meal"]["completeness_status"] == "complete"


# ------------------------------------------------------------------ explicit states
def test_partial_meal_is_never_finalized_as_complete(tmp_path):
    c, rid, _, s = make_client([(48, "sausage"), (23, "egg"), (65, "rice")], tmp_path)
    setup_meal(c, rid, 300.0)
    confirm(c, rid, 1, "sausage"); confirm(c, rid, 2, "egg", fdc=EGG_BOILED)
    a = analysis(c, rid)                                          # rice (140 g worth of area) is unconfirmed
    assert a["allocation_status"] == "allocation_partial" and a["allocation"]["unresolved_weight_g"] == 140.0 and a["allocation"]["allocated_weight_g"] == 160.0
    assert any(w["code"] == "unresolved_regions" for w in a["warnings"]) and not a["allocation"]["mass_conserved"]
    r = c.post(f"/api/results/{rid}/meal", json={})
    assert r.status_code == 409 and r.json()["error"]["code"] == "incomplete_meal"
    assert c.get("/api/meals").json()["meals"] == []
    saved = c.post(f"/api/results/{rid}/meal", json={"accept_incomplete": True}).json()["meal"]
    assert saved["completeness_status"] == "incomplete" and saved["allocation_status"] == "allocation_partial"
    assert saved["unresolved_weight_g"] == 140.0 and saved["unresolved"] == [{"region_id": 3, "unresolved_weight_g": 140.0}]
    assert len(saved["items"]) == 2


def test_rejected_allocations_cannot_be_saved(tmp_path):
    c, rid, _, _ = make_client([(48, "sausage"), (23, "egg"), (65, "rice")], tmp_path)
    for i, (lab, f) in enumerate([("sausage", None), ("egg", EGG_BOILED), ("rice", RICE_COOKED)], start=1):
        confirm(c, rid, i, lab, f)
    assert c.post(f"/api/results/{rid}/meal", json={}).status_code == 409                  # inputs missing
    # plate somewhere with no food: confirmed foods have no area, so the allocation is rejected instead of inventing shares
    c.put(f"/api/results/{rid}/plate", json={"center_x": 40, "center_y": 40, "radius_px": 20})
    c.put(f"/api/results/{rid}/total-weight", json={"net_weight_g": 300})
    a = analysis(c, rid)
    assert a["allocation_status"] == "allocation_rejected" and a["completeness_status"] == "incomplete" and a["allocation"]["difference_g"] is None
    assert {i["code"] for i in a["warnings"]} >= {"missing_mask"}
    assert c.post(f"/api/results/{rid}/meal", json={"accept_incomplete": True}).status_code == 409
    assert c.get("/api/meals").json()["meals"] == []


def test_input_validation(tmp_path):
    c, rid, _, _ = make_client([(48, "sausage")], tmp_path, rects=[RECTS[0]])
    put = lambda path, body: c.put(f"/api/results/{rid}/{path}", json=body)
    for bad in (0, -5, 5001, "abc", None, True, "100", 200.05):          # no bool/str coercion; 0.1 g resolution keeps mass conserved
        assert put("total-weight", {"net_weight_g": bad}).status_code == 422
    assert put("total-weight", {"net_weight_g": 100, "source": "load_cell"}).status_code == 422          # no sensor connected yet
    assert put("plate", {"center_x": 10}).json()["error"]["code"] == "invalid_plate"
    assert put("plate", {"center_x": 10, "center_y": 10, "radius_px": 0}).status_code == 422
    assert put("plate", {"center_x": 999, "center_y": 10, "radius_px": 5}).status_code == 422             # centre outside the image
    assert put("plate", {"rim_points": [[10, 10], [20, 20], [30, 30]]}).status_code == 422                # collinear
    assert put("regions/1/entity-count", {"user_entity_count": 0}).status_code == 422
    assert put("regions/1/entity-count", {"user_entity_count": 2.5}).status_code == 422
    assert put("regions/1/entity-count", {"user_entity_count": True}).status_code == 422
    assert put("regions/1/excluded", {"excluded": "yes"}).status_code == 422
    assert c.post(f"/api/results/{rid}/meal", json={"accept_incomplete": "true"}).status_code == 422
    assert c.get("/api/results/nope/plate-analysis").status_code == 404
    assert c.put(f"/api/results/{rid}/plate", json=PLATE).status_code == 200
    assert c.delete(f"/api/results/{rid}/plate").json()["plate"] is None and analysis(c, rid)["plate"] is None


# ------------------------------------------------------------------ calibration, masks, regions
def test_full_masks_are_pixel_exact_and_align_with_crops(tmp_path):
    c, rid, masks, _ = make_client([(48, "sausage"), (65, "rice")], tmp_path, rects=[RECTS[0], RECTS[2]])
    rec = c.app.state.store.get(rid)
    h, w = rec.image["height"], rec.image["width"]
    for seg, m in zip(rec.segments, masks):
        full = unpack_mask(rec.segment_masks[seg.id], h, w)
        assert np.array_equal(full, m)
        # the stored crop PNG placed at box_xyxy reproduces the same pixels (so a record without full masks could be rebuilt)
        from io import BytesIO
        from PIL import Image
        crop = np.array(Image.open(BytesIO(rec.images[seg.image_keys["mask"]]))) > 0
        x1, y1, x2, y2 = seg.box_xyxy
        rebuilt = np.zeros((h, w), dtype=bool)
        rebuilt[y1:y2, x1:x2] = crop
        assert np.array_equal(rebuilt, full)
    assert rec.size_bytes() > sum(len(b) for b in rec.images.values())


def test_plate_rescaled_from_another_resolution(tmp_path):
    c, rid, _, _ = make_client([(48, "sausage")], tmp_path, rects=[RECTS[0]])
    r = c.put(f"/api/results/{rid}/plate", json={"center_x": 300, "center_y": 200, "radius_px": 190, "image_width": 600, "image_height": 400})
    assert r.status_code == 200 and (r.json()["plate"]["center_x"], r.json()["plate"]["radius_px"]) == (150, 95)
    assert r.json()["plate"]["image_width"] == 300
    assert c.put(f"/api/results/{rid}/plate", json={"center_x": 300, "center_y": 200, "radius_px": 190, "image_width": 600, "image_height": 800}).status_code == 422
    rim = c.put(f"/api/results/{rid}/plate", json={"rim_points": [[55, 100], [150, 5], [245, 100]], "plate_diameter_mm": 260}).json()["plate"]
    assert (rim["center_x"], rim["center_y"], rim["radius_px"]) == pytest.approx((150, 100, 95)) and rim["mm_per_px"] == pytest.approx(260 / 190)


def test_exclude_region_and_manual_added_food(tmp_path):
    c, rid, masks, _ = make_client([(48, "sausage"), (65, "rice")], tmp_path, rects=[RECTS[0], RECTS[2]])     # the egg was missed by YOLO
    setup_meal(c, rid, 100.0)
    confirm(c, rid, 1, "sausage"); confirm(c, rid, 2, "rice", fdc=RICE_COOKED)
    a = analysis(c, rid)
    assert a["allocation_status"] == "allocation_complete"
    assert any(w["code"] == "missed_food_limitation" for w in a["warnings"])           # shown even though the allocation is "complete"
    assert [r["allocation"]["estimated_weight_g"] for r in a["regions"]] == pytest.approx([100 * 2400 / 6600, 100 * 4200 / 6600], abs=0.06)

    bad = lambda poly: c.post(f"/api/results/{rid}/regions/manual", json={"polygon": poly})
    assert bad([[0, 0], [10, 10]]).status_code == 422 and bad([[0, 0], [10, 10], [999, 5]]).status_code == 422
    added = bad([[115, 70], [155, 70], [155, 130], [115, 130]])                         # outline of the missed egg
    assert added.status_code == 201 and added.json()["source"] == "manual" and added.json()["region_id"] == 3
    a = analysis(c, rid)
    assert a["allocation_status"] == "allocation_partial" and any(w["code"] == "manual_region" for w in a["warnings"])
    row = next(r for r in a["regions"] if r["region_id"] == 3)
    assert row["entity_count"]["detected_segments"] == 0 and row["area"]["attributed_area_px"] == 41 * 61           # cv2 polygon fill includes the border pixels
    confirm(c, rid, 3, "egg", fdc=EGG_BOILED)
    a = analysis(c, rid)
    assert a["allocation_status"] == "allocation_complete" and sum(r["allocation"]["estimated_weight_g"] for r in a["regions"]) == pytest.approx(100.0, abs=1e-6)
    assert c.delete(f"/api/results/{rid}/regions/1").status_code == 409              # detected regions are excluded, not deleted
    # excluding a detected region removes its area and weight share
    assert c.put(f"/api/results/{rid}/regions/1/excluded", json={"excluded": True}).json()["excluded"] is True
    a = analysis(c, rid)
    assert [r["region_id"] for r in a["regions"]] == [2, 3] and a["food_area"]["unique_food_area_px"] == 4200 + 41 * 61
    assert c.delete(f"/api/results/{rid}/regions/3").status_code == 200


def test_legacy_manual_weights_are_untouched_by_total_weight_workflow(tmp_path):
    c, rid, _, _ = make_client([(48, "sausage"), (65, "rice")], tmp_path, rects=[RECTS[0], RECTS[2]])
    confirm(c, rid, 1, "sausage"); confirm(c, rid, 2, "rice", fdc=RICE_COOKED)
    c.put(f"/api/results/{rid}/regions/1/portion", json={"weight_g": 55})
    setup_meal(c, rid, 200.0)
    a = analysis(c, rid)
    assert a["regions"][0]["legacy_manual_weight_g"] == 55 and a["regions"][0]["allocation"]["estimated_weight_g"] == pytest.approx(200 * 2400 / 6600, abs=0.06)
    detail = c.get(f"/api/results/{rid}").json()
    assert next(r for r in detail["regions"] if r["region_id"] == 1)["nutrition"]["weight_g"] == 55
    legacy = c.get(f"/api/results/{rid}/nutrition").json()
    assert [i["region_id"] for i in legacy["included_regions"]] == [1] and legacy["included_regions"][0]["weight_g"] == 55
    assert detail["total_weight"]["source"] == "manual_simulation" and detail["total_weight"]["simulated"] and detail["plate"]["radius_px"] == 95


# ------------------------------------------------------------------ backend configuration
def test_config_driven_policies(tmp_path):
    labels, rects = [(48, "sausage"), (65, "rice")], [RECTS[0], RECTS[2]]
    # optional food coefficient (test value only; the shipped default is empty)
    c, rid, _, s = make_client(labels, tmp_path / "a", rects=rects, food_coefficients={"rice": 2.0})
    assert Settings().food_coefficients == {}
    setup_meal(c, rid, 100.0); confirm(c, rid, 1, "sausage"); confirm(c, rid, 2, "rice", fdc=RICE_COOKED)
    w = [r["allocation"]["estimated_weight_g"] for r in analysis(c, rid)["regions"]]
    assert w == pytest.approx([100 * 2400 / (2400 + 2 * 4200), 100 * 8400 / (2400 + 8400)], abs=0.06)
    # strict mode: plate partly out of frame is a rejection, not a warning
    edge = {"center_x": 150, "center_y": 100, "radius_px": 140}
    for strict, expect in ((False, "allocation_complete"), (True, "allocation_rejected")):
        c, rid, _, _ = make_client(labels, tmp_path / f"s{strict}", rects=rects, strict_coverage=strict)
        c.put(f"/api/results/{rid}/plate", json=edge); c.put(f"/api/results/{rid}/total-weight", json={"net_weight_g": 100})
        confirm(c, rid, 1, "sausage"); confirm(c, rid, 2, "rice", fdc=RICE_COOKED)
        a = analysis(c, rid)
        assert a["allocation_status"] == expect and "plate_partially_out_of_frame" in a["allocation"]["coverage_flags"]
    # priority overlap policy is selectable and runs
    c, rid, _, _ = make_client(labels, tmp_path / "p", rects=rects, overlap_policy="priority")
    setup_meal(c, rid, 100.0)
    assert analysis(c, rid)["food_area"]["overlap_policy"] == "priority"


def test_app_creation_does_not_create_meal_db(tmp_path):
    s = replace(Settings(), meals_db=tmp_path / "x" / "meals.sqlite")
    with TestClient(create_app(settings=s, pipeline=None, nutrition=NutritionService(DB))) as c:
        c.get("/api/health")
    assert not s.meals_db.exists()


# ------------------------------------------------------------------ meal log transactions
def snapshot(**kw):
    item = {"region_id": 1, "region_source": "detected", "label": "rice", "taxonomy": "foodseg103", "detected_segments": 1, "user_entity_count": None,
            "estimated_weight_g": 100.0, "weight_per_entity_g": None, "area_px": 10.0, "food_area_pct": 100.0, "usda": {"fdc_id": 1, "description": "d", "data_source": "SR", "preparation": "cooked"},
            "per_100g": {}, "nutrition": {}}
    base = {"result_id": "r1", "filename": "f.jpg", "completeness_status": "complete", "incomplete_reasons": [], "allocation_status": "allocation_complete",
            "validation_status": "estimate_unvalidated", "allocation_method": "area_proportional", "weight_source": "manual_simulation", "total_weight_g": 100.0,
            "allocated_weight_g": 100.0, "unresolved_weight_g": 0.0, "plate_summary": {"radius_px": 1}, "items": [item], "unresolved": [], "totals": None,
            "warnings": [], "disclaimer": "d"}
    return {**base, **kw}


def test_meal_log_is_atomic_and_deduplicates(tmp_path):
    log = MealLog(tmp_path / "m.sqlite")
    bad = snapshot(items=[snapshot()["items"][0], {**snapshot()["items"][0], "region_id": 2, "label": None}])      # second item violates NOT NULL
    with pytest.raises(Exception):
        log.save(bad)
    assert log.list() == []                                                    # the meal row was rolled back with the failed item
    meal, created = log.save(snapshot())
    assert created and log.save(snapshot())[1] is False and len(log.list()) == 1
    changed, created2 = log.save(snapshot(total_weight_g=120.0))
    assert created2 and changed["meal_id"] != meal["meal_id"] and len(log.list()) == 2
    log.close()
    assert MealLog(tmp_path / "m.sqlite").get(meal["meal_id"]) == meal         # reopened file returns the identical snapshot
