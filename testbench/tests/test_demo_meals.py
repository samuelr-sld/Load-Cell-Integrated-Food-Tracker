"""The two demonstration meals against the REAL cached USDA database (testbench/data/nutrition.sqlite).
Skipped when the cache has not been imported. Expected numbers are the USDA SR Legacy per-100 g values printed by the importer."""
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from testbench.config import Settings
from testbench.crops import CropProcessor
from testbench.main import create_app
from testbench.nutrition import NutritionService
from testbench.orchestrator import InferencePipeline
from testbench.regions import MAX_WEIGHT_G
from .conftest import FakeClassifier, FakeSegmenter, detection, jpeg_bytes, rect_mask

DB = Settings().nutrition_db
pytestmark = pytest.mark.skipif(not Path(DB).exists() or NutritionService(DB).stats()["usda_records"] < 14, reason="USDA cache not imported")

SHAPE = (200, 300)
YOLO = {-1: "background", 23: "egg", 47: "chicken duck", 48: "sausage", 65: "rice"}
FOOD101 = ["hot_dog", "pizza"]
# FDC id -> (kcal, protein, carbs, fat) per 100 g, as imported from USDA
HOTDOG, EGG_BOILED, EGG_FRIED, RICE_COOKED, RICE_RAW = 174614, 173424, 173423, 168878, 169756
CHICKEN_BREAST, CHICKEN_SKIN = 171477, 171450
P100 = {HOTDOG: (322.0, 11.69, 2.66, 29.36), EGG_BOILED: (155.0, 12.58, 1.12, 10.61), RICE_COOKED: (130.0, 2.69, 28.17, 0.28),
        CHICKEN_BREAST: (165.0, 31.02, 0.0, 3.57)}


def serve(labels):
    """One image with one separate region per (class_id, name); returns (client, result_id)."""
    img = np.full((200, 300, 3), 128, dtype=np.uint8)
    dets = []
    for i, (cid, name) in enumerate(labels):
        m = rect_mask(SHAPE, 10 + 95 * i, 20, 90 + 95 * i, 120)
        img[m] = (60 * (i + 1), 90, 200 - 50 * i)
        dets.append(detection(i, m, class_id=cid, name=name, conf=0.8))
    s = Settings()
    pipe = InferencePipeline(FakeSegmenter(dets), CropProcessor(s.min_mask_pixels, s.min_crop_side), FakeClassifier(FOOD101 + ["x"] * 99), s,
                             torch.device("cpu"), {"device": {"resolved": "cpu"}}, YOLO)
    c = TestClient(create_app(pipeline=pipe, nutrition=NutritionService(DB)))
    c.__enter__()
    r = c.post("/api/inference", files=[("files", ("meal.jpg", jpeg_bytes(img, 100), "image/jpeg"))]).json()["items"][0]["result"]
    return c, r["result_id"], r


def confirm(c, rid, region, label, fdc=None, w=None, taxonomy="foodseg103"):
    body = {"label": label, "taxonomy": taxonomy, **({"fdc_id": fdc} if fdc else {}), **({"weight_g": w} if w else {})}
    return c.put(f"/api/results/{rid}/regions/{region}/identity", json=body)


def expect(fdc, w):
    return [v * w / 100 for v in P100[fdc]]


def test_all_four_demo_foods_are_listed():
    with TestClient(create_app(pipeline=None, nutrition=NutritionService(DB))) as c:
        listed = {(f["taxonomy"], f["label"]): f["num_usda_records"] for f in c.get("/api/nutrition/foods").json()["foods"]}
    assert listed[("food101", "hot dog")] == 1 and listed[("foodseg103", "sausage")] == 1       # hotdog
    assert listed[("foodseg103", "egg")] == 2 and listed[("foodseg103", "chicken duck")] == 2   # explicit preparation choice
    assert listed[("foodseg103", "rice")] == 2                                                   # raw vs cooked


def test_meal_1_hotdog_egg_rice():
    c, rid, res = serve([(48, "sausage"), (23, "egg"), (65, "rice")])
    assert [r["status"] for r in res["regions"]] == ["needs_confirmation"] * 3
    hd = confirm(c, rid, 1, "sausage", w=60).json()
    assert hd["nutrition"]["status"] == "ok" and hd["nutrition"]["usda_record"]["fdc_id"] == HOTDOG   # single record, user-confirmed label
    # egg and rice need an explicit preparation: nothing is guessed
    egg = confirm(c, rid, 2, "egg").json()["nutrition"]
    rice = confirm(c, rid, 3, "rice").json()["nutrition"]
    assert egg["status"] == rice["status"] == "needs_usda_selection" and egg["per_100g"] is None and rice["per_100g"] is None
    assert {x["preparation"] for x in rice["usda_candidates"]} == {"raw", "cooked"}
    assert confirm(c, rid, 2, "egg", fdc=EGG_BOILED, w=50).json()["nutrition"]["usda_record"]["description"] == "Egg, whole, cooked, hard-boiled"
    r3 = confirm(c, rid, 3, "rice", fdc=RICE_COOKED, w=150).json()["nutrition"]
    assert r3["usda_record"]["preparation"] == "cooked" and r3["per_100g"]["energy_kcal"] == 130.0
    assert r3["serving"]["energy_kcal"] == 195.0
    # a fried-egg pick would change the numbers; the cooked-rice pick must not be a raw record
    assert confirm(c, rid, 3, "rice", fdc=9999999).status_code == 422
    meal = c.get(f"/api/results/{rid}/nutrition").json()
    assert [i["region_id"] for i in meal["included_regions"]] == [1, 2, 3] and meal["excluded_regions"] == []
    want = np.sum([expect(HOTDOG, 60), expect(EGG_BOILED, 50), expect(RICE_COOKED, 150)], axis=0)
    got = [meal["totals"][k] for k in ("energy_kcal", "protein_g", "carbs_g", "fat_g")]
    assert got == pytest.approx(want, abs=0.05)
    assert meal["totals"]["energy_kcal"] == pytest.approx(193.2 + 77.5 + 195.0, abs=0.05)


def test_meal_2_chicken_rice():
    c, rid, _ = serve([(47, "chicken duck"), (65, "rice")])
    ch = confirm(c, rid, 1, "chicken duck").json()["nutrition"]
    assert ch["status"] == "needs_usda_selection" and {x["fdc_id"] for x in ch["usda_candidates"]} == {CHICKEN_BREAST, CHICKEN_SKIN}
    ch = confirm(c, rid, 1, "chicken duck", fdc=CHICKEN_BREAST, w=120).json()["nutrition"]
    assert ch["serving"]["carbs_g"] == 0.0 and ch["serving"]["protein_g"] == pytest.approx(37.22, abs=0.01)    # USDA carbs 0.0 is a value, not missing
    confirm(c, rid, 2, "rice", fdc=RICE_COOKED)                                   # confirmed but no weight yet
    meal = c.get(f"/api/results/{rid}/nutrition").json()
    assert [i["region_id"] for i in meal["included_regions"]] == [1]
    assert meal["excluded_regions"] == [{"region_id": 2, "label": "rice", "reason": "no portion weight entered"}]
    c.put(f"/api/results/{rid}/regions/2/portion", json={"weight_g": 180})
    meal = c.get(f"/api/results/{rid}/nutrition").json()
    want = np.sum([expect(CHICKEN_BREAST, 120), expect(RICE_COOKED, 180)], axis=0)
    assert [meal["totals"][k] for k in ("energy_kcal", "protein_g", "carbs_g", "fat_g")] == pytest.approx(want, abs=0.05)
    assert meal["totals"]["energy_kcal"] == pytest.approx(198.0 + 234.0, abs=0.05)
    # unconfirming drops a food from the total; invalid weights are refused
    c.delete(f"/api/results/{rid}/regions/1/identity")
    assert [i["region_id"] for i in c.get(f"/api/results/{rid}/nutrition").json()["included_regions"]] == [2]
    for bad in (0, -1, MAX_WEIGHT_G + 1):
        assert c.put(f"/api/results/{rid}/regions/2/portion", json={"weight_g": bad}).status_code == 422


def test_food101_hot_dog_label_also_resolves():
    c, rid, _ = serve([(48, "sausage")])
    n = confirm(c, rid, 1, "hot_dog", taxonomy="food101", w=60).json()["nutrition"]
    assert n["status"] == "ok" and n["usda_record"]["fdc_id"] == HOTDOG and n["usda_record"]["preparation"] == "prepared"
