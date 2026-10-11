"""Duplicate-region handling, identity confirmation, USDA mapping/lookup and nutrient maths.
The nutrition DB here holds *test fixture* rows (invented numbers, tmp file); the shipped DB only ever comes from the USDA API."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from testbench.main import create_app
from testbench.nutrition import NutritionService, connect, serving_nutrition
from testbench.regions import group_by_mask_overlap, mask_iou
from .conftest import circle_mask, detection, jpeg_bytes, make_pipeline, rect_mask

SHAPE = (200, 300)


# ------------------------------------------------------------------ duplicate grouping
def test_strongly_overlapping_masks_form_one_region_with_highest_confidence_primary():
    a, b = rect_mask(SHAPE, 20, 20, 100, 100), rect_mask(SHAPE, 22, 22, 100, 100)
    groups = group_by_mask_overlap([(1, 0.5, a), (2, 0.9, b)], 0.7)
    assert [[sid for sid, _ in g] for g in groups] == [[2, 1]]


def test_adjacent_portions_with_overlapping_boxes_stay_separate():
    # two diagonal discs: bounding boxes overlap heavily, masks barely touch
    a, b = circle_mask(SHAPE, 60, 60, 40), circle_mask(SHAPE, 120, 120, 40)
    ba, bb = detection(0, a).box_xyxy, detection(1, b).box_xyxy
    assert min(ba[2], bb[2]) > max(ba[0], bb[0]) and min(ba[3], bb[3]) > max(ba[1], bb[1])   # boxes really overlap
    assert mask_iou(a, b) < 0.1
    assert len(group_by_mask_overlap([(1, 0.9, a), (2, 0.8, b)], 0.7)) == 2


def test_small_item_inside_a_large_one_is_not_a_duplicate():
    big, small = rect_mask(SHAPE, 10, 10, 200, 150), rect_mask(SHAPE, 50, 50, 70, 70)
    assert len(group_by_mask_overlap([(1, 0.9, big), (2, 0.8, small)], 0.7)) == 2


def test_chain_of_overlaps_does_not_merge_ends():
    m = [rect_mask(SHAPE, x, 20, x + 100, 120) for x in (0, 40, 80)]
    groups = group_by_mask_overlap([(1, 0.9, m[0]), (2, 0.8, m[1]), (3, 0.7, m[2])], 0.4)
    assert [[sid for sid, _ in g] for g in groups] == [[1, 2], [3]]        # 3 overlaps 2 but not the primary 1


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def nutri(tmp_path):
    db = tmp_path / "n.sqlite"
    conn = connect(db)
    rows = [(1001, "Test rice, raw", "SR Legacy", "raw", 360.0, 7.0, 80.0, 0.6),
            (1002, "Test rice, cooked", "SR Legacy", "cooked", 130.0, 2.7, 28.0, 0.3),
            (2001, "Test apple, raw", "Foundation", "raw", 52.0, 0.3, 14.0, None)]
    for r in rows:
        conn.execute("INSERT INTO foods VALUES (?,?,?,?,?,?,?,?,'reported','2026-01-01')", r)
    for tax, label, fdc in [("foodseg103", "red thing", 1001), ("foodseg103", "red thing", 1002), ("foodseg103", "blue thing", 2001),
                            ("food101", "red", 2001)]:
        conn.execute("INSERT INTO label_map VALUES (?,?,?)", (tax, label, fdc))
    conn.commit()
    return NutritionService(db)


@pytest.fixture
def client(nutri):
    img = np.full((200, 300, 3), 128, dtype=np.uint8)
    a, a2, b = rect_mask(SHAPE, 20, 20, 100, 100), rect_mask(SHAPE, 22, 22, 100, 100), circle_mask(SHAPE, 220, 120, 40)
    img[a], img[b] = (255, 0, 0), (0, 0, 255)
    dets = [detection(0, a, class_id=0, name="red-thing", conf=0.9), detection(1, a2, class_id=3, name="egg tart", conf=0.6),
            detection(2, b, class_id=2, name="blue-thing", conf=0.8)]
    pipe, _, _ = make_pipeline(detections=dets)
    with TestClient(create_app(pipeline=pipe, nutrition=nutri)) as c:
        c.img = jpeg_bytes(img, 100)
        r = c.post("/api/inference", files=[("files", ("m.jpg", c.img, "image/jpeg"))])
        c.res = r.json()["items"][0]["result"]
        c.rid = c.res["result_id"]
        yield c


def url(c, region, tail):
    return f"/api/results/{c.rid}/regions/{region}/{tail}"


# ------------------------------------------------------------------ pipeline integration
def test_duplicates_collapse_to_regions_but_originals_are_kept(client):
    res = client.res
    assert len(res["segments"]) == 3 and len(res["regions"]) == 2 and res["num_regions"] == 2
    r1 = res["regions"][0]
    assert r1["primary_segment_id"] == 1 and r1["segment_ids"] == [1, 2] and r1["duplicate_segment_ids"] == [2]
    seg2 = res["segments"][1]
    assert seg2["region_id"] == 1 and seg2["duplicate_of"] == 1 and seg2["mask_iou_with_primary"] > 0.7
    assert {c["label"] for c in r1["candidates"] if c["source"] == "yolo"} == {"red thing", "egg tart"}
    assert any(c["source"] == "mobilenet" for c in r1["candidates"])          # compare mode adds dish suggestions
    assert all(r["status"] == "needs_confirmation" and r["identity"] is None for r in res["regions"])
    assert r1["nutrition"]["status"] == "needs_confirmation"
    # backward compatible fields untouched
    assert {"id", "predictions", "top1", "primary_identity", "yolo_feedback"} <= set(res["segments"][0])


def test_yolo_only_regions_have_no_dish_suggestions(client):
    r = client.post("/api/inference", files=[("files", ("m.jpg", client.img, "image/jpeg"))], data={"mode": "yolo_only"})
    regions = r.json()["items"][0]["result"]["regions"]
    assert len(regions) == 2 and all(c["source"] == "yolo" for reg in regions for c in reg["candidates"])


# ------------------------------------------------------------------ confirmation
def test_high_confidence_never_auto_confirms_then_user_confirms_and_corrects(client):
    assert client.get(f"/api/results/{client.rid}").json()["regions"][1]["status"] == "needs_confirmation"
    r = client.put(url(client, 2, "identity"), json={"label": "blue-thing", "taxonomy": "foodseg103"})
    assert r.status_code == 200 and r.json()["status"] == "confirmed" and r.json()["identity"]["from_suggestion"] is True
    assert r.json()["nutrition"]["status"] == "ok" and r.json()["nutrition"]["usda_record"]["fdc_id"] == 2001   # single record -> used
    # correction to a custom label: unmapped, not a guess
    r = client.put(url(client, 2, "identity"), json={"label": "family stew", "taxonomy": "custom"})
    assert r.json()["identity"]["from_suggestion"] is False and r.json()["nutrition"]["status"] == "unmapped"
    assert r.json()["nutrition"]["per_100g"] is None
    # un-confirm
    r = client.delete(url(client, 2, "identity"))
    assert r.json()["status"] == "needs_confirmation" and r.json()["nutrition"]["per_100g"] is None


def test_label_must_belong_to_its_taxonomy(client):
    r = client.put(url(client, 1, "identity"), json={"label": "unobtainium", "taxonomy": "foodseg103"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "unknown_label"
    assert client.put(url(client, 9, "identity"), json={"label": "red-thing"}).status_code == 404


# ------------------------------------------------------------------ USDA selection + nutrient maths
def test_multiple_usda_records_require_an_explicit_choice(client):
    n = client.put(url(client, 1, "identity"), json={"label": "red thing"}).json()["nutrition"]
    assert n["status"] == "needs_usda_selection" and n["per_100g"] is None and n["usda_record"] is None
    assert {c["preparation"] for c in n["usda_candidates"]} == {"raw", "cooked"}
    bad = client.put(url(client, 1, "identity"), json={"label": "red thing", "fdc_id": 2001})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_fdc_id"
    n = client.put(url(client, 1, "identity"), json={"label": "red thing", "fdc_id": 1002}).json()["nutrition"]
    assert n["status"] == "ok" and n["usda_record"]["preparation"] == "cooked" and n["per_100g"]["energy_kcal"] == 130.0
    assert n["serving"] is None and "weight" in n["serving_message"]


def test_serving_needs_a_valid_manual_weight(client):
    client.put(url(client, 1, "identity"), json={"label": "red thing", "fdc_id": 1002})
    r = client.put(url(client, 1, "portion"), json={"weight_g": 150})
    assert r.json()["nutrition"]["serving"] == {"energy_kcal": 195.0, "protein_g": 4.05, "carbs_g": 42.0, "fat_g": 0.45}
    for bad in (0, -5, 10_000):
        assert client.put(url(client, 1, "portion"), json={"weight_g": bad}).status_code == 422
    assert client.put(url(client, 1, "portion"), json={"weight_g": None}).json()["nutrition"]["serving"] is None


def test_missing_nutrient_stays_null_not_zero(nutri):
    assert nutri.food(2001)["per_100g"]["fat_g"] is None
    assert serving_nutrition(nutri.food(2001)["per_100g"], 200) == {"energy_kcal": 104.0, "protein_g": 0.6, "carbs_g": 28.0, "fat_g": None}


def test_unmapped_and_empty_database(tmp_path):
    svc = NutritionService(tmp_path / "empty.sqlite")
    r = svc.lookup("foodseg103", "rice")
    assert r["status"] == "unmapped" and r["selected"] is None and r["candidates"] == []
    assert svc.lookup("custom", "rice")["status"] == "unmapped"


def test_meal_total_counts_each_region_once_and_lists_exclusions(client):
    client.put(url(client, 1, "identity"), json={"label": "red thing", "fdc_id": 1002, "weight_g": 100})
    client.put(url(client, 2, "identity"), json={"label": "blue thing"})        # confirmed, mapped, but no weight
    m = client.get(f"/api/results/{client.rid}/nutrition").json()
    assert m["totals"]["energy_kcal"] == 130.0                                   # duplicate segment 2 of region 1 not added
    assert [i["region_id"] for i in m["included_regions"]] == [1]
    assert m["excluded_regions"] == [{"region_id": 2, "label": "blue thing", "reason": "no portion weight entered"}]


def test_foods_listing(client):
    f = client.get("/api/nutrition/foods").json()
    assert f["foods"] and f["foods"][0].keys() >= {"taxonomy", "label", "num_usda_records"}
