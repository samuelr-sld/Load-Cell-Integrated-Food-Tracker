"""/api/camera/* and POST /api/results/{id}/plate/auto through the real FastAPI app with a FAKE camera and the fake networks.
The full-pipeline test also uses the real cached USDA database (skipped if it has not been imported)."""
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from testbench.camera import CameraService
from testbench.config import Settings
from testbench.crops import CropProcessor
from testbench.main import create_app
from testbench.nutrition import NutritionService
from testbench.orchestrator import InferencePipeline
from .conftest import FakeClassifier, FakeSegmenter, detection, rect_mask
from .synthetic_plate import flat_gray, plate_scene
from .test_camera import Devices, FakeCap

DB = Settings().nutrition_db
needs_usda = pytest.mark.skipif(not Path(DB).exists() or NutritionService(DB).stats()["usda_records"] < 14, reason="USDA cache not imported")

SHAPE = (480, 640)
RECTS = [(200, 170, 260, 250), (270, 170, 330, 250), (340, 170, 440, 250)]           # all inside the synthetic plate (r = 201.6)
YOLO = {-1: "background", 23: "egg", 48: "sausage", 65: "rice"}
EGG_BOILED, RICE_COOKED = 173424, 168878
TRUE = (320.0, 240.0, 201.6)


def frame_of(rgb):
    return np.ascontiguousarray(rgb[..., ::-1])                                    # the camera hands out BGR


def make(tmp_path, frames=None, timeout_s=2.0, with_pipeline=True, mask_scale=1, **overrides):
    """frames: {camera index: RGB scene}. A mutable dict, so a test can change what the 'camera' sees between captures."""
    frames = {1: plate_scene()} if frames is None else frames

    class LiveDevices(Devices):
        def __call__(self, source, flag):
            self.caps = {k: FakeCap(frame_of(v)) for k, v in frames.items()}      # a fresh fake device per open, showing the current scene
            return super().__call__(source, flag)

    dev = LiveDevices({})
    s = replace(Settings(), meals_db=tmp_path / "meals.sqlite", camera_state_file=tmp_path / "camera_state.json", camera_source="1", **overrides)
    pipe = None
    if with_pipeline:
        shape = (SHAPE[0] * mask_scale, SHAPE[1] * mask_scale)       # the fake segmenter's masks must match the frame it is shown
        masks = [rect_mask(shape, *(v * mask_scale for v in rc)) for rc in RECTS]
        dets = [detection(i, m, class_id=cid, name=n, conf=0.9 - 0.1 * i) for i, (m, (cid, n)) in enumerate(zip(masks, [(48, "sausage"), (23, "egg"), (65, "rice")]))]
        pipe = InferencePipeline(FakeSegmenter(dets), CropProcessor(s.min_mask_pixels, s.min_crop_side), FakeClassifier(["hot_dog", "pizza"] + ["x"] * 99),
                                 s, torch.device("cpu"), {"device": {"resolved": "cpu"}}, YOLO)
    cam = CameraService("any", 0, 0, warmup_frames=0, timeout_s=timeout_s, max_probe_index=s.camera_max_probe_index, capture_factory=dev)
    c = TestClient(create_app(settings=s, pipeline=pipe or object(), nutrition=NutritionService(DB) if Path(DB).exists() else None, camera=cam))
    c.__enter__()
    return c, dev, frames


def capture(c, **body):
    r = c.post("/api/camera/capture", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def upload_capture(c, cap):
    jpg = c.get(cap["image_url"])
    assert jpg.status_code == 200 and jpg.headers["content-type"] == "image/jpeg"
    r = c.post("/api/inference", files=[("files", (cap["filename"], jpg.content, "image/jpeg"))], data={"mode": "yolo_only"})
    assert r.status_code == 200, r.text
    return r.json()["items"][0]["result"]


# ------------------------------------------------------------------ status, devices, selection
def test_status_never_opens_the_camera_and_reports_defaults(tmp_path):
    c, dev, _ = make(tmp_path)
    s = c.get("/api/camera/status").json()
    assert s["selected_camera"] == {"source": 1, "camera_id": "index:1", "persisted": False}
    assert s["localization_mode"] == "automatic_with_fallback" and s["calibration"] is None and s["latest_capture"] is None
    assert dev.opened == []


def test_devices_lists_every_probed_index_with_actual_frame_size(tmp_path):
    c, dev, _ = make(tmp_path, frames={1: plate_scene(720, 1280)}, camera_max_probe_index=3)
    d = c.get("/api/camera/devices").json()
    assert [x["camera_id"] for x in d["devices"]] == ["index:0", "index:1", "index:2"]
    assert d["available"] == ["index:1"] and d["selected"] == "index:1"
    one = d["devices"][1]
    assert one["capture_ok"] and (one["width"], one["height"]) == (1280, 720)
    assert d["devices"][0]["error_code"] == "camera_unavailable"
    assert c.get("/api/camera/devices?max_index=0").status_code == 422


def test_test_endpoint_reports_failure_as_data_not_a_server_error(tmp_path):
    c, _, _ = make(tmp_path)
    ok = c.post("/api/camera/test", json={"source": 1}).json()
    bad = c.post("/api/camera/test", json={"source": 4})
    assert ok["capture_ok"] and (ok["width"], ok["height"]) == (640, 480)
    assert bad.status_code == 200 and bad.json()["capture_ok"] is False and bad.json()["error_code"] == "camera_unavailable"
    assert c.post("/api/camera/test", json={"source": -3}).status_code == 422
    assert c.post("/api/camera/test", json={"source": "rtsp://x"}).status_code == 422


def test_selection_persists_across_app_restart_and_rejects_bad_sources(tmp_path):
    c, _, _ = make(tmp_path)
    assert c.put("/api/camera/selection", json={"source": 2}).json()["selected_camera"]["camera_id"] == "index:2"
    assert c.put("/api/camera/selection", json={"source": 99}).status_code == 422
    assert c.put("/api/camera/selection", json={}).status_code == 422
    c2, _, _ = make(tmp_path)                                                       # new app, same state file
    assert c2.get("/api/camera/status").json()["selected_camera"] == {"source": 2, "camera_id": "index:2", "persisted": True}


def test_localization_mode_can_be_changed_and_validated(tmp_path):
    c, _, _ = make(tmp_path)
    assert c.put("/api/camera/localization-mode", json={"mode": "fixed_calibration"}).json() == {"localization_mode": "fixed_calibration"}
    assert c.get("/api/camera/status").json()["localization_mode"] == "fixed_calibration"
    assert c.put("/api/camera/localization-mode", json={"mode": "psychic"}).status_code == 422


# ------------------------------------------------------------------ capture
def test_capture_returns_metadata_and_a_decodable_jpeg_of_the_real_frame_size(tmp_path):
    c, dev, _ = make(tmp_path)
    cap = capture(c)
    assert (cap["width"], cap["height"], cap["camera_id"]) == (640, 480, "index:1") and cap["capture_id"] and cap["filename"].endswith(".jpg")
    img = c.get(cap["image_url"])
    assert img.content[:2] == b"\xff\xd8"
    assert all(x.released for _, x in dev.opened)                                  # camera released after the still
    assert c.get("/api/camera/status").json()["latest_capture"]["capture_id"] == cap["capture_id"]
    assert c.get("/api/camera/capture/image?capture_id=nope").status_code == 404


def test_capture_from_an_unavailable_camera_is_a_clean_409(tmp_path):
    c, _, _ = make(tmp_path)
    r = c.post("/api/camera/capture", json={"source": 3})
    assert r.status_code == 409 and r.json()["error"]["code"] == "camera_unavailable"
    assert c.get("/api/health").status_code in (200, 503)                          # the server is alive afterwards


def test_capture_with_invalid_index_is_422(tmp_path):
    c, dev, _ = make(tmp_path)
    r = c.post("/api/camera/capture", json={"source": 500})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_camera" and dev.opened == []


def test_hung_camera_returns_504_within_the_timeout_instead_of_blocking(tmp_path):
    import time
    gate = threading.Event()
    s = replace(Settings(), meals_db=tmp_path / "m2.sqlite", camera_state_file=tmp_path / "s2.json", camera_source="1")
    cam = CameraService("any", 0, 0, 0, 0.3, capture_factory=Devices({1: FakeCap(hang=gate)}))
    c = TestClient(create_app(settings=s, pipeline=object(), camera=cam))
    c.__enter__()
    t0 = time.perf_counter()
    r = c.post("/api/camera/capture", json={})
    assert r.status_code == 504 and r.json()["error"]["code"] == "camera_timeout" and time.perf_counter() - t0 < 3
    assert c.post("/api/camera/capture", json={}).json()["error"]["code"] == "camera_busy"      # the stuck driver call still owns the device
    gate.set()


def test_detect_and_localize_need_a_capture_first(tmp_path):
    c, _, _ = make(tmp_path)
    assert c.post("/api/camera/detect-plate", json={}).status_code == 404
    assert c.post("/api/camera/localize", json={}).json()["error"]["code"] == "no_capture"
    assert c.get("/api/camera/capture/image").status_code == 404


# ------------------------------------------------------------------ detection + calibration endpoints
def test_detect_plate_on_the_captured_frame(tmp_path):
    c, _, _ = make(tmp_path)
    cap = capture(c)
    d = c.post("/api/camera/detect-plate", json={"capture_id": cap["capture_id"]}).json()
    assert d["status"] == "detected" and d["capture_id"] == cap["capture_id"] and d["camera_id"] == "index:1"
    p = d["plate"]
    assert abs(p["center_x"] - TRUE[0]) < 3 and abs(p["center_y"] - TRUE[1]) < 3 and abs(p["radius_px"] - TRUE[2]) < 3
    assert d["evidence"]["evidence_score"] > 0.7 and "NOT a calibrated probability" in d["evidence_note"]
    assert c.post("/api/camera/detect-plate", json={"capture_id": "stale"}).status_code == 404


def test_detect_plate_failure_is_explicit_and_carries_no_geometry(tmp_path):
    c, _, _ = make(tmp_path, frames={1: flat_gray()})
    capture(c)
    d = c.post("/api/camera/detect-plate", json={}).json()
    assert d["status"] == "failed" and d["plate"] is None and d["reason"] == "low_contrast_image"


def test_calibration_save_get_replace_clear_cycle(tmp_path):
    c, _, _ = make(tmp_path)
    assert c.get("/api/camera/calibration").json() == {"camera_id": "index:1", "calibration": None, "all_cameras": []}
    cap = capture(c)
    r = c.put("/api/camera/calibration", json={"center_x": 320, "center_y": 240, "radius_px": 201.6, "image_width": 640, "image_height": 480,
                                                "plate_diameter_mm": 270.0, "method": "automatic_opencv"})
    assert r.status_code == 200 and r.json()["saved"]["method"] == "automatic_opencv"
    cal = c.get("/api/camera/calibration").json()["calibration"]
    assert (cal["camera_id"], cal["image_width"], cal["image_height"], cal["radius_px"], cal["plate_diameter_mm"]) == ("index:1", 640, 480, 201.6, 270.0)
    # rim points replace it, defaulting the image size to the latest capture of this camera
    r = c.put("/api/camera/calibration", json={"rim_points": [[521.6, 240], [320, 38.4], [118.4, 240]]})
    assert r.json()["saved"]["method"] == "manual_three_point" and r.json()["saved"]["radius_px"] == pytest.approx(201.6, abs=1e-6)
    assert c.get("/api/camera/status").json()["calibration"]["method"] == "manual_three_point"
    assert c.delete("/api/camera/calibration").json() == {"camera_id": "index:1", "cleared": True, "calibration": None}
    assert c.delete("/api/camera/calibration").json()["cleared"] is False
    assert c.get("/api/camera/calibration").json()["calibration"] is None
    assert cap["capture_id"]


@pytest.mark.parametrize("body", [
    {"center_x": 320, "center_y": 240, "radius_px": -5, "image_width": 640, "image_height": 480},
    {"center_x": 900, "center_y": 240, "radius_px": 50, "image_width": 640, "image_height": 480},
    {"rim_points": [[0, 0], [10, 10], [20, 20]], "image_width": 640, "image_height": 480},         # collinear
    {"center_x": 320, "center_y": 240, "image_width": 640, "image_height": 480},                   # no radius
    {"center_x": "320", "center_y": 240, "radius_px": 50, "image_width": 640, "image_height": 480},  # strict types
    {"center_x": 320, "center_y": 240, "radius_px": 50},                                           # no size and no capture yet
    {"camera_id": "banana", "center_x": 320, "center_y": 240, "radius_px": 50, "image_width": 640, "image_height": 480},
])
def test_invalid_calibrations_are_rejected_and_nothing_is_saved(tmp_path, body):
    c, _, _ = make(tmp_path)
    assert c.put("/api/camera/calibration", json=body).status_code == 422
    assert c.get("/api/camera/calibration").json()["calibration"] is None


def test_calibrations_are_per_camera(tmp_path):
    c, _, _ = make(tmp_path)
    c.put("/api/camera/calibration", json={"camera_id": "index:2", "center_x": 100, "center_y": 100, "radius_px": 50, "image_width": 300, "image_height": 300})
    assert c.get("/api/camera/calibration").json()["calibration"] is None             # selected camera is index:1
    assert c.get("/api/camera/calibration?camera_id=index:2").json()["calibration"]["radius_px"] == 50
    assert c.get("/api/camera/calibration").json()["all_cameras"] == ["index:2"]


def test_localize_endpoint_follows_the_priority_order(tmp_path):
    c, _, frames = make(tmp_path)
    capture(c)
    first = c.post("/api/camera/localize", json={}).json()
    assert first["status"] == "localized" and first["source"] == "automatic_detection"           # nothing saved yet
    c.put("/api/camera/calibration", json={"center_x": 320, "center_y": 240, "radius_px": 201.6})
    second = c.post("/api/camera/localize", json={}).json()
    assert second["source"] == "saved_calibration" and second["detection"] is None
    frames[1] = plate_scene(cx=380, cy=250)                                                       # the plate moved
    capture(c)
    third = c.post("/api/camera/localize", json={}).json()
    assert third["source"] == "automatic_detection" and abs(third["plate"]["center_x"] - 380) < 3
    frames[1] = flat_gray()                                                                        # nothing usable at all
    capture(c)
    fourth = c.post("/api/camera/localize", json={}).json()
    assert fourth["status"] == "needs_manual" and fourth["plate"] is None


# ------------------------------------------------------------------ result integration
def test_captured_frame_goes_through_the_unchanged_inference_endpoint_and_gets_auto_plate(tmp_path):
    c, _, _ = make(tmp_path)
    res = upload_capture(c, capture(c))
    rid = res["result_id"]
    assert res["plate"] is None                                                                    # inference alone does not touch the plate
    before = {k: bytes(v) for k, v in c.app.state.store.get(rid).segment_masks.items()}
    r = c.post(f"/api/results/{rid}/plate/auto", json={"camera_id": "index:1"}).json()
    assert r["applied"] and r["status"] == "localized" and r["source"] == "automatic_detection"
    assert abs(r["plate"]["center_x"] - TRUE[0]) < 3 and abs(r["plate"]["radius_px"] - TRUE[2]) < 3
    assert r["plate"]["input"]["source"] == "automatic_detection" and r["plate"]["input"]["evidence"]["edge_support"] > 0.7
    detail = c.get(f"/api/results/{rid}").json()
    assert detail["plate"]["input"]["source"] == "automatic_detection" and detail["plate"]["image_width"] == 640
    assert c.get(f"/api/results/{rid}/plate-analysis").json()["plate_source"] == "automatic_detection"
    assert {k: bytes(v) for k, v in c.app.state.store.get(rid).segment_masks.items()} == before   # masks untouched by localization


def test_saved_calibration_is_used_for_camera_frames_but_not_for_plain_uploads(tmp_path):
    c, _, _ = make(tmp_path)
    c.put("/api/camera/calibration", json={"center_x": 320, "center_y": 240, "radius_px": 201.6, "image_width": 640, "image_height": 480, "plate_diameter_mm": 270})
    rid = upload_capture(c, capture(c))["result_id"]
    cam = c.post(f"/api/results/{rid}/plate/auto", json={"camera_id": "index:1", "mode": "fixed_calibration"}).json()
    assert cam["applied"] and cam["source"] == "saved_calibration" and cam["plate"]["plate_diameter_mm"] == 270
    assert c.get(f"/api/results/{rid}/plate-analysis").json()["plate_source"] == "saved_calibration"
    plain = c.post(f"/api/results/{rid}/plate/auto", json={"mode": "fixed_calibration"}).json()       # no camera id: an arbitrary upload
    assert plain["status"] == "needs_manual" and plain["applied"] is False


def test_scaled_calibration_is_applied_to_a_higher_resolution_frame(tmp_path):
    c, _, frames = make(tmp_path, frames={1: plate_scene(960, 1280)}, mask_scale=2)
    c.put("/api/camera/calibration", json={"center_x": 320, "center_y": 240, "radius_px": 201.6, "image_width": 640, "image_height": 480})
    rid = upload_capture(c, capture(c))["result_id"]
    r = c.post(f"/api/results/{rid}/plate/auto", json={"camera_id": "index:1", "mode": "fixed_calibration"}).json()
    assert r["source"] == "saved_calibration" and r["steps"][0]["scaled_from"] == [640, 480]
    assert (r["plate"]["image_width"], r["plate"]["center_x"]) == (1280, 640.0) and r["plate"]["radius_px"] == pytest.approx(403.2)


def test_failed_localization_requests_manual_calibration_and_keeps_the_manual_fallback_working(tmp_path):
    c, _, _ = make(tmp_path, frames={1: flat_gray()})
    rid = upload_capture(c, capture(c))["result_id"]
    r = c.post(f"/api/results/{rid}/plate/auto", json={"camera_id": "index:1"}).json()
    assert r["status"] == "needs_manual" and r["applied"] is False and r["plate"] is None and "manually" in r["message"].lower()
    assert c.get(f"/api/results/{rid}/plate-analysis").json()["plate_source"] is None
    # the existing manual three-point route is unchanged and now reports its source as manual
    m = c.put(f"/api/results/{rid}/plate", json={"rim_points": [[521.6, 240], [320, 38.4], [118.4, 240]]})
    assert m.status_code == 200 and set(m.json()) == {"result_id", "plate", "input"} and m.json()["input"]["source"] == "manual"
    assert c.get(f"/api/results/{rid}/plate-analysis").json()["plate_source"] == "manual"
    # a later failed automatic attempt must not wipe the user's manual plate
    again = c.post(f"/api/results/{rid}/plate/auto", json={"camera_id": "index:1"}).json()
    assert again["applied"] is False
    assert c.get(f"/api/results/{rid}").json()["plate"]["input"]["source"] == "manual"


def test_plate_auto_validation_and_unknown_result(tmp_path):
    c, _, _ = make(tmp_path)
    assert c.post("/api/results/nope/plate/auto", json={}).status_code == 404
    rid = upload_capture(c, capture(c))["result_id"]
    assert c.post(f"/api/results/{rid}/plate/auto", json={"mode": "psychic"}).status_code == 422
    assert c.post(f"/api/results/{rid}/plate/auto", json={"camera_id": "banana"}).status_code == 422
    assert c.post(f"/api/results/{rid}/plate/auto").status_code == 200              # body is optional


def test_existing_api_contract_is_unchanged(tmp_path):
    c, _, _ = make(tmp_path)
    paths = set(c.get("/api/openapi.json").json()["paths"])
    for p in ("/api/inference", "/api/results", "/api/results/{result_id}", "/api/results/{result_id}/plate", "/api/results/{result_id}/total-weight",
              "/api/results/{result_id}/plate-analysis", "/api/results/{result_id}/meal", "/api/meals", "/api/health", "/api/nutrition/foods"):
        assert p in paths
    new = {"/api/camera/status", "/api/camera/devices", "/api/camera/test", "/api/camera/selection", "/api/camera/localization-mode", "/api/camera/capture",
           "/api/camera/capture/image", "/api/camera/detect-plate", "/api/camera/localize", "/api/camera/calibration", "/api/results/{result_id}/plate/auto"}
    assert new <= paths
    res = upload_capture(c, capture(c))
    assert {"result_id", "plate", "total_weight", "regions", "segments", "image", "image_urls"} <= set(res)       # same detail schema as an uploaded file
    assert c.get("/").status_code == 200 and "camera.js" in c.get("/").text and c.get("/static/camera.js").status_code == 200


# ------------------------------------------------------------------ full pipeline
def run_meal(c, rid, plate_call):
    plate_call(rid)
    assert c.put(f"/api/results/{rid}/total-weight", json={"net_weight_g": 300.0, "source": "manual_simulation"}).status_code == 200
    for region, label, fdc in ((1, "sausage", None), (2, "egg", EGG_BOILED), (3, "rice", RICE_COOKED)):
        r = c.put(f"/api/results/{rid}/regions/{region}/identity", json={"label": label, "taxonomy": "foodseg103", **({"fdc_id": fdc} if fdc else {})})
        assert r.status_code == 200, r.text
    return c.get(f"/api/results/{rid}/plate-analysis").json()


@needs_usda
def test_full_pipeline_camera_to_meal_and_localization_does_not_change_the_numbers(tmp_path):
    c, _, _ = make(tmp_path)
    # A: captured frame -> automatic plate
    a_id = upload_capture(c, capture(c))["result_id"]
    a = run_meal(c, a_id, lambda rid: assert_applied(c.post(f"/api/results/{rid}/plate/auto", json={"camera_id": "index:1"}).json()))
    # B: the same frame, the SAME circle entered manually through the pre-existing route
    b_id = upload_capture(c, capture(c))["result_id"]
    geo = a["plate"]
    b = run_meal(c, b_id, lambda rid: c.put(f"/api/results/{rid}/plate", json={"center_x": geo["center_x"], "center_y": geo["center_y"], "radius_px": geo["radius_px"],
                                                                                "image_width": geo["image_width"], "image_height": geo["image_height"]}))
    assert a["plate_source"] == "automatic_detection" and b["plate_source"] == "manual"
    assert a["allocation_status"] == "allocation_complete" and a["completeness_status"] == "complete" and a["allocation"]["mass_conserved"]
    # identical masks, areas, weights and nutrition regardless of how the circle was obtained
    for key in ("food_area", "allocation", "regions", "nutrition", "plate_analysis"):
        assert a[key] == b[key], key
    assert [r["area"]["attributed_area_px"] for r in a["regions"]] == [int(rect_mask(SHAPE, *rc).sum()) for rc in RECTS]
    assert sum(r["allocation"]["estimated_weight_g"] for r in a["regions"]) == pytest.approx(300.0, abs=1e-6)     # manual total weight preserved
    assert a["nutrition"]["totals"]["energy_kcal"] > 0
    # finalize: the same persistent meal route as before
    m = c.post(f"/api/results/{a_id}/meal", json={})
    assert m.status_code == 201 and m.json()["meal"]["completeness_status"] == "complete" and m.json()["meal"]["total_weight_g"] == 300.0
    assert c.get("/api/meals").json()["meals"][0]["meal_id"] == m.json()["meal"]["meal_id"]


def assert_applied(r):
    assert r["applied"] and r["source"] == "automatic_detection"
