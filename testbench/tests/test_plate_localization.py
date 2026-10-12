"""CalibrationStore + PlateLocalizer: persistence, scaling, invalidation, the three modes and the manual fallback."""
import json
from dataclasses import replace

import cv2
import pytest

from testbench.camera_calibration import CalibrationStore, geometry_of
from testbench.config import Settings
from testbench.plate_analysis import PlateGeometry
from testbench.plate_localization import PlateLocalizer
from .synthetic_plate import flat_gray, plate_scene, table

CAM = "index:1"
TRUE = PlateGeometry(320, 240, 201.6, 640, 480, 270.0)


def make(tmp_path, **overrides):
    store = CalibrationStore(tmp_path / "camera_state.json")
    return store, PlateLocalizer(replace(Settings(), camera_state_file=tmp_path / "camera_state.json", **overrides), store)


# ---------------------------------------------------------------- persistence
def test_calibration_round_trips_through_a_new_store_instance(tmp_path):
    CalibrationStore(tmp_path / "s.json").save(CAM, TRUE, "manual_three_point")
    cal = CalibrationStore(tmp_path / "s.json").get(CAM)            # a fresh object = a server restart
    assert (cal["image_width"], cal["image_height"], cal["center_x"], cal["center_y"], cal["radius_px"], cal["plate_diameter_mm"]) == (640, 480, 320, 240, 201.6, 270.0)
    assert cal["camera_id"] == CAM and cal["method"] == "manual_three_point" and cal["calibrated_at"].endswith("+00:00")
    assert geometry_of(cal) == TRUE


def test_saving_replaces_and_clearing_removes_only_that_camera(tmp_path):
    s = CalibrationStore(tmp_path / "s.json")
    s.save(CAM, TRUE, "manual_numeric")
    s.save("index:2", PlateGeometry(100, 100, 50, 300, 300), "manual_numeric")
    s.save(CAM, PlateGeometry(300, 200, 150, 640, 480), "automatic_opencv")      # replace
    assert s.get(CAM)["radius_px"] == 150 and s.get(CAM)["method"] == "automatic_opencv"
    assert s.clear(CAM) is True and s.get(CAM) is None and s.get("index:2") is not None
    assert s.clear(CAM) is False


def test_cameras_have_independent_calibrations_and_selection_and_mode_persist(tmp_path):
    s = CalibrationStore(tmp_path / "s.json")
    s.save("index:1", TRUE, "manual_numeric")
    s.set_selected_camera("2")
    s.set_localization_mode("fixed_calibration")
    s2 = CalibrationStore(tmp_path / "s.json")
    assert s2.selected_camera() == "2" and s2.localization_mode() == "fixed_calibration" and s2.get("index:2") is None and s2.get("index:1")
    with pytest.raises(ValueError):
        s2.set_localization_mode("anything")


def test_unknown_method_is_rejected_and_corrupt_file_is_treated_as_empty(tmp_path):
    s = CalibrationStore(tmp_path / "s.json")
    with pytest.raises(ValueError):
        s.save(CAM, TRUE, "guess")
    (tmp_path / "s.json").write_text("{not json", encoding="utf-8")
    assert s.get(CAM) is None and s.localization_mode() is None
    s.save(CAM, TRUE, "manual_numeric")                              # the next save replaces the corrupt file
    assert json.loads((tmp_path / "s.json").read_text())["calibrations"][CAM]["radius_px"] == 201.6


# ---------------------------------------------------------------- strategy / priority
def test_saved_calibration_is_used_first_when_it_matches_the_image(tmp_path):
    store, loc = make(tmp_path)
    store.save(CAM, TRUE, "manual_three_point")
    out = loc.localize(plate_scene(), CAM, "automatic_with_fallback")
    assert out["status"] == "localized" and out["source"] == "saved_calibration"
    assert out["plate"]["center_x"] == 320 and out["plate"]["radius_px"] == 201.6 and out["plate"]["plate_diameter_mm"] == 270.0
    assert out["detection"] is None                                  # OpenCV was not even needed
    assert out["steps"][0]["outcome"] == "used" and out["steps"][0]["verification"]["verified"]


def test_calibration_scales_to_a_compatible_resolution(tmp_path):
    store, loc = make(tmp_path)
    store.save(CAM, TRUE, "manual_numeric")
    out = loc.localize(plate_scene(960, 1280), CAM, "fixed_calibration")      # 640x480 -> 1280x960
    assert out["source"] == "saved_calibration" and out["steps"][0]["scaled_from"] == [640, 480]
    p = out["plate"]
    assert (p["image_width"], p["image_height"]) == (1280, 960)
    assert p["center_x"] == 640 and p["center_y"] == 480 and abs(p["radius_px"] - 403.2) < 1e-6 and p["plate_diameter_mm"] == 270.0


def test_incompatible_aspect_ratio_is_rejected_not_stretched(tmp_path):
    store, loc = make(tmp_path)
    store.save(CAM, TRUE, "manual_numeric")                                     # 4:3
    out = loc.localize(plate_scene(720, 1280), CAM, "fixed_calibration")        # 16:9
    assert out["status"] == "needs_manual" and out["plate"] is None
    s = out["steps"][0]
    assert s["outcome"] == "rejected" and s["reason"] == "incompatible_resolution_or_invalid_record" and s["calibrated_size"] == [640, 480]


def test_calibration_is_invalidated_when_the_plate_moved(tmp_path):
    store, loc = make(tmp_path)
    store.save(CAM, TRUE, "manual_numeric")
    moved = plate_scene(cx=380, cy=250)                                          # plate shifted 60 px
    fixed = loc.localize(moved, CAM, "fixed_calibration")
    assert fixed["status"] == "needs_manual" and fixed["steps"][0]["reason"] == "calibration_not_supported_by_image"
    assert fixed["steps"][0]["verification"]["verified"] is False


def test_calibration_is_invalidated_when_the_plate_size_changed(tmp_path):
    store, loc = make(tmp_path)
    store.save(CAM, TRUE, "manual_numeric")
    assert loc.localize(plate_scene(r=140), CAM, "fixed_calibration")["status"] == "needs_manual"


def test_with_fallback_a_moved_plate_is_found_automatically_not_forced_into_the_old_circle(tmp_path):
    store, loc = make(tmp_path)
    store.save(CAM, TRUE, "manual_numeric")
    out = loc.localize(plate_scene(cx=380, cy=250), CAM, "automatic_with_fallback")
    assert out["status"] == "localized" and out["source"] == "automatic_detection"
    assert abs(out["plate"]["center_x"] - 380) < 3 and abs(out["plate"]["center_y"] - 250) < 3
    assert [s["outcome"] for s in out["steps"]] == ["rejected", "used"]


def test_verification_can_be_disabled_by_configuration(tmp_path):
    store, loc = make(tmp_path, calibration_verify=False)
    store.save(CAM, TRUE, "manual_numeric")
    out = loc.localize(table(480, 640), CAM, "fixed_calibration")                # empty platform
    assert out["source"] == "saved_calibration" and "verification" not in out["steps"][0]


def test_saved_calibration_is_never_applied_without_a_camera_id_or_for_another_camera(tmp_path):
    store, loc = make(tmp_path)
    store.save(CAM, TRUE, "manual_numeric")
    for cam in (None, "index:3"):
        out = loc.localize(plate_scene(), cam, "fixed_calibration")
        assert out["status"] == "needs_manual"
        assert out["steps"][0]["reason"] in ("image_not_from_a_calibrated_camera", "no_saved_calibration")
    assert loc.localize(plate_scene(), None, "automatic_with_fallback")["source"] == "automatic_detection"   # an upload still gets detection


def test_hand_edited_invalid_calibration_record_is_rejected_safely(tmp_path):
    store, loc = make(tmp_path)
    state = {"version": 1, "calibrations": {CAM: {"camera_id": CAM, "image_width": 640, "image_height": 480, "center_x": 9999, "center_y": 240, "radius_px": 100}}}
    (tmp_path / "camera_state.json").write_text(json.dumps(state), encoding="utf-8")
    out = loc.localize(plate_scene(), CAM, "fixed_calibration")
    assert out["status"] == "needs_manual" and out["steps"][0]["outcome"] == "rejected"


# ---------------------------------------------------------------- modes + manual fallback
def test_automatic_detection_mode_ignores_saved_calibration(tmp_path):
    store, loc = make(tmp_path)
    store.save(CAM, PlateGeometry(100, 100, 50, 640, 480), "manual_numeric")           # a wrong calibration must be irrelevant here
    out = loc.localize(plate_scene(), CAM, "automatic_detection")
    assert out["source"] == "automatic_detection" and [s["source"] for s in out["steps"]] == ["automatic_detection"]


def test_fixed_calibration_mode_never_runs_detection(tmp_path):
    store, loc = make(tmp_path)
    out = loc.localize(plate_scene(), CAM, "fixed_calibration")                        # nothing saved
    assert out["status"] == "needs_manual" and out["detection"] is None and out["steps"][0]["reason"] == "no_saved_calibration"


@pytest.mark.parametrize("mode", ["automatic_detection", "automatic_with_fallback"])
def test_failed_detection_requests_manual_calibration_and_invents_nothing(tmp_path, mode):
    store, loc = make(tmp_path)
    out = loc.localize(flat_gray(), CAM, mode)
    assert out["status"] == "needs_manual" and out["plate"] is None and out["suggestion"] is None
    assert out["detection"]["status"] == "failed" and "Calibrate manually" in out["message"]


def test_uncertain_detection_is_offered_as_a_suggestion_but_not_applied(tmp_path):
    store, loc = make(tmp_path)
    img = table(480, 640)
    for x in (170, 470):
        cv2.circle(img, (x, 240), 150, (235, 235, 232), -1, cv2.LINE_AA)               # two equal plates
    out = loc.localize(img, None, "automatic_detection")
    assert out["status"] == "needs_manual" and out["plate"] is None
    assert out["suggestion"] is not None and out["detection"]["reason"] == "ambiguous_candidates"


def test_unknown_mode_is_rejected(tmp_path):
    _, loc = make(tmp_path)
    with pytest.raises(ValueError):
        loc.localize(plate_scene(), CAM, "magic")


def test_default_mode_comes_from_settings_and_can_be_overridden_persistently(tmp_path):
    store, loc = make(tmp_path)
    assert loc.mode() == "automatic_with_fallback"
    store.set_localization_mode("fixed_calibration")
    assert loc.mode() == "fixed_calibration"
    with pytest.raises(ValueError):
        replace(Settings(), plate_localization="nope")
