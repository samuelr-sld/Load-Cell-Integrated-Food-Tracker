"""AutoPlateDetectionService on synthetic images with known ground truth. No model, no camera."""
import cv2
import numpy as np
import pytest

from testbench.plate_analysis import PlateGeometry
from testbench.plate_detection import AutoPlateDetectionService
from .synthetic_plate import flat_gray, plate_scene, table

svc = AutoPlateDetectionService()
TOL_PX = 3.0


def near(d, cx, cy, r, tol=TOL_PX):
    p = d["plate"]
    assert p is not None, d["message"]
    assert abs(p["center_x"] - cx) <= tol and abs(p["center_y"] - cy) <= tol and abs(p["radius_px"] - r) <= tol, (p, (cx, cy, r))


def white(img, draw):
    draw(img, (235, 235, 232))
    return img


@pytest.mark.parametrize("h,w", [(480, 640), (720, 1280), (1080, 1920)])
def test_clean_plate_is_detected_accurately_at_several_resolutions(h, w):
    d = svc.detect(plate_scene(h, w))
    assert d["status"] == "detected" and d["reason"] is None
    near(d, w / 2, h / 2, 0.42 * min(h, w), tol=0.01 * min(h, w) + 1)
    ev = d["evidence"]
    assert ev["edge_support"] >= 0.7 and ev["arc_span_deg"] >= 300 and 0.5 <= ev["evidence_score"] <= 1.0
    assert "NOT a calibrated probability" in d["evidence_note"]


def test_geometry_is_a_valid_platgeometry_for_the_detected_image():
    d = svc.detect(plate_scene())
    p = d["plate"]
    geo = PlateGeometry(p["center_x"], p["center_y"], p["radius_px"], p["image_width"], p["image_height"])
    assert (geo.image_width, geo.image_height) == (640, 480) and geo.fully_in_frame


def test_off_centre_plate_and_food_on_it():
    near(svc.detect(plate_scene(cx=250, cy=200, r=170)), 250, 200, 170)


def test_plate_with_food_overlapping_the_rim_still_detected_with_reduced_support():
    img = plate_scene(food=False)
    for ang in (10, 70, 140, 200, 260, 320):
        a = np.deg2rad(ang)
        cv2.circle(img, (int(320 + 190 * np.cos(a)), int(240 + 190 * np.sin(a))), 38, (60, 140, 70), -1, cv2.LINE_AA)
    d = svc.detect(img)
    assert d["status"] in ("detected", "uncertain")           # never a wrong circle; either confident or explicitly unsure
    near(d, 320, 240, 201.6)
    assert d["evidence"]["edge_support"] < 0.9


def test_partially_visible_plate_is_flagged_and_never_silently_complete():
    d = svc.detect(plate_scene(cx=520, cy=240, r=200))       # right part of the plate is outside the frame
    near(d, 520, 240, 200)
    assert d["evidence"]["partially_visible"] is True and d["evidence"]["visible_fraction"] < 0.98
    assert any("partial" in w.lower() or "inside the image" in w for w in d["warnings"])
    assert d["plate"]["fully_in_frame"] is False


def test_plate_mostly_out_of_frame_with_centre_outside_is_an_explicit_failure():
    d = svc.detect(plate_scene(cx=700, cy=240, r=200))       # centre 60 px beyond the right edge
    assert d["status"] in ("failed", "uncertain")
    if d["status"] == "uncertain":
        assert d["plate"] is not None and "insufficient_arc" in d["evidence"]["reasons"] + [d["reason"]]
    else:
        assert d["plate"] is None


def test_low_contrast_plate_gives_a_lower_score_but_not_a_wrong_circle():
    easy = svc.detect(plate_scene(well=False, food=False))
    hard = svc.detect(plate_scene(rim=(78, 108, 148), well=False, food=False))      # rim only ~8 grey levels brighter than the table
    assert easy["status"] == "detected"
    if hard["plate"]:
        near(hard, 320, 240, 201.6)
        assert hard["evidence"]["evidence_score"] < easy["evidence"]["evidence_score"]


def test_jpeg_compression_and_noise_do_not_break_detection():
    img = plate_scene(noise=15)
    ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 30])
    d = svc.detect(cv2.imdecode(buf, 1)[..., ::-1].copy())
    assert d["status"] == "detected"
    near(d, 320, 240, 201.6)


# ---------------------------------------------------------------- rejection of non-plates
def test_rectangle_board_is_not_a_plate():
    img = table(480, 640)
    cv2.rectangle(img, (120, 60), (520, 420), (235, 235, 232), -1)
    d = svc.detect(img)
    assert d["status"] != "detected" and d["plate"] is None


def test_triangle_is_not_a_plate():
    img = table(480, 640)
    cv2.fillPoly(img, [np.array([[320, 30], [560, 420], [80, 420]])], (235, 235, 232))
    assert svc.detect(img)["status"] != "detected"


def test_strongly_eccentric_ellipse_is_not_accepted_as_a_circle():
    img = table(480, 640)
    cv2.ellipse(img, (320, 240), (230, 120), 0, 0, 360, (235, 235, 232), -1, cv2.LINE_AA)
    d = svc.detect(img)
    assert d["status"] != "detected"
    assert d["plate"] is None or d["status"] == "uncertain"


def test_texture_without_a_plate_is_never_reported_as_detected():
    assert svc.detect(table(480, 640, tex=25))["status"] != "detected"


@pytest.mark.parametrize("img,reason", [(flat_gray(), "low_contrast_image"),
                                        (np.clip(flat_gray() + np.random.default_rng(0).normal(0, 2, (480, 640, 3)), 0, 255).astype(np.uint8), "low_contrast_image")])
def test_blank_or_flat_image_fails_explicitly_with_no_geometry(img, reason):
    d = svc.detect(img)
    assert d["status"] == "failed" and d["reason"] == reason and d["plate"] is None and d["evidence"] is None
    assert "contrast" in d["message"]


def test_tiny_or_non_colour_input_fails_explicitly():
    assert svc.detect(np.zeros((32, 32, 3), np.uint8))["status"] == "failed"
    assert svc.detect(np.zeros((200, 200), np.uint8))["status"] == "failed"


def test_blurred_image_without_usable_edges_fails_rather_than_guessing():
    d = svc.detect(cv2.GaussianBlur(plate_scene(), (0, 0), 9))
    assert d["status"] != "detected" or d["plate"] is not None   # if it is confident, it must be a real circle
    if d["status"] == "detected":
        near(d, 320, 240, 201.6, tol=8)
    else:
        assert d["status"] in ("failed", "uncertain")


# ---------------------------------------------------------------- multiple candidates
def test_two_equal_plates_are_reported_ambiguous_not_arbitrarily_chosen():
    img = table(480, 640)
    for x in (170, 470):
        cv2.circle(img, (x, 240), 150, (235, 235, 232), -1, cv2.LINE_AA)
    d = svc.detect(img)
    assert d["status"] == "uncertain" and d["reason"] == "ambiguous_candidates"
    assert len(d["candidates"]) >= 2 and d["plate"] is not None   # shown to the user, never applied


def test_plate_on_a_faint_placemat_picks_the_much_stronger_plate_rim():
    img = table(480, 640)
    cv2.circle(img, (320, 240), 235, (60, 50, 120), -1, cv2.LINE_AA)       # low-contrast placemat edge
    cv2.circle(img, (320, 240), 180, (235, 235, 232), -1, cv2.LINE_AA)
    d = svc.detect(img)
    assert d["status"] == "detected"
    near(d, 320, 240, 180)
    assert len(d["candidates"]) >= 2 and d["candidates"][0]["evidence_score"] > d["candidates"][1]["evidence_score"]


def test_plate_on_an_equally_strong_placemat_is_uncertain_not_silently_the_wrong_circle():
    img = table(480, 640, base=(205, 200, 190), tex=3)
    cv2.circle(img, (320, 240), 235, (25, 25, 60), -1, cv2.LINE_AA)        # high-contrast placemat: as strong an edge as the plate
    cv2.circle(img, (320, 240), 180, (235, 235, 232), -1, cv2.LINE_AA)
    d = svc.detect(img)
    assert d["status"] == "uncertain" and "enclosing_candidates" in d["evidence"]["reasons"]
    assert any("encloses" in w for w in d["warnings"])
    assert d["plate"]["radius_px"] > 200                                   # the outer circle is only suggested, never applied


def test_small_round_food_blob_outside_plausible_plate_size_is_ignored():
    img = table(480, 640)
    cv2.circle(img, (320, 240), 60, (235, 235, 232), -1, cv2.LINE_AA)      # radius 12.5 % of the short side: a cookie, not a plate
    assert svc.detect(img)["status"] == "failed"


def test_best_scoring_plate_wins_when_a_weaker_circle_is_also_present():
    img = plate_scene(cx=200, cy=240, r=170)
    cv2.circle(img, (540, 240), 100, (130, 112, 150), -1, cv2.LINE_AA)    # faint distractor disc
    d = svc.detect(img)
    near(d, 200, 240, 170)


# ---------------------------------------------------------------- verification of a saved circle
def test_verify_accepts_the_true_circle_and_rejects_a_moved_or_resized_one():
    img = plate_scene()
    true = PlateGeometry(320, 240, 201.6, 640, 480)
    assert svc.verify(img, true, 0.35)["verified"]
    moved = PlateGeometry(320 + 60, 240, 201.6, 640, 480)
    shrunk = PlateGeometry(320, 240, 150, 640, 480)
    assert not svc.verify(img, moved, 0.35)["verified"]
    assert not svc.verify(img, shrunk, 0.35)["verified"]
    assert svc.verify(img, moved, 0.35)["edge_support"] < svc.verify(img, true, 0.35)["edge_support"]


def test_verify_scales_a_calibration_to_the_image_resolution():
    img = plate_scene(960, 1280)
    small = PlateGeometry(320, 240, 201.6, 640, 480).scaled_to(640, 480)
    assert svc.verify(img, small, 0.35)["verified"]          # 640x480 -> 1280x960 is the same aspect ratio


def test_verify_fails_on_an_empty_platform():
    assert not svc.verify(table(480, 640), PlateGeometry(320, 240, 201.6, 640, 480), 0.35)["verified"]
