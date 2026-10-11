"""Plate calibration geometry, clipping to the plate, unique food area and overlap attribution. Synthetic masks only."""
import math

import numpy as np
import pytest

from testbench.plate_analysis import FoodRegion, PlateAnalysisConfig, PlateAnalysisService, PlateGeometry, region_mask
from .conftest import circle_mask, rect_mask

W, H = 400, 300
SHAPE = (H, W)


def plate(cx=200, cy=150, r=120, **kw) -> PlateGeometry:
    return PlateGeometry(cx, cy, r, W, H, **kw)


def analyze(regions, p=None, **cfg):
    priority = cfg.pop("priority", None)
    return PlateAnalysisService(PlateAnalysisConfig(**cfg)).analyze(p or plate(), regions, priority)


# ------------------------------------------------------------------ geometry
def test_plate_mask_matches_disc_pixel_convention():
    p = plate()
    assert np.array_equal(p.mask(), circle_mask(SHAPE, 200, 150, 120))
    assert p.fully_in_frame
    assert int(p.mask().sum()) == pytest.approx(math.pi * 120 ** 2, rel=0.01)


@pytest.mark.parametrize("kw", [dict(radius_px=0), dict(radius_px=-5), dict(radius_px=float("nan")), dict(center_x=float("inf")),
                                dict(center_x=-1), dict(center_y=H), dict(image_width=0), dict(image_height=12.5),
                                dict(plate_diameter_mm=0), dict(center_x=True)])
def test_invalid_geometry_is_rejected(kw):
    args = dict(center_x=200, center_y=150, radius_px=100, image_width=W, image_height=H) | kw
    with pytest.raises(ValueError):
        PlateGeometry(**args)


def test_circle_from_three_rim_points():
    cx, cy, r = 210.0, 140.0, 95.0
    pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in (0.3, 2.1, 4.4)]
    p = PlateGeometry.from_rim_points(pts, W, H)
    assert (p.center_x, p.center_y, p.radius_px) == pytest.approx((cx, cy, r), abs=1e-9)


@pytest.mark.parametrize("pts", [[(0, 0), (10, 10), (20, 20)], [(5, 5), (5, 5), (30, 40)], [(1, 1), (2, 2)]])
def test_collinear_repeated_or_too_few_rim_points_are_rejected(pts):
    with pytest.raises(ValueError):
        PlateGeometry.from_rim_points(pts, W, H)


def test_scaling_to_downscaled_image_coordinates():
    p = plate(plate_diameter_mm=260).scaled_to(200, 150)
    assert (p.center_x, p.center_y, p.radius_px, p.image_width, p.image_height) == (100, 75, 60, 200, 150)
    assert p.mm_per_px == pytest.approx(260 / 120)
    with pytest.raises(ValueError):
        plate().scaled_to(200, 300)                      # aspect ratio change


def test_plate_partially_out_of_frame_uses_in_frame_area_and_is_flagged():
    p = plate(cx=50, cy=150, r=100)
    assert not p.fully_in_frame
    a = analyze([FoodRegion(1, circle_mask(SHAPE, 60, 150, 30))], p)
    assert a.plate_area_px == int(p.mask().sum())
    d, r = 50.0, 100.0                                   # left edge is a chord 50 px from the centre
    cut = (r * r * math.acos(d / r) - d * math.sqrt(r * r - d * d)) / (math.pi * r * r)
    assert a.plate_in_frame_fraction == pytest.approx(1 - cut, abs=0.01)
    assert "plate_partially_out_of_frame" in a.coverage_flags and a.coverage_uncertain


# ------------------------------------------------------------------ clipping and percentages
def test_mask_is_clipped_to_plate_and_off_plate_area_reported():
    m = rect_mask(SHAPE, 0, 140, W, 160)                 # 20 px band across the whole image, plate spans 241 columns
    a = analyze([FoodRegion(1, m, [1])])
    r = a.region(1)
    assert r.mask_area_px == 20 * W
    assert r.on_plate_px == int((m & plate().mask()).sum())
    assert r.off_plate_px == r.mask_area_px - r.on_plate_px > 0
    assert "extends_beyond_plate" in r.flags and "food_beyond_plate_rim" in a.coverage_flags


def test_single_region_percentages():
    m = rect_mask(SHAPE, 150, 100, 250, 200)             # 100 x 100, fully on the plate
    a = analyze([FoodRegion(1, m)])
    r = a.region(1)
    assert r.attributed_area_px == a.food_area_px == 10_000
    assert r.food_area_pct == 100.0
    assert r.plate_coverage_pct == pytest.approx(100 * 10_000 / a.plate_area_px, abs=1e-3)
    assert a.plate_coverage_pct == pytest.approx(r.plate_coverage_pct, abs=1e-3)
    assert not a.coverage_uncertain and r.flags == []


def test_food_entirely_off_plate_has_no_area():
    a = analyze([FoodRegion(1, rect_mask(SHAPE, 0, 0, 30, 30))])
    assert a.region(1).on_plate_px == 0 and not a.region(1).has_area
    assert "no_area_on_plate" in a.region(1).flags and "no_food_area_on_plate" in a.coverage_flags


def test_area_in_cm2_only_with_physical_plate_diameter():
    m = rect_mask(SHAPE, 150, 100, 250, 200)
    assert analyze([FoodRegion(1, m)]).region(1).attributed_area_cm2 is None
    r = analyze([FoodRegion(1, m)], plate(plate_diameter_mm=240)).region(1)  # 1 mm/px -> 10 000 mm^2 = 100 cm^2
    assert r.attributed_area_cm2 == pytest.approx(100.0)


# ------------------------------------------------------------------ overlaps
def test_disjoint_regions_share_nothing_and_sum_to_union():
    a = analyze([FoodRegion(1, rect_mask(SHAPE, 120, 100, 180, 200)), FoodRegion(2, rect_mask(SHAPE, 220, 100, 280, 200))])
    assert a.overlap_px == 0 and a.food_area_px == a.raw_on_plate_area_sum_px == 12_000
    assert [r.food_area_pct for r in a.regions] == [50.0, 50.0]


def test_overlap_is_not_double_counted_with_split_policy():
    m1, m2 = rect_mask(SHAPE, 120, 100, 220, 200), rect_mask(SHAPE, 170, 100, 270, 200)   # 100x100 each, 50x100 shared
    a = analyze([FoodRegion(1, m1), FoodRegion(2, m2)])
    assert a.food_area_px == 15_000 and a.raw_on_plate_area_sum_px == 20_000 and a.overlap_px == 5_000
    r1, r2 = a.region(1), a.region(2)
    assert (r1.exclusive_px, r1.shared_px) == (5_000, 5_000)
    assert r1.attributed_area_px == r2.attributed_area_px == 7_500
    assert r1.attributed_area_px + r2.attributed_area_px == a.food_area_px
    assert r1.food_area_pct + r2.food_area_pct == pytest.approx(100.0)


def test_triple_overlap_split_conserves_unique_area():
    masks = [circle_mask(SHAPE, 180, 150, 40), circle_mask(SHAPE, 215, 150, 40), circle_mask(SHAPE, 198, 180, 40)]
    a = analyze([FoodRegion(i + 1, m) for i, m in enumerate(masks)])
    union = int(np.logical_or.reduce(masks).sum())
    assert a.food_area_px == union
    assert sum(r.attributed_area_px for r in a.regions) == pytest.approx(union)
    assert sum(r.food_area_pct for r in a.regions) == pytest.approx(100.0, abs=1e-3)


def test_topping_inside_dish_with_priority_policy():
    dish, topping = rect_mask(SHAPE, 120, 80, 280, 220), rect_mask(SHAPE, 180, 130, 220, 170)
    a = analyze([FoodRegion(1, dish), FoodRegion(2, topping)], overlap_policy="priority", priority=[2, 1])
    assert a.region(2).attributed_area_px == 1_600                       # topping keeps its whole visible footprint
    assert a.region(1).attributed_area_px == 160 * 140 - 1_600
    assert a.food_area_px == 160 * 140
    assert "mostly_overlapping" in a.region(2).flags and "heavy_mask_overlap" in a.coverage_flags


def test_priority_policy_requires_complete_priority_list():
    regions = [FoodRegion(1, rect_mask(SHAPE, 150, 100, 200, 150)), FoodRegion(2, rect_mask(SHAPE, 160, 100, 210, 150))]
    with pytest.raises(ValueError):
        analyze(regions, overlap_policy="priority", priority=[1])
    with pytest.raises(ValueError):
        analyze(regions, overlap_policy="priority")


# ------------------------------------------------------------------ input validation and helpers
def test_mask_shape_mismatch_and_duplicate_ids_are_rejected():
    with pytest.raises(ValueError):
        analyze([FoodRegion(1, np.zeros((10, 10), dtype=bool))])
    m = rect_mask(SHAPE, 150, 100, 200, 150)
    with pytest.raises(ValueError):
        analyze([FoodRegion(1, m), FoodRegion(1, m)])


def test_unknown_overlap_policy_is_rejected():
    with pytest.raises(ValueError):
        PlateAnalysisConfig(overlap_policy="largest")


def test_integer_masks_are_accepted_as_boolean():
    a = analyze([FoodRegion(1, rect_mask(SHAPE, 150, 100, 200, 150).astype(np.uint8) * 255)])
    assert a.food_area_px == 2_500


def test_tiny_region_is_flagged():
    a = analyze([FoodRegion(1, rect_mask(SHAPE, 200, 150, 205, 155))])
    assert "tiny_region" in a.region(1).flags


def test_no_regions_gives_empty_analysis():
    a = analyze([])
    assert a.food_area_px == 0 and a.regions == [] and a.plate_coverage_pct == 0.0


def test_region_mask_unions_duplicate_segments():
    a, b = rect_mask(SHAPE, 100, 100, 150, 150), rect_mask(SHAPE, 105, 100, 155, 150)
    assert int(region_mask([a, b]).sum()) == 55 * 50
    with pytest.raises(ValueError):
        region_mask([])
    with pytest.raises(ValueError):
        region_mask([a, np.zeros((5, 5), dtype=bool)])


def test_detected_segment_count_is_reported_not_interpreted():
    a = analyze([FoodRegion(1, rect_mask(SHAPE, 150, 100, 200, 150), [3, 7])])
    assert a.region(1).detected_segment_count == 2


def test_to_dict_is_plain_and_carries_area_note():
    d = analyze([FoodRegion(1, rect_mask(SHAPE, 150, 100, 200, 150))]).to_dict()
    assert d["regions"][0]["region_id"] == 1 and "not volume or mass" in d["note"]
    assert d["plate"]["radius_px"] == 120
