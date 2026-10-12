"""Area-proportional allocation of one measured total weight: conservation, unresolved mass, rejections, coefficients, counts."""
import pytest

from testbench.plate_analysis import FoodRegion, PlateAnalysisConfig, PlateAnalysisService, PlateGeometry
from testbench.weight_allocation import (DISCLAIMER, RegionIdentity, TotalWeightMeasurement, WeightAllocationConfig,
                                         WeightAllocationService, round_preserving_total)
from .conftest import circle_mask, rect_mask

W, H = 400, 300
SHAPE = (H, W)
PLATE = PlateGeometry(200, 150, 120, W, H)


def analysis(*masks, plate=PLATE, **cfg):
    return PlateAnalysisService(PlateAnalysisConfig(**cfg)).analyze(plate, [FoodRegion(i + 1, m, [i + 1]) for i, m in enumerate(masks)])


def ok(rid, label="rice", **kw):
    return RegionIdentity(rid, "confirmed", label, "foodseg103", **kw)


def allocate(a, grams, idents, coefficients=None, **cfg):
    return WeightAllocationService(WeightAllocationConfig(**cfg)).allocate(a, TotalWeightMeasurement(grams), idents, coefficients)


def codes(res, severity=None):
    return {i.code for i in res.issues if severity is None or i.severity == severity}


# disjoint 100x100 and 50x100 rectangles on the plate -> area ratio 2:1
A = rect_mask(SHAPE, 110, 100, 210, 200)
B = rect_mask(SHAPE, 220, 100, 270, 200)


# ------------------------------------------------------------------ conservation
def test_all_confirmed_allocates_proportionally_and_conserves_total():
    res = allocate(analysis(A, B), 300.0, [ok(1, "rice"), ok(2, "chicken")])
    assert res.status == "complete" and res.mass_conserved
    assert [x.estimated_weight_g for x in res.allocations] == [200.0, 100.0]
    assert res.allocated_weight_g == 300.0 and res.unresolved_weight_g == 0.0
    assert [x.weight_share for x in res.allocations] == pytest.approx([2 / 3, 1 / 3])


@pytest.mark.parametrize("grams", [1.0, 33.3, 100.0, 257.9, 999.9, 4999.9])
def test_rounded_weights_always_sum_to_the_measurement(grams):
    masks = [circle_mask(SHAPE, 160, 120, 25), circle_mask(SHAPE, 240, 130, 33), rect_mask(SHAPE, 150, 180, 260, 210)]
    res = allocate(analysis(*masks), grams, [ok(1, "a"), ok(2, "b"), ok(3, "c")])
    assert res.status == "complete"
    assert sum(x.estimated_weight_g for x in res.allocations) == pytest.approx(round(grams, 1), abs=1e-9)
    assert res.allocated_weight_g == round(grams, 1)


def test_overlapping_masks_are_allocated_on_unique_area():
    m1, m2 = rect_mask(SHAPE, 120, 100, 220, 200), rect_mask(SHAPE, 170, 100, 270, 200)
    res = allocate(analysis(m1, m2), 150.0, [ok(1, "a"), ok(2, "b")])
    assert [x.estimated_weight_g for x in res.allocations] == [75.0, 75.0]
    assert [x.attributed_area_px for x in res.allocations] == [7_500, 7_500]


def test_off_plate_pixels_do_not_receive_weight():
    spill = rect_mask(SHAPE, 0, 100, 50, 200) | B          # left part lies off the plate and is clipped away
    res = allocate(analysis(A, spill), 300.0, [ok(1, "a"), ok(2, "b")])
    assert res.allocations[1].attributed_area_px == 5_000
    assert [x.estimated_weight_g for x in res.allocations] == [200.0, 100.0]
    assert "extends_beyond_plate" in codes(res, "warning") and "uncertain_coverage" in codes(res, "warning")


def test_round_preserving_total():
    assert round_preserving_total([33.333, 33.333, 33.334]) == [33.3, 33.3, 33.4]
    assert sum(round_preserving_total([10 / 3] * 3)) == pytest.approx(10.0)
    assert round_preserving_total([]) == []


# ------------------------------------------------------------------ unresolved identities
def test_unconfirmed_region_mass_stays_unresolved_and_is_not_redistributed():
    res = allocate(analysis(A, B), 300.0, [ok(1), RegionIdentity(2, "needs_confirmation")])
    assert res.status == "partial" and not res.mass_conserved
    a1, a2 = res.allocations
    assert (a1.status, a1.estimated_weight_g) == ("allocated", 200.0)     # NOT 300: region 2's mass is not handed over
    assert (a2.status, a2.estimated_weight_g, a2.unresolved_weight_g) == ("unresolved", None, 100.0)
    assert res.allocated_weight_g + res.unresolved_weight_g == 300.0


def test_missing_identity_is_treated_as_unconfirmed():
    res = allocate(analysis(A, B), 300.0, [ok(1)])
    assert res.status == "partial" and res.unresolved_weight_g == 100.0
    assert "missing_identity" in codes(res, "warning")


def test_nothing_confirmed_allocates_nothing():
    res = allocate(analysis(A, B), 300.0, [])
    assert res.status == "partial" and res.allocated_weight_g == 0.0 and res.unresolved_weight_g == 300.0


def test_confirmed_without_label_is_incomplete():
    res = allocate(analysis(A), 100.0, [RegionIdentity(1, "confirmed", "  ")])
    assert res.status == "rejected" and "incomplete_identity" in codes(res, "blocking")


@pytest.mark.parametrize("status", ["maybe", "", None])
def test_unknown_identity_status_is_rejected(status):
    res = allocate(analysis(A), 100.0, [RegionIdentity(1, status, "rice")])
    assert res.status == "rejected" and "invalid_identity_status" in codes(res)


def test_duplicate_identity_for_one_region_is_rejected():
    res = allocate(analysis(A), 100.0, [ok(1), ok(1, "beans")])
    assert res.status == "rejected" and "duplicate_identity" in codes(res)


def test_not_food_region_left_in_analysis_is_rejected():
    res = allocate(analysis(A, B), 300.0, [ok(1), RegionIdentity(2, "not_food")])
    assert res.status == "rejected" and "not_food_region_in_analysis" in codes(res)
    assert all(x.estimated_weight_g is None and x.status == "not_allocated" for x in res.allocations)


# ------------------------------------------------------------------ missing masks
def test_confirmed_food_without_mask_in_analysis_is_rejected():
    res = allocate(analysis(A), 100.0, [ok(1), ok(9, "beans")])
    assert res.status == "rejected" and "missing_mask" in codes(res, "blocking")
    assert res.allocated_weight_g == 0.0


def test_confirmed_food_with_empty_mask_is_rejected():
    off = rect_mask(SHAPE, 0, 0, 20, 20)                      # entirely off the plate
    res = allocate(analysis(A, off), 100.0, [ok(1), ok(2, "beans")])
    assert res.status == "rejected" and "missing_mask" in codes(res, "blocking")


def test_unconfirmed_empty_region_is_only_a_warning():
    off = rect_mask(SHAPE, 0, 0, 20, 20)
    res = allocate(analysis(A, off), 100.0, [ok(1), RegionIdentity(2)])
    assert res.status == "partial" and "empty_region" in codes(res, "warning")
    assert res.allocations[0].estimated_weight_g == 100.0 and res.allocations[1].unresolved_weight_g == 0.0


def test_no_regions_or_no_area_is_rejected():
    assert "no_food_regions" in codes(allocate(analysis(), 100.0, []))
    res = allocate(analysis(rect_mask(SHAPE, 0, 0, 20, 20)), 100.0, [RegionIdentity(1)])
    assert res.status == "rejected" and "no_food_area" in codes(res)


# ------------------------------------------------------------------ weights
@pytest.mark.parametrize("grams", [0, -5.0, float("nan"), float("inf"), 5000.1, None, "150", True])
def test_invalid_total_weight_is_rejected(grams):
    res = allocate(analysis(A), grams, [ok(1)])
    assert res.status == "rejected" and "invalid_weight" in codes(res, "blocking")
    assert res.allocations[0].estimated_weight_g is None


def test_unknown_weight_source_is_rejected():
    res = WeightAllocationService().allocate(analysis(A), TotalWeightMeasurement(100.0, source="guess"), [ok(1)])
    assert res.status == "rejected" and "invalid_weight_source" in codes(res)


def test_load_cell_source_is_accepted():
    res = WeightAllocationService().allocate(analysis(A), TotalWeightMeasurement(80.0, source="load_cell", tare_g=410.0), [ok(1)])
    assert res.status == "complete" and res.allocations[0].estimated_weight_g == 80.0


# ------------------------------------------------------------------ calibration coefficients
def test_coefficients_default_to_one():
    res = allocate(analysis(A, B), 300.0, [ok(1, "rice"), ok(2, "chicken")], {})
    assert [x.coefficient for x in res.allocations] == [1.0, 1.0]


def test_food_specific_coefficient_reweights_and_still_conserves():
    # rice area 2 x coef 1 vs chicken area 1 x coef 2 -> equal split
    res = allocate(analysis(A, B), 300.0, [ok(1, "rice"), ok(2, "chicken")], {"Chicken": 2.0})
    assert [x.estimated_weight_g for x in res.allocations] == [150.0, 150.0]
    assert res.allocated_weight_g == 300.0


def test_coefficients_do_not_apply_to_unconfirmed_regions():
    res = allocate(analysis(A, B), 300.0, [ok(1, "rice"), RegionIdentity(2, label="chicken")], {"chicken": 5.0})
    assert res.allocations[1].coefficient == 1.0 and res.allocations[1].unresolved_weight_g == 100.0


@pytest.mark.parametrize("k", [0.0, -1.0, float("nan"), float("inf"), "2", True])
def test_invalid_coefficient_is_rejected(k):
    res = allocate(analysis(A), 100.0, [ok(1, "rice")], {"rice": k})
    assert res.status == "rejected" and "invalid_coefficient" in codes(res)


# ------------------------------------------------------------------ coverage uncertainty
def test_uncertain_coverage_is_flagged_by_default_and_blocks_when_strict():
    edge_plate = PlateGeometry(60, 150, 100, W, H)            # partly out of frame
    m = circle_mask(SHAPE, 70, 150, 30)
    lenient = allocate(analysis(m, plate=edge_plate), 100.0, [ok(1)])
    assert lenient.status == "complete" and "uncertain_coverage" in codes(lenient, "warning")
    assert lenient.coverage_flags == ["plate_partially_out_of_frame"]
    strict = allocate(analysis(m, plate=edge_plate), 100.0, [ok(1)], reject_uncertain_coverage=True)
    assert strict.status == "rejected" and "uncertain_coverage" in codes(strict, "blocking")


# ------------------------------------------------------------------ entity counts
def test_detected_and_user_entity_counts_stay_separate():
    a = PlateAnalysisService().analyze(PLATE, [FoodRegion(1, A, [1, 4]), FoodRegion(2, B, [2])])
    res = WeightAllocationService().allocate(a, TotalWeightMeasurement(300.0), [ok(1, "dumpling", user_entity_count=4), ok(2, "sauce")])
    d, s = res.allocations
    assert (d.entity_count.detected_segments, d.entity_count.user_count) == (2, 4)
    assert d.weight_per_entity_g == 50.0                        # 200 g / user's 4 items, never / 2 detected segments
    assert (s.entity_count.detected_segments, s.entity_count.user_count, s.weight_per_entity_g) == (1, None, None)
    assert res.detected_region_count == 2


@pytest.mark.parametrize("n", [0, -1, 2.5, True])
def test_invalid_user_entity_count_is_rejected(n):
    res = allocate(analysis(A), 100.0, [ok(1, user_entity_count=n)])
    assert res.status == "rejected" and "invalid_entity_count" in codes(res)


# ------------------------------------------------------------------ output contract
def test_result_dict_carries_method_and_disclaimer_without_accuracy_claim():
    d = allocate(analysis(A, B), 300.0, [ok(1), ok(2, "beans")]).to_dict()
    assert d["method"] == "area_proportional" and d["disclaimer"] == DISCLAIMER
    assert "no validated accuracy margin" in d["disclaimer"]
    assert d["mass_conserved"] is True and d["allocations"][0]["entity_count"]["user_count"] is None


@pytest.mark.parametrize("grams", [200.05, 99.99, 333.333])
def test_total_finer_than_rounding_is_rejected_not_silently_unconserved(grams):
    """Audit finding: a 200.05 g total used to allocate 200.0 g while reporting mass_conserved."""
    res = allocate(analysis(A, B), grams, [ok(1, "rice"), ok(2, "chicken")])
    assert res.status == "rejected" and "weight_precision" in codes(res, "blocking") and not res.mass_conserved
