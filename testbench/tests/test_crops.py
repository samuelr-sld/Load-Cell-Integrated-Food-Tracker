import numpy as np
import pytest

from testbench.crops import CropProcessor, pixel_box
from .conftest import circle_mask, color_scene, detection, rect_mask


@pytest.fixture
def cropper():
    return CropProcessor(min_mask_pixels=100, min_crop_side=10)


def test_masked_crop_keeps_original_pixels_inside_mask_and_black_outside(cropper):
    img, dets = color_scene()
    blue = dets[2]
    c = cropper.crop(img, blue)
    x1, y1, x2, y2 = c.box_px
    region, mask = img[y1:y2, x1:x2], blue.mask[y1:y2, x1:x2]
    out = np.asarray(c.masked_crop)
    assert c.status == "ok"
    assert (out[mask] == region[mask]).all()                 # exact original pixels where the mask is set
    assert (out[~mask] == 0).all()                           # black everywhere else
    assert (~mask).any()                                     # a disc inside a square box really has background to remove


def test_bbox_crop_is_unmasked_slice_with_same_geometry(cropper):
    img, dets = color_scene()
    c = cropper.crop(img, dets[2])
    x1, y1, x2, y2 = c.box_px
    assert np.array_equal(np.asarray(c.bbox_crop), img[y1:y2, x1:x2])
    assert c.masked_crop.size == c.bbox_crop.size == (x2 - x1, y2 - y1)    # aspect ratio untouched (no resize here)


def test_crop_aspect_ratio_of_non_square_instance(cropper):
    img = np.full((100, 200, 3), 50, np.uint8)
    d = detection(0, rect_mask((100, 200), 10, 20, 150, 60))      # 140 x 40
    c = cropper.crop(img, d)
    assert c.masked_crop.size == (140, 40)


def test_irregular_mask_fill_ratio(cropper):
    img = np.full((100, 100, 3), 77, np.uint8)
    d = detection(0, circle_mask((100, 100), 50, 50, 30))
    c = cropper.crop(img, d)
    assert 0.7 < c.mask_fill_ratio < 0.85            # pi/4 ~ 0.785


def test_overlapping_boxes_do_not_leak_pixels_between_instances(cropper):
    img = np.full((120, 160, 3), 128, np.uint8)
    a = circle_mask((120, 160), 60, 60, 35)
    b = circle_mask((120, 160), 95, 60, 35) & ~a          # second object overlaps A's box, disjoint pixels
    img[a] = (255, 0, 0)
    img[b] = (0, 0, 255)
    da, db = detection(0, a), detection(1, b)
    ca, cb = cropper.crop(img, da), cropper.crop(img, db)
    # boxes overlap...
    assert max(ca.box_px[0], cb.box_px[0]) < min(ca.box_px[2], cb.box_px[2])
    # ...but each masked crop contains only its own object's colour
    ra, rb = np.asarray(ca.masked_crop).reshape(-1, 3), np.asarray(cb.masked_crop).reshape(-1, 3)
    assert not ((ra == (0, 0, 255)).all(axis=1)).any()
    assert not ((rb == (255, 0, 0)).all(axis=1)).any()
    assert ((ra == (255, 0, 0)).all(axis=1)).any() and ((rb == (0, 0, 255)).all(axis=1)).any()


@pytest.mark.parametrize("mask,box,reason", [
    (np.zeros((100, 100), bool), (5.0, 5.0, 40.0, 40.0), "empty_mask"),
    (rect_mask((100, 100), 5, 5, 8, 8), (5.0, 5.0, 40.0, 40.0), "mask_too_small"),    # 9 px
    (rect_mask((100, 100), 5, 5, 85, 8), (5.0, 5.0, 85.0, 8.0), "box_too_small"),     # 240 px but only 3 px tall
])
def test_empty_small_and_thin_instances_are_skipped_with_a_reason(cropper, mask, box, reason):
    img = np.full((100, 100, 3), 10, np.uint8)
    c = cropper.crop(img, detection(0, mask, box=box))
    assert c.status == "skipped" and c.skip_reason.startswith(reason)
    assert c.masked_crop is None and c.bbox_crop is None


def test_degenerate_and_out_of_bounds_boxes(cropper):
    img = np.full((50, 60, 3), 10, np.uint8)
    assert cropper.crop(img, detection(0, rect_mask((50, 60), 5, 5, 30, 30), box=(30.0, 30.0, 30.0, 30.0))).skip_reason == "invalid_box"
    # a box hanging off the image is clamped, never indexed out of range
    c = cropper.crop(img, detection(0, rect_mask((50, 60), 30, 20, 60, 50), box=(-5.0, 10.0, 99.0, 99.0)))
    assert c.box_px == (0, 10, 60, 50) and c.status == "ok"
    assert pixel_box((1.2, 2.7, 9.1, 9.0), 100, 100) == (1, 2, 10, 9)      # outward rounding


def test_mask_image_shape_mismatch_is_an_error_not_a_silent_resize(cropper):
    img = np.zeros((100, 100, 3), np.uint8)
    d = detection(0, rect_mask((80, 100), 5, 5, 40, 40))
    with pytest.raises(ValueError, match="does not match"):
        cropper.crop(img, d)


def test_rgb_channel_order_is_preserved_in_crops(cropper):
    img = np.zeros((60, 60, 3), np.uint8)
    img[10:50, 10:50] = (200, 10, 30)       # R, G, B
    c = cropper.crop(img, detection(0, rect_mask((60, 60), 10, 10, 50, 50)))
    assert tuple(np.asarray(c.masked_crop)[20, 20]) == (200, 10, 30)
