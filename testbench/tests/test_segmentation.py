"""SegmentationService against a fake Ultralytics model: channel order, mask alignment, empty results."""
from types import SimpleNamespace

import numpy as np
import torch

from testbench.config import Settings
from testbench.segmentation import SegmentationService, render_overlay
from .conftest import color_scene, detection, rect_mask


class FakeYolo:
    def __init__(self, result):
        self.result, self.kwargs, self.seen = result, None, None

    def predict(self, source, **kwargs):
        self.seen, self.kwargs = source, kwargs
        return [self.result]


class FakeBoxes:
    def __init__(self, boxes, confs, cls):
        self.xyxy = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        self.conf = torch.tensor(confs, dtype=torch.float32)
        self.cls = torch.tensor(cls, dtype=torch.float32)

    def __len__(self):
        return len(self.xyxy)


def _result(mask_data, polys, boxes, confs, cls, names):
    return SimpleNamespace(masks=SimpleNamespace(data=mask_data, xy=polys), boxes=FakeBoxes(boxes, confs, cls),
                           names=names, speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0})


def _service(result):
    yolo = FakeYolo(result)
    return SegmentationService(yolo, {0: "a"}, torch.device("cpu"), Settings()), yolo


def test_no_detections_returns_empty_output():
    res = SimpleNamespace(masks=None, boxes=FakeBoxes([], [], []), names={}, speed={})
    svc, _ = _service(res)
    out = svc.segment(np.zeros((40, 50, 3), np.uint8))
    assert out.detections == [] and out.image_hw == (40, 50)


def test_yolo_receives_bgr_and_retina_masks_and_configured_thresholds():
    img = np.zeros((40, 50, 3), np.uint8)
    img[..., 0] = 200                       # pure red in RGB
    svc, yolo = _service(_result(torch.zeros(0, 40, 50), [], [], [], [], {}))
    svc.segment(img)
    assert yolo.seen[..., 2].min() == 200 and yolo.seen[..., 0].max() == 0      # Ultralytics wants BGR: red lands in the last channel
    assert yolo.kwargs["retina_masks"] is True and yolo.kwargs["conf"] == 0.25 and yolo.kwargs["verbose"] is False


def test_masks_already_at_image_resolution_are_used_directly():
    h, w = 40, 50
    m = torch.zeros(1, h, w)
    m[0, 10:20, 5:25] = 1.0
    svc, _ = _service(_result(m, [np.zeros((0, 2))], [[5, 10, 25, 20]], [0.9], [0], {0: "rice"}))
    d = svc.segment(np.zeros((h, w, 3), np.uint8)).detections[0]
    assert d.mask.dtype == bool and d.mask.shape == (h, w) and d.mask.sum() == 10 * 20
    assert d.class_name == "rice" and abs(d.confidence - 0.9) < 1e-6 and d.index == 0


def test_misaligned_mask_resolution_falls_back_to_original_coordinate_polygons():
    """Letterboxed 32x48 masks must NOT be stretched onto the 40x50 image; polygons (original coords) are rasterized."""
    h, w = 40, 50
    letterboxed = torch.ones(1, 32, 48)       # garbage that would corrupt alignment if it were resized
    poly = np.array([[10, 10], [30, 10], [30, 25], [10, 25]], dtype=np.float32)
    svc, _ = _service(_result(letterboxed, [poly], [[10, 10, 30, 25]], [0.8], [0], {0: "x"}))
    mask = svc.segment(np.zeros((h, w, 3), np.uint8)).detections[0].mask
    ys, xs = np.where(mask)
    assert mask.shape == (h, w)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (10, 30, 10, 25)


def test_unknown_class_id_gets_a_placeholder_name():
    m = torch.zeros(1, 20, 20)
    m[0, 2:10, 2:10] = 1
    svc, _ = _service(_result(m, [np.zeros((0, 2))], [[2, 2, 10, 10]], [0.7], [99], {0: "a"}))
    assert svc.segment(np.zeros((20, 20, 3), np.uint8)).detections[0].class_name == "class_99"


def test_overlay_marks_masks_without_mutating_the_input():
    img, dets = color_scene()
    before = img.copy()
    out = render_overlay(img, dets, skipped_indices={1})
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(img, before)
    assert not np.array_equal(out, img)
    far = (150, 250)                                  # a background pixel away from every instance and label
    assert tuple(out[far]) == tuple(img[far])
