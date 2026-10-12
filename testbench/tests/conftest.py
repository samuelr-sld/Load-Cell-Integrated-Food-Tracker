"""Shared fixtures and fakes. Fakes stand in for the two networks only; every other component is the real one."""
from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from testbench.classification import Prediction
from testbench.config import Settings
from testbench.crops import CropProcessor
from testbench.orchestrator import InferencePipeline
from testbench.segmentation import SegmentDetection

CLASS_NAMES = ["red", "green", "blue"] + [f"filler_{i}" for i in range(98)]   # 101 entries like Food-101


# ------------------------------------------------------------------ image helpers
def jpeg_bytes(arr: np.ndarray, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def rect_mask(shape, x1, y1, x2, y2) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def circle_mask(shape, cx, cy, r) -> np.ndarray:
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def detection(index, mask, box=None, class_id=0, name="thing", conf=0.9) -> SegmentDetection:
    if box is None:
        ys, xs = np.where(mask)
        box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    return SegmentDetection(index, class_id, name, conf, box, mask)


def color_scene(h=200, w=300):
    """Gray background with a red square, a green 3x3 speck and a blue disc. Returns (rgb, detections)."""
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    red = rect_mask((h, w), 20, 20, 90, 90)
    green = rect_mask((h, w), 150, 40, 153, 43)        # 9 px: too small to classify
    blue = circle_mask((h, w), 220, 120, 40)
    img[red] = (255, 0, 0)
    img[green] = (0, 255, 0)
    img[blue] = (0, 0, 255)
    dets = [detection(0, red, name="red-thing"), detection(1, green, name="speck"), detection(2, blue, name="blue-thing")]
    return img, dets


# ------------------------------------------------------------------ fakes
class FakeSegmenter:
    """Returns pre-built detections regardless of the image (or raises)."""

    def __init__(self, detections=None, error: Exception | None = None):
        self.detections, self.error, self.calls = detections or [], error, 0

    def segment(self, image_rgb):
        from testbench.segmentation import SegmentationOutput
        self.calls += 1
        if self.error:
            raise self.error
        return SegmentationOutput(list(self.detections), image_rgb.shape[:2], {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0})


class FakeClassifier:
    """Label depends on the crop CONTENT (dominant colour channel of the tensor), so mis-associated crops are detectable:
    red object -> 'red', green -> 'green', blue -> 'blue'."""

    def __init__(self, class_names=CLASS_NAMES):
        self.class_names, self.calls, self.batch_sizes = class_names, 0, []

    def classify(self, tensors):
        self.calls += 1
        self.batch_sizes.append(len(tensors))
        out = []
        for t in tensors:
            top = int(t.view(3, -1).max(dim=1).values.argmax())
            order = [top] + [i for i in range(5) if i != top][:4]
            out.append([Prediction(r + 1, ci, self.class_names[ci], 0.6 / (r + 1)) for r, ci in enumerate(order)])
        return out


YOLO_NAMES = {-1: "background", 0: "red-thing", 1: "speck", 2: "blue-thing", 3: "egg tart"}   # like the checkpoint: stray -1, names with spaces


def make_pipeline(detections=None, error=None, settings: Settings | None = None):
    settings = settings or Settings()
    seg, cls = FakeSegmenter(detections, error), FakeClassifier()
    pipe = InferencePipeline(seg, CropProcessor(settings.min_mask_pixels, settings.min_crop_side), cls, settings,
                             torch.device("cpu"), {"device": {"resolved": "cpu"}}, YOLO_NAMES)
    return pipe, seg, cls


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def small_limits(settings):
    return replace(settings, max_upload_bytes=200_000, max_files_per_request=3, max_results=2)
