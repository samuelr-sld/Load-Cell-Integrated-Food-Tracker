"""Mask-based food cropping and deterministic classifier preprocessing."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .preprocessing import build_letterbox_resize, build_tensor_stage
from .segmentation import SegmentDetection


@dataclass
class CropResult:
    index: int                              # SegmentDetection.index this crop came from
    status: str                             # "ok" | "skipped"
    skip_reason: str | None
    box_px: tuple[int, int, int, int]       # integer, clamped (x1, y1, x2, y2); x2/y2 exclusive
    mask_area_px: int                       # mask pixels inside the box
    mask_fill_ratio: float                  # mask_area / box_area
    masked_crop: Image.Image | None = None  # primary classifier input: food pixels only, rest = black (0)
    bbox_crop: Image.Image | None = None    # same box, background retained (debug comparison only)
    mask_crop: Image.Image | None = None    # binary mask ('L': 255 = food) over the same box, for lossless export

    @property
    def size_wh(self) -> tuple[int, int]:
        return (self.box_px[2] - self.box_px[0], self.box_px[3] - self.box_px[1])


def pixel_box(box_xyxy, width: int, height: int) -> tuple[int, int, int, int]:
    """Outward-rounded integer box clamped to the image. x2/y2 are exclusive (numpy slice convention)."""
    x1, y1, x2, y2 = box_xyxy
    return (max(0, int(math.floor(x1))), max(0, int(math.floor(y1))),
            min(width, int(math.ceil(x2))), min(height, int(math.ceil(y2))))


class CropProcessor:
    def __init__(self, min_mask_pixels: int, min_crop_side: int):
        self.min_mask_pixels, self.min_crop_side = min_mask_pixels, min_crop_side
        self._letterbox_resize = build_letterbox_resize()
        self._tensor_stage = build_tensor_stage()

    def crop(self, image_rgb: np.ndarray, det: SegmentDetection) -> CropResult:
        """Crop from the decoded original pixels (never from the overlay or a re-encoded copy)."""
        h, w = image_rgb.shape[:2]
        if det.mask.shape != (h, w):
            # Mask/image mismatch must never be resized silently: that is exactly how misaligned crops happen.
            raise ValueError(f"mask shape {det.mask.shape} does not match image shape {(h, w)}")
        x1, y1, x2, y2 = box = pixel_box(det.box_xyxy, w, h)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return CropResult(det.index, "skipped", "invalid_box", box, 0, 0.0)

        mask = det.mask[y1:y2, x1:x2]
        area = int(mask.sum())
        fill = area / float(bw * bh)
        if area == 0:
            return CropResult(det.index, "skipped", "empty_mask", box, 0, 0.0)
        if area < self.min_mask_pixels:
            return CropResult(det.index, "skipped", f"mask_too_small (<{self.min_mask_pixels} px)", box, area, fill)
        if min(bw, bh) < self.min_crop_side:
            return CropResult(det.index, "skipped", f"box_too_small (<{self.min_crop_side} px side)", box, area, fill)

        region = image_rgb[y1:y2, x1:x2]
        masked = np.where(mask[:, :, None], region, np.uint8(0)).astype(np.uint8)  # black outside the mask
        return CropResult(det.index, "ok", None, box, area, fill,
                          masked_crop=Image.fromarray(masked, "RGB"), bbox_crop=Image.fromarray(region.copy(), "RGB"),
                          mask_crop=Image.fromarray(mask.astype(np.uint8) * 255, "L"))

    def model_input_preview(self, crop: Image.Image) -> Image.Image:
        """The exact 224x224 PIL image (after letterbox + resize) that is turned into the classifier tensor."""
        return self._letterbox_resize(crop)

    def to_tensor(self, model_input: Image.Image) -> torch.Tensor:
        """ToTensor + ImageNet normalization on the 224x224 preview -> float32 (3, 224, 224)."""
        return self._tensor_stage(model_input)
