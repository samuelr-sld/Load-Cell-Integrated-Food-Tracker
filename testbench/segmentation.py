"""YOLO11s-seg inference: instance masks, boxes, confidences, and the visualization overlay."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .config import Settings

# BGR-ordered Ultralytics-style palette (cv2 drawing happens in BGR)
_PALETTE_RGB = [(255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49), (72, 249, 10),
                (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
                (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255), (82, 0, 133), (203, 56, 255),
                (255, 149, 200), (255, 55, 199)]


@dataclass
class SegmentDetection:
    index: int                 # 0-based position in YOLO's output order
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]   # original-image pixels
    mask: np.ndarray           # bool, H x W of the original image


@dataclass
class SegmentationOutput:
    detections: list[SegmentDetection]
    image_hw: tuple[int, int]
    ultralytics_speed_ms: dict = field(default_factory=dict)


def color_for(index: int) -> tuple[int, int, int]:
    return _PALETTE_RGB[index % len(_PALETTE_RGB)]


def _rasterize_polygons(polys: list[np.ndarray], h: int, w: int) -> np.ndarray:
    out = np.zeros((len(polys), h, w), dtype=bool)
    for i, poly in enumerate(polys):
        if len(poly) >= 3:
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
            out[i] = m.astype(bool)
    return out


class SegmentationService:
    def __init__(self, yolo, names: dict[int, str], device: torch.device, settings: Settings):
        self.yolo, self.names, self.device, self.settings = yolo, names, device, settings

    def segment(self, image_rgb: np.ndarray) -> SegmentationOutput:
        h, w = image_rgb.shape[:2]
        # Ultralytics treats numpy arrays as BGR (OpenCV convention) and flips to RGB internally. Handing it RGB would
        # silently swap channels, so convert explicitly here.
        bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        results = self.yolo.predict(
            bgr, device=str(self.device), imgsz=self.settings.yolo_imgsz, conf=self.settings.yolo_conf,
            iou=self.settings.yolo_iou, max_det=self.settings.max_segments,
            retina_masks=True,  # masks are produced at the original image resolution -> no letterbox un-mapping needed
            verbose=False,
        )
        r = results[0]
        speed = {k: round(float(v), 2) for k, v in (r.speed or {}).items()}
        if r.boxes is None or len(r.boxes) == 0 or r.masks is None:
            return SegmentationOutput([], (h, w), speed)

        data = r.masks.data
        if tuple(data.shape[1:]) == (h, w):
            masks = (data > 0.5).cpu().numpy()
        else:
            # Never stretch a letterboxed mask onto the image. Ultralytics' polygons are already in original-image
            # coordinates (it applies scale_coords itself), so rasterize those instead.
            masks = _rasterize_polygons(r.masks.xy, h, w)

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        names = r.names if r.names else self.names
        dets = [
            SegmentDetection(i, int(cls[i]), str(names.get(int(cls[i]), f"class_{int(cls[i])}")), float(confs[i]),
                             tuple(float(v) for v in boxes[i]), masks[i])
            for i in range(len(boxes))
        ]
        return SegmentationOutput(dets, (h, w), speed)


def render_overlay(image_rgb: np.ndarray, detections: list[SegmentDetection], skipped_indices: set[int] | None = None) -> np.ndarray:
    """RGB overlay: translucent mask fill, contour, box and '#id class conf' label. Own drawing code (cv2 Hershey fonts)
    so no font download is ever attempted. Segments excluded from classification are labelled '(skipped)'."""
    skipped_indices = skipped_indices or set()
    h, w = image_rgb.shape[:2]
    canvas = np.ascontiguousarray(image_rgb[:, :, ::-1]).copy()  # BGR for cv2
    scale = max(0.45, min(h, w) / 900.0)
    thick = max(1, int(round(scale * 2)))

    for d in detections:
        bgr = color_for(d.index)[::-1]
        canvas[d.mask] = (0.55 * canvas[d.mask] + 0.45 * np.array(bgr, dtype=np.float32)).astype(np.uint8)
    for d in detections:
        bgr = color_for(d.index)[::-1]
        contours, _ = cv2.findContours(d.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, bgr, thick)
        x1, y1, x2, y2 = (int(round(v)) for v in d.box_xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bgr, thick)
        label = f"#{d.index + 1} {d.class_name} {d.confidence:.2f}" + (" (skipped)" if d.index in skipped_indices else "")
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6, thick)
        ty = y1 - 4 if y1 - th - base - 4 >= 0 else y1 + th + base + 2
        cv2.rectangle(canvas, (x1, ty - th - base), (x1 + tw + 4, ty + base // 2), bgr, -1)
        cv2.putText(canvas, label, (x1 + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6, (255, 255, 255), thick, cv2.LINE_AA)
    return np.ascontiguousarray(canvas[:, :, ::-1])
