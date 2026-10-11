"""Runs the whole pipeline for one image: decode -> YOLO -> mask crops -> MobileNet -> assembled record."""
from __future__ import annotations

import io
import logging
import threading
import uuid
from datetime import datetime, timezone

import numpy as np
import torch
from PIL import Image

from .classification import ClassificationService
from .config import Settings
from .crops import CropProcessor
from .ingest import DecodedImage, decode_image
from .model_loader import LoadedModels
from .records import InferenceRecord, SegmentRecord
from .segmentation import SegmentationService, color_for, render_overlay
from .timing import StageTimer

log = logging.getLogger("testbench.pipeline")


def encode_jpeg(image: np.ndarray | Image.Image, quality: int = 92) -> bytes:
    img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class InferencePipeline:
    def __init__(self, segmenter: SegmentationService, cropper: CropProcessor, classifier: ClassificationService,
                 settings: Settings, device: torch.device, info: dict | None = None):
        self.segmenter, self.cropper, self.classifier = segmenter, cropper, classifier
        self.settings, self.device, self.info = settings, device, info or {}
        self.class_names = classifier.class_names
        self.ready = True
        self._lock = threading.Lock()   # sequential inference: never two images on the GPU at once

    @classmethod
    def from_models(cls, models: LoadedModels, settings: Settings) -> "InferencePipeline":
        return cls(SegmentationService(models.yolo, models.yolo_names, models.device, settings),
                   CropProcessor(settings.min_mask_pixels, settings.min_crop_side),
                   ClassificationService(models.classifier, models.class_names, models.device, settings.classifier_batch_size),
                   settings, models.device, models.info)

    def health(self) -> dict:
        return {"models_loaded": True, **self.info, "class_names": self.class_names}

    # ------------------------------------------------------------------ public
    def process_upload(self, data: bytes, filename: str) -> InferenceRecord:
        """Validate+decode (ImageValidationError propagates -> upload rejected) and run inference.
        Failures *after* a successful decode become a stored record with status='failed'."""
        decoded = decode_image(data, filename, self.settings)
        return self.process_decoded(decoded, filename)

    def process_decoded(self, decoded: DecodedImage, filename: str) -> InferenceRecord:
        record = InferenceRecord(
            result_id=uuid.uuid4().hex[:12], filename=filename, status="completed", error=None,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), device=str(self.device),
            image={"width": decoded.size[0], "height": decoded.size[1], "original_width": decoded.original_size[0],
                   "original_height": decoded.original_size[1], "downscaled": decoded.downscaled, "source_format": decoded.source_format},
            settings_snapshot={"yolo_conf": self.settings.yolo_conf, "yolo_iou": self.settings.yolo_iou,
                               "yolo_imgsz": self.settings.yolo_imgsz, "retina_masks": True,
                               "max_segments": self.settings.max_segments, "min_mask_pixels": self.settings.min_mask_pixels,
                               "min_crop_side": self.settings.min_crop_side},
        )
        if decoded.downscaled:
            record.notes.append(f"Image downscaled from {decoded.original_size[0]}x{decoded.original_size[1]} to "
                                f"{decoded.size[0]}x{decoded.size[1]} (max side {self.settings.max_image_side}); all stages use the downscaled pixels.")
        try:
            with self._lock:
                self._run(decoded, record)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._fail(record, "GPU ran out of memory. Lower TESTBENCH_MAX_IMAGE_SIDE or set TESTBENCH_DEVICE=cpu.")
        except Exception as exc:  # internals stay in the server log, not in the browser
            log.exception("inference failed for %s", filename)
            self._fail(record, f"Inference failed ({type(exc).__name__}). See the server log for details.")
        return record

    # ---------------------------------------------------------------- internals
    @staticmethod
    def _fail(record: InferenceRecord, message: str) -> None:
        record.status, record.error = "failed", message
        record.segments, record.images = [], {}

    def _run(self, decoded: DecodedImage, record: InferenceRecord) -> None:
        rgb = decoded.rgb
        timer = StageTimer(self.device)

        with timer.stage("yolo_ms"):
            seg = self.segmenter.segment(rgb)

        with timer.stage("crop_preprocess_ms"):
            crops = [self.cropper.crop(rgb, det) for det in seg.detections]
            ok = [c for c in crops if c.status == "ok"]
            previews = {c.index: self.cropper.model_input_preview(c.masked_crop) for c in ok}
            tensors = [self.cropper.to_tensor(previews[c.index]) for c in ok]

        with timer.stage("classify_ms"):
            # Only valid crops reach the classifier; if there are none it is never called.
            predictions = self.classifier.classify(tensors) if tensors else []
        # ok[i] <-> tensors[i] <-> predictions[i]: the association is positional and index-checked here.
        if len(predictions) != len(ok):
            raise RuntimeError("classifier returned a different number of predictions than crops submitted")
        pred_by_index = {c.index: p for c, p in zip(ok, predictions)}

        with timer.stage("render_encode_ms"):
            skipped = {c.index for c in crops if c.status != "ok"}
            record.images["original"] = encode_jpeg(rgb)
            record.images["overlay"] = encode_jpeg(render_overlay(rgb, seg.detections, skipped))
            for det, crop in zip(seg.detections, crops):
                sid = det.index + 1
                rec = SegmentRecord(
                    id=sid, yolo_class_id=det.class_id, yolo_class_name=det.class_name, yolo_confidence=round(det.confidence, 4),
                    box_xyxy=list(crop.box_px), box_xyxy_raw=[round(v, 2) for v in det.box_xyxy], color_rgb=list(color_for(det.index)),
                    crop_size_wh=list(crop.size_wh), mask_area_px=crop.mask_area_px, mask_fill_ratio=round(crop.mask_fill_ratio, 4),
                    status="classified" if crop.status == "ok" else "skipped", skip_reason=crop.skip_reason)
                if crop.status == "ok":
                    rec.predictions = [{"rank": p.rank, "class_index": p.class_index, "label": p.label,
                                        "softmax_score": round(p.softmax_score, 6)} for p in pred_by_index[det.index]]
                    keys = {"masked": f"seg{sid}_masked", "bbox": f"seg{sid}_bbox", "model_input": f"seg{sid}_input"}
                    record.images[keys["masked"]] = encode_jpeg(crop.masked_crop, 95)
                    record.images[keys["bbox"]] = encode_jpeg(crop.bbox_crop, 95)
                    record.images[keys["model_input"]] = encode_jpeg(previews[det.index], 95)
                    rec.image_keys = keys
                record.segments.append(rec)

        ms = timer.ms
        inference_ms = ms["yolo_ms"] + ms["crop_preprocess_ms"] + ms["classify_ms"]
        n_cls = len(ok)
        record.timings_ms = {
            "decode_ms": round(decoded.decode_ms, 2), "yolo_ms": round(ms["yolo_ms"], 2),
            "crop_preprocess_ms": round(ms["crop_preprocess_ms"], 2), "classify_ms": round(ms["classify_ms"], 2),
            "render_encode_ms": round(ms["render_encode_ms"], 2),
            "inference_ms": round(inference_ms, 2),               # yolo + crop/preprocess + classify
            "total_ms": round(decoded.decode_ms + timer.total_ms(), 2),   # wall clock incl. decode and rendering/encoding
            "classify_ms_per_segment": round(ms["classify_ms"] / n_cls, 2) if n_cls else None,
            "yolo_ultralytics_breakdown_ms": seg.ultralytics_speed_ms,
            "num_segments": len(seg.detections), "num_classified": n_cls,
        }
        if len(seg.detections) >= self.settings.max_segments:
            record.notes.append(f"{len(seg.detections)} segments reached the max_segments cap ({self.settings.max_segments}); "
                                "lower-confidence segments may have been dropped.")
        if not seg.detections:
            record.notes.append("YOLO found no regions above the confidence threshold; the classifier was not called.")
