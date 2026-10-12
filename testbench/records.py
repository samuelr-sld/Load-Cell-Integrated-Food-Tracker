"""Plain data records for one processed image. Serialization lives here so API routes stay thin."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

SCORE_NOTE = ("softmax_score is the raw softmax output of the classifier. It is not a calibrated probability "
              "and must not be read as certainty.")

# A segment with a valid crop is "classified" (compare mode: MobileNet ran) or "identified" (yolo_only: YOLO's class is
# the identity). Anything else is "skipped" and has no crops.
USABLE_STATUSES = ("classified", "identified")

PREPROCESSING_DESCRIPTION = {
    "source": "mobilenet_pipeline/scripts/train_mobnet.py data_transforms['val']",
    "steps": ["masked RGB crop (background = black, 0)", "LetterboxPad(fill=0) to a square", "Resize((224, 224)) bilinear",
              "ToTensor", "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"],
    "random_augmentation": False,
}


@dataclass
class Feedback:
    verdict: str                      # "correct" | "incorrect" | "ambiguous"
    actual_label: Optional[str]       # normalized (lower_snake_case), only with verdict == "incorrect"
    actual_label_raw: Optional[str]
    in_taxonomy: Optional[bool]       # is actual_label one of the classifier's classes? None when no label given
    updated_at: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class SegmentRecord:
    id: int                           # 1-based, the '#n' shown on the overlay
    yolo_class_id: int
    yolo_class_name: str
    yolo_confidence: float
    box_xyxy: list[int]               # integer, clamped box used for both crops
    box_xyxy_raw: list[float]         # as emitted by YOLO
    color_rgb: list[int]              # colour used for this segment on the overlay (so the UI legend matches)
    crop_size_wh: list[int]
    mask_area_px: int
    mask_fill_ratio: float
    status: str                       # "classified" | "identified" | "skipped"
    skip_reason: Optional[str] = None
    predictions: list[dict] = field(default_factory=list)   # MobileNet: rank, class_index, label, softmax_score (compare mode only)
    image_keys: dict = field(default_factory=dict)          # masked / bbox / mask (/ model_input in compare mode) -> image key
    feedback: Optional[Feedback] = None                     # verdict on the MobileNet top-1 (kept for the original UI/API)
    yolo_feedback: Optional[Feedback] = None                # verdict on the YOLO class
    region_id: Optional[int] = None                         # the physical food region this segment belongs to (None when skipped)
    duplicate_of: Optional[int] = None                      # primary segment id when this segment re-describes the same region
    mask_iou_with_primary: Optional[float] = None

    def to_dict(self, result_id: str) -> dict:
        d = dict(self.__dict__)
        d["feedback"] = self.feedback.to_dict() if self.feedback else None
        d["yolo_feedback"] = self.yolo_feedback.to_dict() if self.yolo_feedback else None
        d["image_urls"] = {k: f"/api/results/{result_id}/images/{v}" for k, v in self.image_keys.items()}
        d.pop("image_keys")
        d["top1"] = self.predictions[0] if self.predictions else None
        # Documented schema. `id`, `predictions` and `top1` above are the original names, kept so existing clients still work.
        d["segment_id"] = self.id
        d["primary_identity"] = {"source": "yolo", "class_id": self.yolo_class_id, "class_name": self.yolo_class_name,
                                 "confidence": self.yolo_confidence, "taxonomy": "FoodSeg103 ingredients"}
        d["mobilenet"] = {"taxonomy": "Food-101 dishes", "top1": d["top1"], "predictions": self.predictions} if self.predictions else None
        return d


@dataclass
class InferenceRecord:
    result_id: str
    filename: str
    created_at: str                   # UTC ISO-8601
    status: str                       # "completed" | "failed"
    error: Optional[str]
    image: dict                       # width, height, original_width/height, downscaled, source_format
    device: str
    segments: list[SegmentRecord] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    settings_snapshot: dict = field(default_factory=dict)
    images: dict[str, bytes] = field(default_factory=dict)   # image key -> encoded bytes (JPEG; "seg{n}_mask" is PNG)
    mode: str = "compare"             # "yolo_only" | "compare"
    regions: list = field(default_factory=list)   # list[regions.RegionRecord]: duplicate-resolved foods awaiting/after user confirmation
    # Full-frame boolean mask per usable segment id, np.packbits-packed (the crops only keep the box). Exact copy of the mask
    # the crops were cut from, so plate area is computed from the very pixels YOLO produced.
    segment_masks: dict[int, bytes] = field(default_factory=dict)
    plate: Optional[dict] = None          # {"geometry": PlateGeometry (inference-image pixels), "input": calibration as submitted}
    total_weight: Optional[dict] = None   # {"net_weight_g", "source", "simulated", "set_at", ...}: one weight for the whole plate

    def size_bytes(self) -> int:
        return (sum(len(b) for b in self.images.values()) + sum(len(b) for b in self.segment_masks.values())
                + 2048 + 1024 * len(self.segments))

    def image_urls(self) -> dict:
        return {k: f"/api/results/{self.result_id}/images/{k}" for k in ("original", "overlay") if k in self.images}

    def to_summary(self) -> dict:
        classified = [s for s in self.segments if s.status == "classified"]
        usable = [s for s in self.segments if s.status in USABLE_STATUSES]
        return {
            "result_id": self.result_id, "filename": self.filename, "created_at": self.created_at, "status": self.status,
            "error": self.error, "device": self.device, "mode": self.mode,
            "num_segments": len(self.segments), "num_classified": len(classified), "num_identified": len(usable),
            "num_evaluated": sum(1 for s in classified if s.feedback and s.feedback.verdict in ("correct", "incorrect")),
            "num_regions": len(self.regions), "num_confirmed": sum(1 for r in self.regions if r.status == "confirmed"),
            "num_yolo_evaluated": sum(1 for s in usable if s.yolo_feedback and s.yolo_feedback.verdict in ("correct", "incorrect")),
            "total_ms": self.timings_ms.get("total_ms"), "image_urls": self.image_urls(),
        }

    def region(self, region_id: int):
        return next((r for r in self.regions if r.id == region_id), None)

    def to_detail(self, nutrition=None) -> dict:
        return {
            **self.to_summary(),
            "image": self.image,
            "segments": [s.to_dict(self.result_id) for s in self.segments],
            "timings_ms": self.timings_ms,
            "notes": self.notes,
            "inference_settings": self.settings_snapshot,
            "plate": ({**self.plate["geometry"].to_dict(), "input": self.plate["input"]} if self.plate else None),
            "total_weight": self.total_weight,
            "identity_source": "yolo",    # the segment's food identity is always YOLO's class; MobileNet is comparison only
            # Duplicate-resolved foods. Identity is a *suggestion* until the user confirms it; `segments` still lists every raw detection.
            "regions": [r.to_dict(nutrition) for r in self.regions] if nutrition is not None else [],
            "classifier_preprocessing": PREPROCESSING_DESCRIPTION if self.mode == "compare" else None,
            "score_note": SCORE_NOTE,
        }
