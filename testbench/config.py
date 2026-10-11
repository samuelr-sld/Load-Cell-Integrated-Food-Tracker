"""Runtime settings. Every value can be overridden with a TESTBENCH_* environment variable."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# yolo_only: YOLO's class is the food identity, MobileNet is never run.
# compare:   YOLO + MobileNet on the same masked crops (the original experimental pipeline), shown side by side.
MODES = ("yolo_only", "compare")

# How the plate circle is found for a new result:
# fixed_calibration:        only a saved (and image-verified) calibration; otherwise ask for manual calibration
# automatic_detection:      only OpenCV detection on the current image; otherwise ask for manual calibration
# automatic_with_fallback:  saved calibration if it verifies, else OpenCV detection, else ask for manual calibration
LOCALIZATION_MODES = ("fixed_calibration", "automatic_detection", "automatic_with_fallback")
CAMERA_BACKENDS = ("auto", "any", "dshow", "msmf", "v4l2")


def _env(name: str, default, cast=str):
    raw = os.environ.get(f"TESTBENCH_{name}")
    return default if raw is None or raw == "" else cast(raw)


def _coefficients() -> dict:
    """TESTBENCH_FOOD_COEFFICIENTS='{"rice": 1.2}': optional per-food weight-per-area factors. Empty unless the operator
    supplies values they have calibrated; none are shipped because none have been measured."""
    raw = os.environ.get("TESTBENCH_FOOD_COEFFICIENTS")
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("TESTBENCH_FOOD_COEFFICIENTS must be a JSON object of label -> coefficient")
    return value


@dataclass(frozen=True)
class Settings:
    # --- model artifacts (located during the repository audit) ---
    yolo_weights: Path = field(default_factory=lambda: _env("YOLO_WEIGHTS", REPO_ROOT / "weights" / "yolo11s_food_best.pt", Path))
    mobilenet_weights: Path = field(default_factory=lambda: _env("MOBILENET_WEIGHTS", REPO_ROOT / "weights" / "mobilenet_v3_food101_best.pt", Path))
    # Saved by train_mobnet.py next to the checkpoint; used only to cross-check the class list stored inside the checkpoint.
    class_names_json: Path = field(default_factory=lambda: _env("CLASS_NAMES_JSON", REPO_ROOT / "mobilenet_pipeline" / "models" / "class_names.json", Path))

    # --- device: "auto" | "cpu" | "cuda" | "cuda:N" ---
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))

    # --- YOLO predict settings. conf=0.25 matches yolo_pipeline/scripts/model_test.py; iou/imgsz are Ultralytics defaults
    # (the checkpoint was trained at imgsz=640). retina_masks only changes mask resolution, never detections. ---
    yolo_conf: float = field(default_factory=lambda: _env("YOLO_CONF", 0.25, float))
    yolo_iou: float = field(default_factory=lambda: _env("YOLO_IOU", 0.7, float))
    yolo_imgsz: int = field(default_factory=lambda: _env("YOLO_IMGSZ", 640, int))
    max_segments: int = field(default_factory=lambda: _env("MAX_SEGMENTS", 40, int))

    # --- upload limits ---
    max_upload_bytes: int = field(default_factory=lambda: _env("MAX_UPLOAD_MB", 15, int) * 1024 * 1024)
    max_files_per_request: int = field(default_factory=lambda: _env("MAX_FILES_PER_REQUEST", 20, int))
    max_image_pixels: int = field(default_factory=lambda: _env("MAX_IMAGE_MEGAPIXELS", 50, int) * 1_000_000)
    # Longest side after decoding. Larger images are downscaled ONCE at ingest and that image is used for every later
    # stage. Keeps retina-resolution masks (N x H x W float32) inside the 4 GB VRAM budget.
    max_image_side: int = field(default_factory=lambda: _env("MAX_IMAGE_SIDE", 1600, int))

    # --- crop validity (extract_local_foodseg.py also required crops > 10 px) ---
    min_mask_pixels: int = field(default_factory=lambda: _env("MIN_MASK_PIXELS", 100, int))
    min_crop_side: int = field(default_factory=lambda: _env("MIN_CROP_SIDE", 10, int))

    classifier_batch_size: int = field(default_factory=lambda: _env("CLASSIFIER_BATCH", 32, int))

    # Mode used when an upload does not ask for one. "compare" keeps the original behaviour.
    default_mode: str = field(default_factory=lambda: _env("DEFAULT_MODE", "compare"))

    # --- in-memory result store ---
    max_results: int = field(default_factory=lambda: _env("MAX_RESULTS", 200, int))
    max_store_bytes: int = field(default_factory=lambda: _env("MAX_STORE_MB", 512, int) * 1024 * 1024)

    # --- same-region duplicate handling: two segments are one physical region only if their *masks* overlap this strongly (IoU) ---
    duplicate_mask_iou: float = field(default_factory=lambda: _env("DUPLICATE_MASK_IOU", 0.7, float))

    # --- USDA nutrition cache (SQLite, populated once by scripts/usda_import.py; the API key is never used by the server) ---
    nutrition_db: Path = field(default_factory=lambda: _env("NUTRITION_DB", REPO_ROOT / "testbench" / "data" / "nutrition.sqlite", Path))

    # --- plate analysis / total-weight allocation (defaults mirror PlateAnalysisConfig / WeightAllocationConfig) ---
    overlap_policy: str = field(default_factory=lambda: _env("OVERLAP_POLICY", "split"))      # "split" | "priority" (YOLO confidence order)
    off_plate_flag_fraction: float = field(default_factory=lambda: _env("OFF_PLATE_FLAG_FRACTION", 0.20, float))
    shared_flag_fraction: float = field(default_factory=lambda: _env("SHARED_FLAG_FRACTION", 0.50, float))
    min_region_px: int = field(default_factory=lambda: _env("MIN_REGION_PX", 50, int))
    strict_coverage: bool = field(default_factory=lambda: _env("STRICT_COVERAGE", "0") in ("1", "true", "True"))   # uncertain coverage rejects instead of warns
    food_coefficients: dict = field(default_factory=_coefficients)

    # --- persistent meal log (separate file from the USDA cache, which is never written by the server) ---
    meals_db: Path = field(default_factory=lambda: _env("MEALS_DB", REPO_ROOT / "testbench" / "data" / "meals.sqlite", Path))

    # --- external camera + plate localization. Nothing here opens a camera at startup. ---
    camera_source: str = field(default_factory=lambda: _env("CAMERA_SOURCE", "0"))               # index ("1") or device path ("/dev/video0"); can be changed at runtime
    camera_backend: str = field(default_factory=lambda: _env("CAMERA_BACKEND", "auto"))          # auto = DirectShow on Windows, V4L2 on Linux
    camera_width: int = field(default_factory=lambda: _env("CAMERA_WIDTH", 1920, int))           # requested capture size (full-resolution stills); 0 = driver default. The ACTUAL size is always reported.
    camera_height: int = field(default_factory=lambda: _env("CAMERA_HEIGHT", 1080, int))
    camera_warmup_frames: int = field(default_factory=lambda: _env("CAMERA_WARMUP_FRAMES", 5, int))   # discarded so auto-exposure settles
    camera_timeout_s: float = field(default_factory=lambda: _env("CAMERA_TIMEOUT_S", 8.0, float))     # hard cap on any open/read; a request never blocks longer
    camera_max_probe_index: int = field(default_factory=lambda: _env("CAMERA_MAX_PROBE_INDEX", 5, int))   # enumeration tries indices 0..N-1
    # --- live preview (MJPEG). The camera still runs at camera_width x camera_height; only the browser copy is downscaled/rate-limited. ---
    preview_fps: float = field(default_factory=lambda: _env("PREVIEW_FPS", 10.0, float))
    preview_width: int = field(default_factory=lambda: _env("PREVIEW_WIDTH", 960, int))                # longest-side-independent: width of the streamed JPEG
    preview_jpeg_quality: int = field(default_factory=lambda: _env("PREVIEW_JPEG_QUALITY", 70, int))
    preview_idle_timeout_s: float = field(default_factory=lambda: _env("PREVIEW_IDLE_TIMEOUT_S", 5.0, float))   # no viewer for this long => camera released
    plate_localization: str = field(default_factory=lambda: _env("PLATE_LOCALIZATION", "automatic_with_fallback"))
    camera_state_file: Path = field(default_factory=lambda: _env("CAMERA_STATE_FILE", REPO_ROOT / "testbench" / "data" / "camera_state.json", Path))
    # A saved calibration is re-checked against the current image (fraction of the saved rim that has an edge on it).
    calibration_verify: bool = field(default_factory=lambda: _env("CALIBRATION_VERIFY", "1") in ("1", "true", "True"))
    calibration_min_support: float = field(default_factory=lambda: _env("CALIBRATION_MIN_SUPPORT", 0.35, float))

    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env("PORT", 8000, int))

    def __post_init__(self):
        if self.default_mode not in MODES:
            raise ValueError(f"TESTBENCH_DEFAULT_MODE must be one of {', '.join(MODES)} (got '{self.default_mode}')")
        if self.plate_localization not in LOCALIZATION_MODES:
            raise ValueError(f"TESTBENCH_PLATE_LOCALIZATION must be one of {', '.join(LOCALIZATION_MODES)} (got '{self.plate_localization}')")
        if not (1 <= self.preview_fps <= 30) or not (160 <= self.preview_width <= 3840) or not (10 <= self.preview_jpeg_quality <= 100):
            raise ValueError("TESTBENCH_PREVIEW_FPS must be 1-30, PREVIEW_WIDTH 160-3840 and PREVIEW_JPEG_QUALITY 10-100")
        if self.camera_backend not in CAMERA_BACKENDS:
            raise ValueError(f"TESTBENCH_CAMERA_BACKEND must be one of {', '.join(CAMERA_BACKENDS)} (got '{self.camera_backend}')")
