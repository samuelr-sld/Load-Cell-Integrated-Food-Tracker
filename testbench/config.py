"""Runtime settings. Every value can be overridden with a TESTBENCH_* environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default, cast=str):
    raw = os.environ.get(f"TESTBENCH_{name}")
    return default if raw is None or raw == "" else cast(raw)


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

    # --- in-memory result store ---
    max_results: int = field(default_factory=lambda: _env("MAX_RESULTS", 200, int))
    max_store_bytes: int = field(default_factory=lambda: _env("MAX_STORE_MB", 512, int) * 1024 * 1024)

    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env("PORT", 8000, int))
