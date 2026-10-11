"""Loads the two existing checkpoints once, validates them, and resolves the inference device.

Nothing here retrains or rewrites a checkpoint. The classifier architecture is derived from the checkpoint's own
tensor shapes (not from a filename or from the task description) and then loaded with strict=True.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from .config import REPO_ROOT, Settings

log = logging.getLogger("testbench.models")

DROPOUT_RATE = 0.2  # train_mobnet.py; irrelevant in eval mode, kept so the module structure matches training

# classifier.0 (Linear) input width distinguishes the two torchvision variants.
_ARCH_BY_CLASSIFIER_IN = {
    960: ("mobilenet_v3_large", models.mobilenet_v3_large),
    576: ("mobilenet_v3_small", models.mobilenet_v3_small),
}


class ModelLoadError(RuntimeError):
    """An essential artifact is missing or incompatible. Never papered over with substitute values."""


@dataclass
class LoadedModels:
    yolo: object
    yolo_names: dict[int, str]
    classifier: nn.Module
    class_names: list[str]
    classifier_arch: str
    device: torch.device
    device_requested: str
    device_fallback_reason: str | None
    info: dict = field(default_factory=dict)


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def resolve_device(preference: str) -> tuple[torch.device, str | None]:
    """Return (device, fallback_reason). fallback_reason is None unless CUDA was wanted but is unusable."""
    pref = (preference or "auto").strip().lower()
    if pref == "cpu":
        return torch.device("cpu"), None
    if pref == "auto" or pref.startswith("cuda"):
        if not torch.cuda.is_available():
            reason = None if pref == "auto" else f"device '{preference}' requested but torch.cuda.is_available() is False"
            return torch.device("cpu"), reason
        dev = torch.device("cuda:0" if pref in ("auto", "cuda") else pref)
        if dev.index is not None and dev.index >= torch.cuda.device_count():
            return torch.device("cpu"), f"device '{preference}' requested but only {torch.cuda.device_count()} CUDA device(s) exist"
        return dev, None
    raise ModelLoadError(f"Unsupported TESTBENCH_DEVICE value '{preference}' (use auto, cpu, cuda or cuda:N)")


def load_classifier(weights: Path, class_names_json: Path | None, device: torch.device):
    """Returns (model, class_names, arch_name, checkpoint_meta)."""
    if not Path(weights).is_file():
        raise ModelLoadError(f"MobileNet checkpoint not found: {weights}")
    try:
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception as exc:  # pickle/format problems
        raise ModelLoadError(f"Could not read MobileNet checkpoint {weights}: {exc}") from exc

    missing = [k for k in ("model_state_dict", "class_names", "num_classes") if k not in (ckpt if isinstance(ckpt, dict) else {})]
    if missing:
        raise ModelLoadError(f"MobileNet checkpoint is missing required keys {missing}; cannot determine class mapping safely")

    state, class_names, num_classes = ckpt["model_state_dict"], list(ckpt["class_names"]), int(ckpt["num_classes"])
    if len(class_names) != num_classes:
        raise ModelLoadError(f"Checkpoint class_names has {len(class_names)} entries but num_classes={num_classes}")

    head = state.get("classifier.3.weight")
    if head is None or head.shape[0] != num_classes:
        raise ModelLoadError(f"Checkpoint head output size {None if head is None else head.shape[0]} != num_classes {num_classes}")

    # The class order that was used in training is the one stored inside the checkpoint. class_names.json (written by the
    # same training run) must agree; if it does not, stop rather than guess which one is right.
    if class_names_json is not None and Path(class_names_json).is_file():
        on_disk = json.loads(Path(class_names_json).read_text(encoding="utf-8"))
        if on_disk != class_names:
            raise ModelLoadError(f"class_names.json ({len(on_disk)} entries) disagrees with the class list stored in the checkpoint")

    in_features = state["classifier.0.weight"].shape[1]
    if in_features not in _ARCH_BY_CLASSIFIER_IN:
        raise ModelLoadError(f"Unrecognised MobileNetV3 variant (classifier.0 input width {in_features})")
    arch_name, ctor = _ARCH_BY_CLASSIFIER_IN[in_features]

    model = ctor(weights=None)  # no ImageNet download: every weight comes from the checkpoint
    head_in = model.classifier[-1].in_features
    model.classifier[2] = nn.Dropout(p=DROPOUT_RATE, inplace=False)
    model.classifier[-1] = nn.Linear(head_in, num_classes)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ModelLoadError(f"Checkpoint weights are incompatible with {arch_name}: {str(exc).splitlines()[0]}") from exc

    model.eval()
    model.requires_grad_(False)
    model.to(device)
    meta = {
        "architecture": arch_name,
        "num_classes": num_classes,
        "checkpoint_best_val_top1": float(ckpt["best_acc_top1"]) if "best_acc_top1" in ckpt else None,
        "parameters": int(sum(p.numel() for p in model.parameters())),
    }
    return model, class_names, arch_name, meta


def load_yolo(weights: Path):
    from ultralytics import YOLO

    if not Path(weights).is_file():
        raise ModelLoadError(f"YOLO checkpoint not found: {weights}")
    try:
        yolo = YOLO(str(weights))
    except Exception as exc:
        raise ModelLoadError(f"Could not load YOLO checkpoint {weights}: {exc}") from exc
    if yolo.task != "segment":
        raise ModelLoadError(f"YOLO checkpoint task is '{yolo.task}', expected 'segment'")
    return yolo, {int(k): str(v) for k, v in yolo.names.items()}


def _warm_up(yolo, classifier: nn.Module, device: torch.device, settings: Settings) -> None:
    """One dummy pass through each model so CUDA init / cuDNN autotune does not distort the first real timing,
    and so an unusable CUDA install fails here instead of on the first upload."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    # low conf + tiny max_det forces the mask code path (retina masks) to execute too
    yolo.predict(noise, device=str(device), imgsz=settings.yolo_imgsz, conf=0.001, max_det=5, retina_masks=True, verbose=False)
    with torch.inference_mode():
        classifier(torch.zeros(1, 3, 224, 224, device=device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build(settings: Settings, device: torch.device, fallback_reason: str | None) -> LoadedModels:
    classifier, class_names, arch, cls_meta = load_classifier(settings.mobilenet_weights, settings.class_names_json, device)
    yolo, yolo_names = load_yolo(settings.yolo_weights)
    _warm_up(yolo, classifier, device, settings)

    gpu = {}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        gpu = {"gpu_name": props.name, "vram_total_mb": round(props.total_memory / 2**20)}
    info = {
        "device": {
            "requested": settings.device, "resolved": str(device), "fallback_reason": fallback_reason,
            "cuda_available": torch.cuda.is_available(), "torch_version": torch.__version__, **gpu,
        },
        "yolo": {
            "weights": _rel(settings.yolo_weights), "task": yolo.task, "num_classes": len([k for k in yolo_names if k >= 0]),
            "imgsz": settings.yolo_imgsz, "conf": settings.yolo_conf, "iou": settings.yolo_iou, "retina_masks": True,
        },
        "classifier": {"weights": _rel(settings.mobilenet_weights), **cls_meta, "input_size": 224},
    }
    return LoadedModels(yolo, yolo_names, classifier, class_names, arch, device, settings.device, fallback_reason, info)


def load_models(settings: Settings) -> LoadedModels:
    device, reason = resolve_device(settings.device)
    try:
        return _build(settings, device, reason)
    except ModelLoadError:
        raise  # artifact problems are not device problems; do not mask them with a CPU retry
    except Exception as exc:
        if device.type != "cuda":
            raise ModelLoadError(f"Model warm-up failed on {device}: {exc}") from exc
        log.warning("CUDA warm-up failed (%s); falling back to CPU", exc)
        torch.cuda.empty_cache()
        return _build(settings, torch.device("cpu"), f"CUDA warm-up failed ({type(exc).__name__}: {exc}); fell back to CPU")
