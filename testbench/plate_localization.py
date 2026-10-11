"""Decides where the plate circle comes from for one image. Pure strategy: no camera access, no inference.

Priority (never skipping a step that the configured mode allows, never inventing a circle):
  1. saved calibration for THIS camera, if it is compatible with the image AND still sits on a real rim edge in it
  2. automatic OpenCV detection, but only a result the detector itself calls "detected"
  3. otherwise: needs_manual (the caller must ask for manual calibration). An "uncertain" detection is returned only as a
     suggestion and is never applied.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .camera_calibration import CalibrationStore, geometry_of
from .config import LOCALIZATION_MODES, Settings
from .plate_analysis import PlateGeometry
from .plate_detection import AutoPlateDetectionService

SOURCE_SAVED = "saved_calibration"
SOURCE_AUTO = "automatic_detection"


class PlateLocalizer:
    def __init__(self, settings: Settings, store: CalibrationStore, detector: Optional[AutoPlateDetectionService] = None):
        self.settings, self.store = settings, store
        self.detector = detector or AutoPlateDetectionService()

    def mode(self) -> str:
        return self.store.localization_mode() or self.settings.plate_localization

    def _try_saved(self, rgb: np.ndarray, camera_id: Optional[str]) -> tuple[Optional[PlateGeometry], dict]:
        step = {"source": SOURCE_SAVED, "outcome": "skipped", "reason": None}
        if camera_id is None:
            step["reason"] = "image_not_from_a_calibrated_camera"       # an upload cannot be tied to the camera the calibration belongs to
            return None, step
        cal = self.store.get(camera_id)
        if cal is None:
            step["reason"] = "no_saved_calibration"
            return None, step
        h, w = rgb.shape[:2]
        try:
            saved = geometry_of(cal)
            geo = saved if (saved.image_width, saved.image_height) == (w, h) else saved.scaled_to(w, h)
        except ValueError as exc:
            step.update(outcome="rejected", reason="incompatible_resolution_or_invalid_record", detail=str(exc),
                        calibrated_size=[cal.get("image_width"), cal.get("image_height")], image_size=[w, h])
            return None, step
        step["scaled_from"] = None if geo is saved else [saved.image_width, saved.image_height]
        if self.settings.calibration_verify:
            check = self.detector.verify(rgb, geo, self.settings.calibration_min_support)
            step["verification"] = check
            if not check["verified"]:
                step.update(outcome="rejected", reason="calibration_not_supported_by_image")   # camera, plate or framing moved
                return None, step
        step.update(outcome="used", calibrated_at=cal.get("calibrated_at"), method=cal.get("method"))
        return geo, step

    def localize(self, rgb: np.ndarray, camera_id: Optional[str] = None, mode: Optional[str] = None) -> dict:
        mode = mode or self.mode()
        if mode not in LOCALIZATION_MODES:
            raise ValueError(f"mode must be one of {', '.join(LOCALIZATION_MODES)}")
        steps: list[dict] = []
        detection = None

        if mode in ("fixed_calibration", "automatic_with_fallback"):
            geo, step = self._try_saved(rgb, camera_id)
            steps.append(step)
            if geo is not None:
                return self._done(mode, SOURCE_SAVED, geo, steps, None, f"Using the saved calibration for {camera_id}.")

        if mode in ("automatic_detection", "automatic_with_fallback"):
            detection = self.detector.detect(rgb)
            steps.append({"source": SOURCE_AUTO, "outcome": "used" if detection["status"] == "detected" else detection["status"], "reason": detection["reason"]})
            if detection["status"] == "detected":
                p = detection["plate"]
                geo = PlateGeometry(p["center_x"], p["center_y"], p["radius_px"], p["image_width"], p["image_height"])
                return self._done(mode, SOURCE_AUTO, geo, steps, detection, "Plate found automatically from image edges.")

        why = "; ".join(f"{s['source']}: {s['reason'] or s['outcome']}" for s in steps)
        return {"status": "needs_manual", "mode": mode, "source": None, "plate": None, "steps": steps, "detection": detection,
                "suggestion": (detection or {}).get("plate") if (detection or {}).get("status") == "uncertain" else None,
                "message": f"The plate could not be localized reliably ({why}). Calibrate manually (click 3 rim points); nothing was applied."}

    @staticmethod
    def _done(mode: str, source: str, geo: PlateGeometry, steps: list, detection: Optional[dict], message: str) -> dict:
        return {"status": "localized", "mode": mode, "source": source, "plate": geo.to_dict(), "steps": steps, "detection": detection,
                "suggestion": None, "message": message}
