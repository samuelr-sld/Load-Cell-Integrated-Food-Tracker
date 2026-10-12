"""Persistent per-camera plate calibration and camera selection (one small JSON file, written atomically).

A calibration is a *claim about one camera's framing at one resolution*. It is therefore stored per camera id, carries the
resolution it was made at, and is never applied blindly: plate_localization.py checks resolution/aspect compatibility and
re-verifies the saved rim against the current image before using it.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import LOCALIZATION_MODES
from .plate_analysis import PlateGeometry

log = logging.getLogger("testbench.calibration")

METHODS = ("manual_three_point", "manual_numeric", "automatic_opencv")
VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CalibrationStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ file
    def _read(self) -> dict:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("calibrations", {}), dict):
                return raw
            raise ValueError("unexpected structure")
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            log.warning("camera state file %s is unreadable (%s); treating it as empty. The next save replaces it.", self.path, exc)
            return {}

    def _write(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"version": VERSION, **state}, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)                      # atomic: a crash never leaves a half-written calibration

    # ------------------------------------------------------------------ selection / mode
    def selected_camera(self) -> Optional[str]:
        with self._lock:
            return self._read().get("selected_camera")

    def set_selected_camera(self, source: str) -> None:
        with self._lock:
            state = self._read()
            state["selected_camera"] = str(source)
            self._write(state)

    def localization_mode(self) -> Optional[str]:
        with self._lock:
            mode = self._read().get("localization_mode")
            return mode if mode in LOCALIZATION_MODES else None

    def set_localization_mode(self, mode: str) -> None:
        if mode not in LOCALIZATION_MODES:
            raise ValueError(f"mode must be one of {', '.join(LOCALIZATION_MODES)}")
        with self._lock:
            state = self._read()
            state["localization_mode"] = mode
            self._write(state)

    # ------------------------------------------------------------------ calibrations
    def get(self, camera_id: str) -> Optional[dict]:
        with self._lock:
            return self._read().get("calibrations", {}).get(camera_id)

    def all(self) -> dict:
        with self._lock:
            return dict(self._read().get("calibrations", {}))

    def save(self, camera_id: str, geometry: PlateGeometry, method: str, evidence: Optional[dict] = None) -> dict:
        if method not in METHODS:
            raise ValueError(f"method must be one of {', '.join(METHODS)}")
        rec = {"camera_id": camera_id, "image_width": geometry.image_width, "image_height": geometry.image_height,
               "center_x": geometry.center_x, "center_y": geometry.center_y, "radius_px": geometry.radius_px,
               "plate_diameter_mm": geometry.plate_diameter_mm, "calibrated_at": _now(), "method": method, "evidence": evidence}
        with self._lock:
            state = self._read()
            state.setdefault("calibrations", {})[camera_id] = rec        # replaces any earlier calibration of this camera
            self._write(state)
        return rec

    def clear(self, camera_id: str) -> bool:
        with self._lock:
            state = self._read()
            existed = state.get("calibrations", {}).pop(camera_id, None) is not None
            if existed:
                self._write(state)
            return existed


def geometry_of(cal: dict) -> PlateGeometry:
    """Raises ValueError if the stored record is not a valid PlateGeometry (hand-edited or corrupted file)."""
    try:
        return PlateGeometry(cal["center_x"], cal["center_y"], cal["radius_px"], cal["image_width"], cal["image_height"], cal.get("plate_diameter_mm"))
    except KeyError as exc:
        raise ValueError(f"stored calibration is missing {exc}") from None
