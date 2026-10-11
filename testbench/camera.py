"""External-camera acquisition: enumerate, probe and grab ONE still. No image processing happens here.

Safety rules this module enforces:
- Nothing opens a camera unless a method is called; the device is opened per operation and always released (try/finally),
  so a still capture never holds the camera between requests.
- Every OpenCV open/read runs in a worker thread with a hard timeout. A driver that hangs can therefore never block an API
  request; the camera is then reported busy (single-flight lock, released by the worker when the driver finally returns).
- Camera index 0 is never assumed to be the external webcam: the source is explicit and selectable (index or device path).
- Only standard cv2.VideoCapture calls are used; no driver, registry or OS camera setting is touched. Requested width/height are
  per-handle capture properties and the actual size is always read back and reported.
"""
from __future__ import annotations

import logging
import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

log = logging.getLogger("testbench.camera")

MAX_INDEX = 63
_DEVICE_PATH = re.compile(r"^/dev/(video\d+|v4l/by-(id|path)/[\w.:\-]+)$")     # Linux/Raspberry Pi only; nothing else (no URLs) is accepted

Source = Union[int, str]


class CameraError(Exception):
    """code is machine-readable: invalid_camera | camera_unavailable | camera_busy | camera_timeout | capture_failed | opencv_unavailable"""
    def __init__(self, code: str, message: str, http_status: int = 409):
        super().__init__(message)
        self.code, self.message, self.http_status = code, message, http_status


def parse_source(raw) -> Source:
    """'1' / 1 -> index 1;  '/dev/video0' -> device path. Anything else (negative, huge, URL, ..) is rejected."""
    if isinstance(raw, bool):
        raise CameraError("invalid_camera", "camera source must be an index or a /dev/video* path", 422)
    if isinstance(raw, int) or (isinstance(raw, str) and raw.strip().isdigit()):
        idx = int(raw)
        if not 0 <= idx <= MAX_INDEX:
            raise CameraError("invalid_camera", f"camera index must be between 0 and {MAX_INDEX} (got {idx})", 422)
        return idx
    if isinstance(raw, str) and _DEVICE_PATH.match(raw.strip()) and ".." not in raw:
        return raw.strip()
    raise CameraError("invalid_camera", f"'{raw}' is not a camera index (0-{MAX_INDEX}) or a /dev/video* device path", 422)


def camera_id(source: Source) -> str:
    """Stable key under which this camera's plate calibration is stored. NOTE: a USB index can change if devices are re-plugged
    in a different order; the device path (Linux) or re-saving the calibration is the remedy."""
    return f"index:{source}" if isinstance(source, int) else f"path:{source}"


def parse_camera_id(raw: str) -> str:
    """Validate a camera id received from a client ('index:1' / 'path:/dev/video0')."""
    kind, _, rest = str(raw).partition(":")
    if kind not in ("index", "path") or not rest:
        raise CameraError("invalid_camera", f"'{raw}' is not a camera id (expected 'index:N' or 'path:/dev/videoN')", 422)
    return camera_id(parse_source(rest))


def _backend_flag(cv2, name: str, source: Source) -> int:
    if isinstance(source, str):
        return cv2.CAP_V4L2
    if name == "auto":
        if sys.platform.startswith("win"):
            return cv2.CAP_DSHOW
        return cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    return {"any": cv2.CAP_ANY, "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY), "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
            "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)}[name]


@dataclass
class Frame:
    bgr: np.ndarray
    source: Source
    width: int
    height: int
    requested: Optional[tuple[int, int]]
    backend: str
    captured_ms: float
    from_preview: bool = False      # taken from the running preview's handle (no second open of the device)

    def info(self) -> dict:
        return {"camera_id": camera_id(self.source), "source": self.source, "width": self.width, "height": self.height,
                "requested_size": list(self.requested) if self.requested else None, "backend": self.backend,
                "capture_ms": round(self.captured_ms, 1), "from_preview": self.from_preview}


class CameraService:
    """`capture_factory(source, backend_flag)` returns a cv2.VideoCapture-like object (isOpened/read/set/get/release).
    Tests inject a fake; production uses cv2.VideoCapture."""

    def __init__(self, backend: str = "auto", width: int = 0, height: int = 0, warmup_frames: int = 5, timeout_s: float = 8.0,
                 max_probe_index: int = 5, capture_factory: Optional[Callable] = None, preview_fps: float = 10.0,
                 preview_width: int = 960, preview_quality: int = 70, preview_idle_timeout_s: float = 5.0):
        self.preview_fps, self.preview_width = float(preview_fps), int(preview_width)
        self.preview_quality, self.preview_idle_timeout_s = int(preview_quality), float(preview_idle_timeout_s)
        self._preview = None                   # the PreviewSession that currently owns (or is releasing) a device, if any
        self._preview_lock = threading.RLock()
        self.backend, self.width, self.height = backend, max(0, int(width)), max(0, int(height))
        self.warmup_frames, self.timeout_s, self.max_probe_index = max(0, int(warmup_frames)), float(timeout_s), int(max_probe_index)
        self._factory = capture_factory
        self._busy = threading.Lock()          # single-flight: one device operation at a time (also guards a hung driver)

    # ------------------------------------------------------------------ plumbing
    def _cv2(self):
        try:
            import cv2
            return cv2
        except ImportError as exc:                                   # pragma: no cover - opencv is a project dependency
            raise CameraError("opencv_unavailable", "OpenCV (cv2) is not installed", 503) from exc

    def _open(self, source: Source):
        cv2 = self._cv2()
        flag = _backend_flag(cv2, self.backend, source)
        cap = self._factory(source, flag) if self._factory else cv2.VideoCapture(source, flag)
        return cap

    def _run(self, fn: Callable, source: Source):
        """Run fn() (which opens, uses and releases the device itself) under the timeout and the single-flight lock."""
        if not self._busy.acquire(timeout=min(1.0, self.timeout_s)):
            raise CameraError("camera_busy", "Another camera operation is still in progress (or a previous one is stuck in the driver). Try again shortly.")
        out: dict = {}

        def worker():
            try:
                out["value"] = fn()
            except BaseException as exc:                              # noqa: BLE001 - re-raised in the caller
                out["error"] = exc
            finally:
                self._busy.release()                                  # released by the worker so a hung driver keeps the camera marked busy

        t = threading.Thread(target=worker, name=f"camera-{camera_id(source)}", daemon=True)
        t.start()
        t.join(self.timeout_s)
        if t.is_alive():
            log.error("camera %s did not respond within %.1fs", camera_id(source), self.timeout_s)
            raise CameraError("camera_timeout", f"Camera {camera_id(source)} did not respond within {self.timeout_s:g} s "
                              "(driver stuck or device busy). No result was produced.", 504)
        if "error" in out:
            raise out["error"]
        return out["value"]

    def _grab(self, source: Source) -> Frame:
        t0 = time.perf_counter()
        cap = None
        try:
            try:
                cap = self._open(source)
            except CameraError:
                raise
            except Exception as exc:                                   # noqa: BLE001 - driver/OpenCV errors differ by platform
                raise CameraError("camera_unavailable", f"Camera {camera_id(source)} could not be opened ({type(exc).__name__}).") from exc
            if cap is None or not cap.isOpened():
                raise CameraError("camera_unavailable", f"Camera {camera_id(source)} is not available: not connected, wrong index, or in use by another application.")
            requested = None
            cv2 = self._cv2()
            if self.width and self.height:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                requested = (self.width, self.height)
            frame = None
            ok = False
            for _ in range(self.warmup_frames + 1):                    # warm-up frames are read and thrown away; the last one is kept
                ok, frame = cap.read()
                if not ok:
                    break
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                raise CameraError("capture_failed", f"Camera {camera_id(source)} opened but returned no frame (disconnected, or still starting).")
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            h, w = frame.shape[:2]
            return Frame(frame, source, int(w), int(h), requested, self.backend, (time.perf_counter() - t0) * 1000.0)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:                                      # noqa: BLE001
                    log.exception("camera release failed")

    # ------------------------------------------------------------------ public
    def capture(self, source) -> Frame:
        src = parse_source(source)
        sess = self.active_preview()
        if sess is not None:                                           # a preview owns a device: never open a second handle
            if sess.source != src:
                raise CameraError("camera_in_use", f"A live preview is running on {camera_id(sess.source)}. Stop it (or capture from that camera) "
                                  f"before using {camera_id(src)}.", 409)
            t0 = time.perf_counter()
            bgr, _ = sess.next_frame(self.timeout_s)                   # a frame taken after this call, at full camera resolution
            h, w = bgr.shape[:2]
            return Frame(bgr.copy(), src, int(w), int(h), sess.requested, self.backend, (time.perf_counter() - t0) * 1000.0, from_preview=True)
        return self._run(lambda: self._grab(src), src)

    # ------------------------------------------------------------------ live preview
    def active_preview(self):
        with self._preview_lock:
            return self._preview if self._preview is not None and self._preview.active else None

    def start_preview(self, source, fps: Optional[float] = None, preview_width: Optional[int] = None) -> dict:
        """Open the camera once and keep it open, publishing frames. Idempotent for the same camera; a different camera replaces it.
        Blocks at most timeout_s (until the first frame arrives), so camera errors surface here, not later in the stream."""
        from .camera_preview import PreviewSession
        src = parse_source(source)
        with self._preview_lock:
            cur = self._preview
            if cur is not None and cur.alive:
                if cur.active and cur.source == src:
                    cur.add_viewer(); cur.remove_viewer()                  # refresh the idle clock: someone just asked for it again
                    return cur.status()
                cur.request_stop("replaced")
                if not cur.join(self.timeout_s):
                    raise CameraError("camera_busy", "The previous preview is still shutting down (driver slow to release). Try again shortly.")
            if not self._busy.acquire(timeout=min(1.0, self.timeout_s)):
                raise CameraError("camera_busy", "Another camera operation is still in progress. Try again shortly.")
            released = threading.Event()

            def open_done():                                               # hand the single-flight lock back once the device is open (or failed)
                if not released.is_set():
                    released.set()
                    self._busy.release()

            sess = PreviewSession(self, src, fps or self.preview_fps, preview_width or self.preview_width, self.preview_quality,
                                  self.preview_idle_timeout_s, self.timeout_s, open_done)
            self._preview = sess
            sess.start()
        sess.wait_started(self.timeout_s)
        if sess.state == "error":
            raise sess.error
        if sess.state == "starting":                                       # open/first read hung: do not leave it half-owned
            sess.request_stop("timeout")
            raise CameraError("camera_timeout", f"Camera {camera_id(src)} produced no frame within {self.timeout_s:g} s "
                              "(driver stuck or device busy). The preview was not started.", 504)
        return sess.status()

    def stop_preview(self) -> dict:
        with self._preview_lock:
            cur = self._preview
        if cur is not None and cur.alive:
            cur.request_stop("stopped")
            cur.join(self.timeout_s)
        return self.preview_status()

    def preview_status(self) -> dict:
        with self._preview_lock:
            cur = self._preview
        if cur is None:
            return {"active": False, "state": "stopped", "camera_id": None}
        return cur.status()

    def shutdown(self) -> None:
        """App shutdown: release the device if a preview still holds it."""
        self.stop_preview()

    def probe(self, source) -> dict:
        """Open, read one frame, release. Never raises for a missing camera: the failure is the report."""
        src = parse_source(source)
        base = {"camera_id": camera_id(src), "source": src}
        sess = self.active_preview()
        if sess is not None and sess.source == src:                    # already open for the preview: report it, do not compete for the device
            st = sess.status()
            return {**base, "connected": True, "capture_ok": st["width"] is not None, "width": st["width"], "height": st["height"], "backend": self.backend,
                    "capture_ms": None, "error": None, "error_code": None, "in_preview": True}
        try:
            fr = self.capture(src) if sess is None else self._run(lambda: self._grab(src), src)     # a different device than the preview's
            return {**base, "connected": True, "capture_ok": True, "width": fr.width, "height": fr.height, "backend": fr.backend,
                    "capture_ms": round(fr.captured_ms, 1), "error": None, "error_code": None}
        except CameraError as exc:
            return {**base, "connected": exc.code in ("capture_failed",), "capture_ok": False, "width": None, "height": None,
                    "backend": self.backend, "capture_ms": None, "error": exc.message, "error_code": exc.code}

    def enumerate(self, max_index: Optional[int] = None) -> list[dict]:
        n = max(1, min(int(max_index if max_index is not None else self.max_probe_index), MAX_INDEX + 1))
        return [self.probe(i) for i in range(n)]
