"""Live preview: ONE background thread owns the ONE cv2.VideoCapture handle for as long as a preview runs.

Why a session object instead of "open/read/release per request": a preview holds the device continuously, so a still capture
or probe on the same camera must NOT open a second handle (that fails or fights over the device). While a session is active,
CameraService serves captures and probes of that camera from the session's latest full-resolution frame instead.

CPU: frames are only decoded (`retrieve`) at the preview rate (the rest are cheap `grab`s that keep the driver buffer fresh), and
the downscaled preview JPEG is only encoded while at least one browser is actually watching the stream.

Release: the handle is released in a `finally` when the session ends for ANY reason: stop requested, no viewer connected for
idle_timeout_s (browser closed / navigated away), the camera stopped delivering frames (unplugged), or an unexpected error.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np

from .camera import CameraError, Source, camera_id

log = logging.getLogger("testbench.camera.preview")

MAX_CONSECUTIVE_FAILURES = 10        # failed grabs in a row (and >= FAIL_WINDOW_S) before the camera is declared gone
FAIL_WINDOW_S = 1.0


class PreviewSession:
    def __init__(self, service, source: Source, fps: float, preview_width: int, jpeg_quality: int, idle_timeout_s: float,
                 stall_timeout_s: float, on_open_done: Callable[[], None]):
        self.service, self.source = service, source
        self.fps, self.preview_width, self.jpeg_quality = float(fps), int(preview_width), int(jpeg_quality)
        self.idle_timeout_s, self.stall_timeout_s = float(idle_timeout_s), float(stall_timeout_s)
        self._on_open_done = on_open_done
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self.state = "starting"                 # starting | running | stopped | error
        self.error: Optional[CameraError] = None
        self.stop_reason: Optional[str] = None
        self.frame: Optional[np.ndarray] = None             # latest full-resolution BGR frame (what a capture returns)
        self.frame_t = 0.0                                  # time.monotonic() of that frame
        self.jpeg: Optional[bytes] = None                   # latest downscaled preview JPEG (only encoded while watched)
        self.seq = 0
        self.frame_size: Optional[tuple[int, int]] = None
        self.requested: Optional[tuple[int, int]] = None
        self.measured_fps = 0.0
        self.frames_published = 0
        self.viewers = 0
        self.started_at = time.monotonic()
        self._viewer_activity = self.started_at             # a freshly started preview gets idle_timeout_s for the browser to connect
        self.thread = threading.Thread(target=self._run, name=f"preview-{camera_id(source)}", daemon=True)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.thread.start()

    @property
    def alive(self) -> bool:
        return self.thread.is_alive()

    @property
    def active(self) -> bool:
        """Owns (or is about to own) the device and has not been asked to stop."""
        return self.alive and self.state in ("starting", "running") and not self._stop.is_set()

    def request_stop(self, reason: str = "stopped") -> None:
        if self.stop_reason is None:
            self.stop_reason = reason
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    def join(self, timeout: float) -> bool:
        self.thread.join(timeout)
        return not self.thread.is_alive()

    def wait_started(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self.state == "starting" and time.monotonic() < deadline:
                self._cond.wait(max(0.0, min(0.1, deadline - time.monotonic())))

    # ------------------------------------------------------------------ viewers
    def add_viewer(self) -> None:
        with self._cond:
            self.viewers += 1
            self._viewer_activity = time.monotonic()

    def remove_viewer(self) -> None:
        with self._cond:
            self.viewers = max(0, self.viewers - 1)
            self._viewer_activity = time.monotonic()        # the idle clock starts when the LAST viewer leaves

    # ------------------------------------------------------------------ frames
    def latest_jpeg(self) -> tuple[int, Optional[bytes]]:
        with self._cond:
            return self.seq, self.jpeg

    def frame_age_s(self) -> Optional[float]:
        return None if self.frame is None else time.monotonic() - self.frame_t

    def next_frame(self, timeout: float) -> tuple[np.ndarray, float]:
        """A full-resolution frame captured AFTER this call (never a stale one). Used by still capture during preview."""
        asked = time.monotonic()
        deadline = asked + timeout
        with self._cond:
            while not (self.frame is not None and self.frame_t > asked):
                if self.state in ("stopped", "error") or self._stop.is_set():
                    raise self.error or CameraError("capture_failed", "The preview ended before a frame could be captured.")
                left = deadline - time.monotonic()
                if left <= 0:
                    raise CameraError("camera_timeout", f"The preview camera delivered no new frame within {timeout:g} s (stalled or disconnected).", 504)
                self._cond.wait(min(0.1, left))
            return self.frame, self.frame_t

    def status(self) -> dict:
        age = self.frame_age_s()
        stalled = self.state == "running" and age is not None and age > self.stall_timeout_s
        state = "stalled" if stalled else ("stopping" if self.alive and self._stop.is_set() else self.state)
        return {"active": self.active, "state": state, "camera_id": camera_id(self.source), "source": self.source,
                "width": self.frame_size[0] if self.frame_size else None, "height": self.frame_size[1] if self.frame_size else None,
                "requested_size": list(self.requested) if self.requested else None, "target_fps": self.fps, "measured_fps": round(self.measured_fps, 1),
                "preview_width": self.preview_width, "frame_age_ms": None if age is None else round(age * 1000.0),
                "frames": self.frames_published, "viewers": self.viewers, "stop_reason": self.stop_reason,
                "error": ({"code": self.error.code, "message": self.error.message} if self.error else None)}

    # ------------------------------------------------------------------ worker
    def _publish(self, frame: np.ndarray, now: float) -> None:
        import cv2
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        jpeg = None
        if self.viewers > 0:                                 # nobody watching = no encoding work
            h, w = frame.shape[:2]
            small = frame if w <= self.preview_width else cv2.resize(frame, (self.preview_width, max(1, round(h * self.preview_width / w))), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            jpeg = buf.tobytes() if ok else None
        with self._cond:
            if self.frame_t:
                inst = 1.0 / max(now - self.frame_t, 1e-3)
                self.measured_fps = inst if not self.measured_fps else 0.8 * self.measured_fps + 0.2 * inst
            self.frame, self.frame_t, self.frame_size = frame, now, (int(frame.shape[1]), int(frame.shape[0]))
            if jpeg is not None:
                self.jpeg, self.seq = jpeg, self.seq + 1
            self.frames_published += 1
            if self.state == "starting":
                self.state = "running"
            self._cond.notify_all()

    def _run(self) -> None:
        svc, cap = self.service, None
        try:
            try:
                cap = svc._open(self.source)
            except CameraError:
                raise
            except Exception as exc:                          # noqa: BLE001 - driver errors differ by platform
                raise CameraError("camera_unavailable", f"Camera {camera_id(self.source)} could not be opened ({type(exc).__name__}).") from exc
            if cap is None or not cap.isOpened():
                raise CameraError("camera_unavailable", f"Camera {camera_id(self.source)} is not available: not connected, wrong index, or in use by another application.")
            if svc.width and svc.height:
                cv2 = svc._cv2()
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, svc.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, svc.height)
                self.requested = (svc.width, svc.height)
            interval = 1.0 / self.fps
            two_step = callable(getattr(cap, "grab", None)) and callable(getattr(cap, "retrieve", None))
            fails, last_ok = 0, time.monotonic()
            while not self._stop.is_set():
                now = time.monotonic()
                if self.viewers == 0 and now - self._viewer_activity > self.idle_timeout_s:
                    self.stop_reason = "no_viewers"          # browser closed / navigated away / never connected
                    break
                frame = None
                if two_step:
                    ok = cap.grab()                          # cheap: keeps the driver buffer fresh without decoding
                    if ok and now - self.frame_t >= interval:
                        ok, frame = cap.retrieve()
                else:
                    ok, frame = cap.read()
                if ok and (frame is not None or two_step):
                    fails, last_ok = 0, now
                    if frame is not None and getattr(frame, "size", 0):
                        self._publish(frame, time.monotonic())
                        self._on_open_done()                 # first frame in: the device is ours, other operations may proceed
                else:
                    fails += 1
                    if fails >= MAX_CONSECUTIVE_FAILURES and now - last_ok >= FAIL_WINDOW_S:
                        raise CameraError("capture_failed", f"Camera {camera_id(self.source)} stopped delivering frames (unplugged, or taken by another application).")
                    time.sleep(0.05)
                    continue
                if not two_step:
                    time.sleep(max(0.0, interval - (time.monotonic() - now)))
                else:
                    time.sleep(0.001)
        except CameraError as exc:
            self.error = exc
        except BaseException as exc:                          # noqa: BLE001
            log.exception("preview crashed")
            self.error = CameraError("capture_failed", f"The preview stopped unexpectedly ({type(exc).__name__}). See the server log.", 500)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:                             # noqa: BLE001
                    log.exception("camera release failed")
            with self._cond:
                self.state = "error" if self.error else "stopped"
                if self.stop_reason is None:
                    self.stop_reason = "error" if self.error else "stopped"
                self._cond.notify_all()
            self._on_open_done()
