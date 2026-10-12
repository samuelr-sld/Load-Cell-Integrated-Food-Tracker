"""/api/camera/* routes. Camera acquisition (camera.py), detection (plate_detection.py), strategy (plate_localization.py) and
persistence (camera_calibration.py) stay separate modules; this file only translates HTTP <-> those services.
Inference is deliberately NOT here: a captured frame is fetched as a normal JPEG and posted to the unchanged /api/inference."""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Annotated, Literal, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from .camera import CameraError, CameraService, camera_id, parse_camera_id, parse_source
from .camera_calibration import CalibrationStore
from .config import LOCALIZATION_MODES, Settings
from .plate_analysis import PlateGeometry
from .plate_detection import AutoPlateDetectionService
from .plate_localization import PlateLocalizer

Point = Annotated[list[float], Field(min_length=2, max_length=2)]


class SourceIn(BaseModel):
    model_config = ConfigDict(strict=True)
    source: Optional[int | str] = None            # index or /dev/video* path; omitted = the selected camera


class PreviewStartIn(BaseModel):
    model_config = ConfigDict(strict=True)
    source: Optional[int | str] = None            # omitted = the selected camera
    fps: Optional[int] = Field(None, ge=1, le=30)                    # browser frame rate; default TESTBENCH_PREVIEW_FPS
    preview_width: Optional[int] = Field(None, ge=160, le=1920)      # streamed JPEG width; the camera itself keeps its full resolution


class ModeIn(BaseModel):
    model_config = ConfigDict(strict=True)
    mode: Literal["fixed_calibration", "automatic_detection", "automatic_with_fallback"]


class CaptureRef(BaseModel):
    model_config = ConfigDict(strict=True)
    capture_id: Optional[str] = None              # default: the latest capture


class CalibrationIn(BaseModel):
    """Either three rim points, or centre + radius, in the pixels of image_width x image_height (default: the latest capture)."""
    model_config = ConfigDict(strict=True)
    camera_id: Optional[str] = None               # default: the selected camera
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    radius_px: Optional[float] = None
    rim_points: Optional[list[Point]] = Field(None, min_length=3, max_length=3)
    image_width: Optional[int] = Field(None, gt=0)
    image_height: Optional[int] = Field(None, gt=0)
    plate_diameter_mm: Optional[float] = Field(None, gt=0)
    method: Optional[Literal["manual_three_point", "manual_numeric", "automatic_opencv"]] = None


class LatestCapture:
    """One in-memory still (JPEG). Bounded by design: a new capture replaces the old one."""
    def __init__(self):
        self.lock = threading.Lock()
        self.data: Optional[dict] = None


def decode_rgb(jpeg: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise CameraError("capture_failed", "The stored capture could not be decoded", 500)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_router(settings: Settings, camera: CameraService, store: CalibrationStore, detector: AutoPlateDetectionService,
                 localizer: PlateLocalizer) -> APIRouter:
    router = APIRouter(prefix="/api/camera", tags=["camera"])
    latest = LatestCapture()

    def selected_source():
        raw = store.selected_camera()
        return parse_source(raw if raw is not None else settings.camera_source)

    def resolve_source(raw):
        return selected_source() if raw is None else parse_source(raw)

    def capture_rgb(capture_id: Optional[str]) -> tuple[np.ndarray, dict]:
        with latest.lock:
            cur = latest.data
        if cur is None or (capture_id and capture_id != cur["capture_id"]):
            raise CameraError("no_capture", "No such capture. Capture an image first (POST /api/camera/capture); only the latest one is kept.", 404)
        return decode_rgb(cur["jpeg"]), cur

    @router.get("/status")
    def status():
        src = selected_source()
        cid = camera_id(src)
        return {"selected_camera": {"source": src, "camera_id": cid, "persisted": store.selected_camera() is not None},
                "configured_default": settings.camera_source, "backend": settings.camera_backend,
                "requested_size": [settings.camera_width, settings.camera_height], "timeout_s": settings.camera_timeout_s,
                "localization_mode": localizer.mode(), "localization_modes": list(LOCALIZATION_MODES),
                "preview": camera.preview_status(), "calibration": store.get(cid), "calibration_verify": settings.calibration_verify,
                "latest_capture": ({k: v for k, v in latest.data.items() if k != "jpeg"} if latest.data else None),
                "note": "The camera is only opened when you test or capture; it is never opened at startup."}

    @router.get("/devices")
    async def devices(max_index: Optional[int] = Query(None, ge=1, le=64)):
        """Probes indices 0..max_index-1 by opening each briefly (the camera LED may flash). A camera another program is using reports unavailable."""
        found = await run_in_threadpool(camera.enumerate, max_index)
        return {"devices": found, "available": [d["camera_id"] for d in found if d["capture_ok"]], "selected": camera_id(selected_source())}

    @router.post("/test")
    async def test(body: SourceIn | None = None):
        return await run_in_threadpool(camera.probe, resolve_source(body.source if body else None))

    @router.put("/selection")
    def select(body: SourceIn):
        if body.source is None:
            raise CameraError("invalid_camera", "source is required", 422)
        src = parse_source(body.source)
        store.set_selected_camera(str(src))
        return {"selected_camera": {"source": src, "camera_id": camera_id(src)}, "calibration": store.get(camera_id(src))}

    @router.put("/localization-mode")
    def set_mode(body: ModeIn):
        store.set_localization_mode(body.mode)
        return {"localization_mode": body.mode}

    # ------------------------------------------------------------------ live preview (MJPEG)
    @router.post("/preview/start")
    async def preview_start(body: PreviewStartIn | None = None):
        """Open the camera once and keep it open for the stream. Idempotent for the same camera."""
        body = body or PreviewStartIn()
        st = await run_in_threadpool(camera.start_preview, resolve_source(body.source), body.fps, body.preview_width)
        return {**st, "stream_url": "/api/camera/preview/stream"}

    @router.post("/preview/stop")
    async def preview_stop():
        return await run_in_threadpool(camera.stop_preview)

    @router.get("/preview/status")
    def preview_status():
        return camera.preview_status()

    @router.get("/preview/stream")
    async def preview_stream(request: Request):
        """multipart/x-mixed-replace JPEG stream. Browser disconnect => viewer count drops => the camera is released after the idle timeout."""
        sess = camera.active_preview()
        if sess is None:
            raise CameraError("no_preview", "No live preview is running. Start it first (POST /api/camera/preview/start).", 409)

        async def frames():
            sess.add_viewer()
            last_seq, last_new = -1, time.monotonic()
            try:
                while not await request.is_disconnected():
                    if not sess.active:
                        break                                              # stopped, errored or replaced: end the stream
                    seq, jpeg = sess.latest_jpeg()
                    now = time.monotonic()
                    if jpeg is not None and seq != last_seq:
                        last_seq, last_new = seq, now
                        yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
                    elif now - last_new > settings.camera_timeout_s:
                        sess.request_stop("stalled")                       # camera stopped delivering: release it instead of hanging the viewer
                        break
                    await asyncio.sleep(min(0.05, 0.5 / sess.fps))
            finally:
                sess.remove_viewer()

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store"})

    @router.post("/capture")
    async def capture(body: SourceIn | None = None):
        src = resolve_source(body.source if body else None)
        frame = await run_in_threadpool(camera.capture, src)
        ok, buf = cv2.imencode(".jpg", frame.bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise CameraError("capture_failed", "Could not encode the captured frame", 500)
        info = {"capture_id": uuid.uuid4().hex[:12], "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "jpeg_bytes": len(buf), **frame.info()}
        with latest.lock:
            latest.data = {**info, "jpeg": buf.tobytes()}
        return {**info, "image_url": f"/api/camera/capture/image?capture_id={info['capture_id']}",
                "filename": f"camera_{info['camera_id'].replace(':', '')}_{info['capture_id']}.jpg"}

    @router.get("/capture/image")
    def capture_image(capture_id: Optional[str] = None):
        with latest.lock:
            cur = latest.data
        if cur is None or (capture_id and capture_id != cur["capture_id"]):
            raise CameraError("no_capture", "No such capture (only the latest one is kept).", 404)
        return Response(cur["jpeg"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @router.post("/detect-plate")
    async def detect_plate(body: CaptureRef | None = None):
        """Pure OpenCV detection on the latest capture. Ignores saved calibration. Never applies anything."""
        rgb, cur = capture_rgb(body.capture_id if body else None)
        det = await run_in_threadpool(detector.detect, rgb)
        return {"capture_id": cur["capture_id"], "camera_id": cur["camera_id"], **det}

    @router.post("/localize")
    async def localize(body: CaptureRef | None = None):
        """The configured strategy (saved calibration -> automatic detection -> manual) on the latest capture, using the capture's camera."""
        rgb, cur = capture_rgb(body.capture_id if body else None)
        out = await run_in_threadpool(localizer.localize, rgb, cur["camera_id"], None)
        return {"capture_id": cur["capture_id"], "camera_id": cur["camera_id"], **out}

    # ------------------------------------------------------------------ calibration
    @router.get("/calibration")
    def get_calibration(camera_id_: Optional[str] = Query(None, alias="camera_id")):
        cid = parse_camera_id(camera_id_) if camera_id_ else camera_id(selected_source())
        cal = store.get(cid)
        return {"camera_id": cid, "calibration": cal, "all_cameras": sorted(store.all())}

    @router.put("/calibration")
    def put_calibration(body: CalibrationIn):
        cid = parse_camera_id(body.camera_id) if body.camera_id else camera_id(selected_source())
        w, h = body.image_width, body.image_height
        if w is None or h is None:
            with latest.lock:
                cur = latest.data
            if cur is None or cur["camera_id"] != cid:
                raise CameraError("invalid_calibration", "image_width and image_height are required (or capture an image from this camera first).", 422)
            w, h = cur["width"], cur["height"]
        try:
            if body.rim_points:
                geo = PlateGeometry.from_rim_points(body.rim_points, w, h, body.plate_diameter_mm)
                method = body.method or "manual_three_point"
            elif None in (body.center_x, body.center_y, body.radius_px):
                raise ValueError("give either rim_points (three points on the plate rim) or center_x, center_y and radius_px")
            else:
                geo = PlateGeometry(body.center_x, body.center_y, body.radius_px, w, h, body.plate_diameter_mm)
                method = body.method or "manual_numeric"
        except ValueError as exc:
            raise CameraError("invalid_calibration", str(exc), 422) from None
        return {"saved": store.save(cid, geo, method), "geometry": geo.to_dict()}

    @router.delete("/calibration")
    def delete_calibration(camera_id_: Optional[str] = Query(None, alias="camera_id")):
        cid = parse_camera_id(camera_id_) if camera_id_ else camera_id(selected_source())
        return {"camera_id": cid, "cleared": store.clear(cid), "calibration": None}

    return router
