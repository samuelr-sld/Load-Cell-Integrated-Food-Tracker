"""FastAPI app: local testbench API plus the static debugging UI. Model-specific logic lives in the other modules."""
from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, Optional

import cv2
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from .camera import CameraError, CameraService, parse_camera_id, parse_source
from .camera_api import build_router as build_camera_router, decode_rgb
from .camera_calibration import CalibrationStore
from .config import MODES, Settings
from .evaluation import FeedbackError, build_summary, make_feedback
from .ingest import ImageValidationError, check_filename
from .model_loader import ModelLoadError, load_models
from . import plate_session
from .meal_log import MealLog
from .nutrition import NutritionService, norm_label
from .plate_detection import AutoPlateDetectionService
from .plate_localization import PlateLocalizer
from .regions import MAX_WEIGHT_G, RegionRecord, meal_totals
from .result_store import ResultStore, ResultTooLarge
from .scale_scan import ScanCoordinator, SerialListener, build_router as build_scale_router, foods_from_regions

log = logging.getLogger("testbench.api")
STATIC_DIR = Path(__file__).resolve().parent / "static"


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


class FeedbackIn(BaseModel):
    verdict: Literal["correct", "incorrect", "ambiguous"]
    actual_label: Optional[str] = None
    model: Literal["mobilenet", "yolo"] = "mobilenet"     # which model's prediction the verdict is about


class IdentityIn(BaseModel):
    label: str = Field(min_length=1, max_length=80)
    taxonomy: Literal["foodseg103", "food101", "custom"] = "foodseg103"   # custom = free text, never mapped to USDA
    fdc_id: Optional[int] = None                                            # chosen USDA record when the label has several
    weight_g: Optional[float] = Field(None, gt=0, le=MAX_WEIGHT_G)          # optional manual portion weight


class PortionIn(BaseModel):
    weight_g: Optional[float] = Field(None, gt=0, le=MAX_WEIGHT_G)         # null clears the weight


Point = Annotated[list[float], Field(min_length=2, max_length=2)]       # [x, y]; a list (not tuple) so strict mode accepts JSON arrays


class PlateIn(BaseModel):
    """Either three rim points, or centre + radius. Coordinates are in the pixels of image_width x image_height (default: the
    inference image); a different resolution is rescaled onto the inference image."""
    model_config = ConfigDict(strict=True)       # true/"100" are not weights or coordinates: no silent coercion
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    radius_px: Optional[float] = None
    rim_points: Optional[list[Point]] = Field(None, min_length=3, max_length=3)
    image_width: Optional[int] = Field(None, gt=0)
    image_height: Optional[int] = Field(None, gt=0)
    plate_diameter_mm: Optional[float] = Field(None, gt=0)


class TotalWeightIn(BaseModel):
    model_config = ConfigDict(strict=True)       # true/"100" are not weights or coordinates: no silent coercion
    net_weight_g: float = Field(gt=0, le=MAX_WEIGHT_G)                      # food only, plate/tare already removed
    source: Literal["manual_simulation"] = "manual_simulation"              # a typed number standing in for the future ESP32 load cell
    tare_g: Optional[float] = Field(None, ge=0)                             # informational

    @field_validator("net_weight_g")
    @classmethod
    def _resolution(cls, v: float) -> float:
        # estimates are rounded to 0.1 g; a finer total could not be reproduced by their sum (mass would not be conserved)
        if abs(round(v, 1) - v) > 1e-9:
            raise ValueError("must have at most 1 decimal place (0.1 g resolution)")
        return v


class EntityCountIn(BaseModel):
    model_config = ConfigDict(strict=True)       # true/"100" are not weights or coordinates: no silent coercion
    user_entity_count: Optional[int] = Field(None, ge=1, le=100)            # null clears; separate from the detected segment count


class ExcludedIn(BaseModel):
    model_config = ConfigDict(strict=True)       # true/"100" are not weights or coordinates: no silent coercion
    excluded: bool


class ManualRegionIn(BaseModel):
    model_config = ConfigDict(strict=True)       # true/"100" are not weights or coordinates: no silent coercion
    polygon: list[Point] = Field(min_length=3, max_length=500)   # outline of a food YOLO missed, inference-image pixels


class FinalizeIn(BaseModel):
    model_config = ConfigDict(strict=True)       # true/"100" are not weights or coordinates: no silent coercion
    accept_incomplete: bool = False     # false: only a complete meal can be saved. true: an incomplete one is stored AS incomplete


class AutoPlateIn(BaseModel):
    model_config = ConfigDict(strict=True)
    camera_id: Optional[str] = None     # set only when the image really is a frame from that camera; plain uploads never use saved calibration
    mode: Optional[Literal["fixed_calibration", "automatic_detection", "automatic_with_fallback"]] = None   # default: the configured mode


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def create_app(settings: Settings | None = None, pipeline=None, nutrition: NutritionService | None = None,
               camera: CameraService | None = None) -> FastAPI:
    """`pipeline` lets tests inject a fake; when omitted the real checkpoints are loaded once at startup.
    `camera` lets tests inject a CameraService with a fake capture device; the real one never opens a device until asked."""
    settings = settings or Settings()
    store = ResultStore(settings.max_results, settings.max_store_bytes)
    nutri = nutrition or NutritionService(settings.nutrition_db)
    meals = MealLog(settings.meals_db)
    camera = camera or CameraService(settings.camera_backend, settings.camera_width, settings.camera_height, settings.camera_warmup_frames,
                                     settings.camera_timeout_s, settings.camera_max_probe_index, preview_fps=settings.preview_fps,
                                     preview_width=settings.preview_width, preview_quality=settings.preview_jpeg_quality,
                                     preview_idle_timeout_s=settings.preview_idle_timeout_s)
    calibrations = CalibrationStore(settings.camera_state_file)
    detector = AutoPlateDetectionService()
    localizer = PlateLocalizer(settings, calibrations, detector)

    def run_scan(scan: dict) -> dict:
        """Scale-triggered scan: the same camera.capture + InferencePipeline.process_upload the manual testbench uses."""
        pipe = app.state.pipeline
        if pipe is None or not getattr(pipe, "ready", False):
            raise RuntimeError(app.state.load_error or "Models are not loaded yet")
        raw = calibrations.selected_camera()
        frame = camera.capture(parse_source(raw if raw is not None else settings.camera_source))
        ok, buf = cv2.imencode(".jpg", frame.bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("Could not encode the captured frame")
        record = pipe.process_upload(buf.tobytes(), f"scale_{scan['scan_id']}.jpg", None)
        if record.status == "failed":
            raise RuntimeError(record.error)
        store.add(record)
        log.info("scan %s stored result %s (store now holds %d)", scan["scan_id"], record.result_id, store.stats()["count"])
        foods = foods_from_regions(record.to_detail(nutri)["regions"], nutri)
        # One food: the plate's net weight is its portion. Several: no per-food weights are invented; the user allocates them.
        return {"result_id": record.result_id, "camera_id": frame.info()["camera_id"], "foods": foods, "needs_manual_food": not foods,
                "suggested_grams": scan["net_grams"] if len(foods) == 1 else None}

    coordinator = ScanCoordinator(run_scan, settings.scale_tare_g, settings.scale_min_net_g)
    store.protected = coordinator.active_result_ids
    listener = SerialListener(settings.scale_port, settings.scale_baud, coordinator.handle) if settings.scale_enabled else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if listener:
            listener.start()
        app.state.load_error = None
        if pipeline is not None:
            app.state.pipeline = pipeline
        else:
            app.state.pipeline = None
            try:
                from .orchestrator import InferencePipeline
                models = load_models(settings)
                app.state.pipeline = InferencePipeline.from_models(models, settings)
                dev = models.info["device"]
                log.info("Models loaded: %s + %s on %s%s", models.info["yolo"]["weights"], models.info["classifier"]["weights"],
                         dev["resolved"], f" ({dev.get('gpu_name')})" if dev.get("gpu_name") else "")
                if dev["fallback_reason"]:
                    log.warning("Device fallback: %s", dev["fallback_reason"])
            except ModelLoadError as exc:
                app.state.load_error = str(exc)
                log.error("Model loading failed: %s", exc)
        yield
        if listener:
            listener.stop()
        camera.shutdown()          # release the camera if a live preview still holds it

    app = FastAPI(title="Food Recognition Inference Testbench", version="0.1.0", lifespan=lifespan,
                  docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
    app.state.settings, app.state.store, app.state.scale = settings, store, coordinator

    # ----------------------------------------------------------------- errors
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return JSONResponse(_error_body(exc.code, exc.message), status_code=exc.status)

    @app.exception_handler(CameraError)
    async def _camera_error(_: Request, exc: CameraError):
        return JSONResponse(_error_body(exc.code, exc.message), status_code=exc.http_status)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        msg = "; ".join(f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}" for e in exc.errors())
        return JSONResponse(_error_body("invalid_request", msg), status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        return JSONResponse(_error_body("http_error", str(exc.detail)), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(_error_body("internal_error", "Internal server error. Details are in the server log."), status_code=500)

    if settings.host not in ("127.0.0.1", "localhost", "::1"):
        from ai_gateway.app import load_token       # same pairing token the phone already holds for the AI gateway
        pairing = load_token()

        @app.middleware("http")
        async def _lan_guard(request: Request, call_next):
            # Reachable from the Wi-Fi: only the scan routes, and only with the token. The rest of the testbench stays laptop-only.
            # The laptop itself counts as local even when it browses to its own Wi-Fi address (client IP == the address it connected to).
            if request.client and request.client.host not in ("127.0.0.1", "::1", (request.scope.get("server") or (None,))[0]):
                if not request.url.path.startswith(("/api/scale/", "/api/scans/")):
                    return JSONResponse(_error_body("forbidden", "Only the scan API is available over the network."), status_code=403)
                if not hmac.compare_digest(request.headers.get("authorization", "").encode(), f"Bearer {pairing}".encode()):
                    return JSONResponse(_error_body("unauthorized", "Missing or wrong pairing token."), status_code=401)
            return await call_next(request)

    @app.middleware("http")
    async def _body_size_guard(request: Request, call_next):
        # Multipart bodies are parsed before the route runs, so oversized requests must be refused up front.
        if request.method == "POST" and request.url.path == "/api/inference":
            declared = request.headers.get("content-length")
            limit = settings.max_files_per_request * settings.max_upload_bytes + 1024 * 1024
            if declared and declared.isdigit() and int(declared) > limit:
                return JSONResponse(_error_body("request_too_large", f"Request body exceeds {limit // 2**20} MB"), status_code=413)
        return await call_next(request)

    def require_pipeline():
        p = app.state.pipeline
        if p is None or not getattr(p, "ready", False):
            raise ApiError(503, "models_not_loaded", app.state.load_error or "Models are not loaded yet")
        return p

    def get_record(result_id: str):
        try:
            return store.get(result_id)
        except KeyError:
            raise ApiError(404, "result_not_found", f"No result '{result_id}'. It may have been cleared or evicted from memory.") from None

    # ----------------------------------------------------------------- routes
    @app.get("/api/health")
    def health():
        p = app.state.pipeline
        body = {"status": "ok" if p is not None else "error", "models_loaded": p is not None, "load_error": app.state.load_error,
                "store": store.stats(),
                "limits": {"max_upload_mb": settings.max_upload_bytes // 2**20, "max_files_per_request": settings.max_files_per_request,
                           "max_image_side": settings.max_image_side, "accepted_extensions": [".jpg", ".jpeg", ".png"]}}
        if p is not None:
            body.update(p.health())
        return body

    @app.post("/api/inference")
    async def inference(files: list[UploadFile] = File(...), mode: Optional[str] = Form(None)):
        """`mode` (optional form field): `yolo_only` = YOLO class is the identity, MobileNet never runs;
        `compare` = YOLO + MobileNet side by side. Defaults to TESTBENCH_DEFAULT_MODE."""
        pipe = require_pipeline()
        if mode and mode not in MODES:
            raise ApiError(422, "invalid_mode", f"mode must be one of {', '.join(MODES)}")
        if len(files) > settings.max_files_per_request:
            raise ApiError(413, "too_many_files", f"At most {settings.max_files_per_request} files per request")
        items: list[dict] = []
        for f in files:
            name = f.filename or "unnamed"
            try:
                check_filename(name)  # reject by name before spending any memory on the body
                data = await f.read(settings.max_upload_bytes + 1)
                if len(data) > settings.max_upload_bytes:
                    raise ImageValidationError("file_too_large", f"'{name}' exceeds the {settings.max_upload_bytes // 2**20} MB upload limit", 413)
                record = await run_in_threadpool(pipe.process_upload, data, name, mode or None)   # blocking ML work off the event loop
            except ImageValidationError as exc:
                items.append({"filename": name, "status": "rejected", "result_id": None, "result": None,
                              "error": {"code": exc.code, "message": exc.message, "http_status": exc.http_status}})
                continue
            finally:
                data = b""
                await f.close()
            if record.status == "failed":
                items.append({"filename": name, "status": "failed", "result_id": record.result_id, "result": None,
                              "error": {"code": "inference_failed", "message": record.error, "http_status": 500}})
                continue
            try:
                store.add(record)
            except ResultTooLarge as exc:
                items.append({"filename": name, "status": "failed", "result_id": None, "result": None,
                              "error": {"code": "result_too_large", "message": str(exc), "http_status": 507}})
                continue
            items.append({"filename": name, "status": "completed", "result_id": record.result_id, "error": None, "result": record.to_detail(nutri)})

        counts = {"received": len(items), **{s: sum(1 for i in items if i["status"] == s) for s in ("completed", "failed", "rejected")}}
        status = 200 if counts["completed"] else (items[0]["error"]["http_status"] if items else 422)
        return JSONResponse({"items": items, "summary": counts}, status_code=status)

    @app.get("/api/results")
    def list_results():
        return {"results": [r.to_summary() for r in store.list()], "store": store.stats()}

    @app.get("/api/results/{result_id}")
    def result_detail(result_id: str):
        return get_record(result_id).to_detail(nutri)

    @app.get("/api/results/{result_id}/images/{image_key}")
    def result_image(result_id: str, image_key: str):
        rec = get_record(result_id)
        blob = rec.images.get(image_key)
        if blob is None:
            raise ApiError(404, "image_not_found", f"Result '{result_id}' has no image '{image_key}'")
        # everything is JPEG except the lossless binary masks
        media = "image/png" if blob.startswith(PNG_MAGIC) else "image/jpeg"
        return Response(blob, media_type=media, headers={"Cache-Control": "private, max-age=600"})

    @app.delete("/api/results")
    def clear_results():
        return {"cleared": store.clear()}

    # ------------------------------------------------- manual evaluation (section 8)
    def get_segment(result_id: str, segment_id: int):
        rec = get_record(result_id)
        seg = next((s for s in rec.segments if s.id == segment_id), None)
        if seg is None:
            raise ApiError(404, "segment_not_found", f"Result '{result_id}' has no segment {segment_id}")
        return seg

    @app.put("/api/results/{result_id}/segments/{segment_id}/feedback")
    def put_feedback(result_id: str, segment_id: int, body: FeedbackIn):
        pipe = require_pipeline()
        with store.lock:
            seg = get_segment(result_id, segment_id)
            taxonomy = pipe.yolo_class_names if body.model == "yolo" else pipe.class_names
            try:
                fb = make_feedback(seg, body.verdict, body.actual_label, taxonomy, body.model)
            except FeedbackError as exc:
                raise ApiError(422, "invalid_feedback", str(exc)) from None
            setattr(seg, "yolo_feedback" if body.model == "yolo" else "feedback", fb)
            return {"result_id": result_id, "segment_id": segment_id, "model": body.model, "feedback": fb.to_dict()}

    @app.delete("/api/results/{result_id}/segments/{segment_id}/feedback")
    def delete_feedback(result_id: str, segment_id: int, model: Literal["mobilenet", "yolo"] = "mobilenet"):
        with store.lock:
            setattr(get_segment(result_id, segment_id), "yolo_feedback" if model == "yolo" else "feedback", None)
        return {"result_id": result_id, "segment_id": segment_id, "model": model, "feedback": None}

    @app.get("/api/evaluation/summary")
    def evaluation_summary():
        with store.lock:
            return build_summary(store.list(), store.evicted_total)

    # ------------------------------------- food identity confirmation + USDA nutrition
    def get_region(result_id: str, region_id: int):
        region = get_record(result_id).region(region_id)
        if region is None:
            raise ApiError(404, "region_not_found", f"Result '{result_id}' has no region {region_id}")
        return region

    @app.get("/api/nutrition/foods")
    def nutrition_foods():
        """Labels that have USDA data (for pickers). Everything else is reported as 'unmapped'."""
        return {"foods": nutri.mapped_labels(), **nutri.stats()}

    @app.put("/api/results/{result_id}/regions/{region_id}/identity")
    def confirm_identity(result_id: str, region_id: int, body: IdentityIn):
        """The user's pick is the food identity; a model confidence never confirms anything. Re-PUT to correct a wrong pick."""
        pipe = require_pipeline()
        label = norm_label(body.label)
        vocab = {"foodseg103": pipe.yolo_class_names, "food101": pipe.class_names}.get(body.taxonomy)
        if vocab is not None and label not in {norm_label(v) for v in vocab}:
            raise ApiError(422, "unknown_label", f"'{body.label}' is not a {body.taxonomy} class. Use taxonomy 'custom' for free text.")
        if not label:
            raise ApiError(422, "invalid_label", "label is empty")
        with store.lock:
            region = get_region(result_id, region_id)
            if body.fdc_id is not None and not any(c["fdc_id"] == body.fdc_id for c in nutri.candidates(body.taxonomy, label)):
                raise ApiError(422, "invalid_fdc_id", f"FDC ID {body.fdc_id} is not a mapped USDA option for '{label}'")
            suggested = any(c["taxonomy"] == body.taxonomy and c["label"] == label for c in region.candidates)
            region.confirm(label, body.taxonomy, suggested, body.fdc_id)
            if body.weight_g is not None:
                region.weight_g = body.weight_g
            return {"result_id": result_id, **region.to_dict(nutri)}

    @app.delete("/api/results/{result_id}/regions/{region_id}/identity")
    def clear_identity(result_id: str, region_id: int):
        with store.lock:
            region = get_region(result_id, region_id)
            region.clear()
            return {"result_id": result_id, **region.to_dict(nutri)}

    @app.put("/api/results/{result_id}/regions/{region_id}/portion")
    def set_portion(result_id: str, region_id: int, body: PortionIn):
        with store.lock:
            region = get_region(result_id, region_id)
            region.weight_g = body.weight_g
            return {"result_id": result_id, **region.to_dict(nutri)}

    @app.get("/api/results/{result_id}/regions/{region_id}/nutrition")
    def region_nutrition(result_id: str, region_id: int):
        with store.lock:
            region = get_region(result_id, region_id)
            return {"result_id": result_id, "region_id": region_id, "identity": region.identity, **region.nutrition(nutri)}

    @app.get("/api/results/{result_id}/nutrition")
    def meal_nutrition(result_id: str):
        with store.lock:
            return {"result_id": result_id, **meal_totals(get_record(result_id).regions, nutri)}

    # ------------------------------- plate analysis, total-weight allocation, meal log
    def analysis_for(rec):
        return plate_session.compute(rec, settings, nutri)

    @app.put("/api/results/{result_id}/plate")
    def set_plate(result_id: str, body: PlateIn):
        with store.lock:
            rec = get_record(result_id)
            try:
                geo = plate_session.build_geometry(rec, **body.model_dump())
            except ValueError as exc:
                raise ApiError(422, "invalid_plate", str(exc)) from None
            rec.plate = {"geometry": geo, "input": {**body.model_dump(), "source": "manual", "set_at": plate_session.now()}}
            return {"result_id": result_id, "plate": geo.to_dict(), "input": rec.plate["input"]}

    @app.post("/api/results/{result_id}/plate/auto")
    async def auto_plate(result_id: str, body: AutoPlateIn | None = None):
        """Localize the plate with the configured strategy (saved calibration -> OpenCV detection -> manual) on this result's own image,
        so the circle lives in exactly the pixels the masks were made in. The circle is applied only when it is verified/detected;
        otherwise status is 'needs_manual', nothing is applied and any existing plate is left untouched."""
        body = body or AutoPlateIn()
        if body.camera_id is not None:
            parse_camera_id(body.camera_id)
        with store.lock:
            jpeg = get_record(result_id).images.get("original")
        if jpeg is None:
            raise ApiError(409, "no_image", "This result has no stored image to localize the plate on")
        out = await run_in_threadpool(localizer.localize, decode_rgb(jpeg), body.camera_id, body.mode)   # same pixels the masks were made from
        if out["status"] == "localized":
            p = out["plate"]
            with store.lock:
                rec = get_record(result_id)
                geo = plate_session.build_geometry(rec, center_x=p["center_x"], center_y=p["center_y"], radius_px=p["radius_px"],
                                                   image_width=p["image_width"], image_height=p["image_height"], plate_diameter_mm=p.get("plate_diameter_mm"))
                rec.plate = {"geometry": geo, "input": {
                    "center_x": geo.center_x, "center_y": geo.center_y, "radius_px": geo.radius_px, "image_width": geo.image_width,
                    "image_height": geo.image_height, "plate_diameter_mm": geo.plate_diameter_mm, "source": out["source"], "camera_id": body.camera_id,
                    "mode": out["mode"], "evidence": (out["detection"] or {}).get("evidence"), "set_at": plate_session.now()}}
                out["plate"] = {**geo.to_dict(), "input": rec.plate["input"]}
        return {"result_id": result_id, "applied": out["status"] == "localized", **out}

    @app.delete("/api/results/{result_id}/plate")
    def clear_plate(result_id: str):
        with store.lock:
            get_record(result_id).plate = None
        return {"result_id": result_id, "plate": None}

    @app.put("/api/results/{result_id}/total-weight")
    def set_total_weight(result_id: str, body: TotalWeightIn):
        """One net weight for the whole plate. Never touches the per-region manual weights."""
        with store.lock:
            rec = get_record(result_id)
            rec.total_weight = {"net_weight_g": body.net_weight_g, "source": body.source, "simulated": True, "tare_g": body.tare_g,
                                "set_at": plate_session.now()}
            return {"result_id": result_id, "total_weight": rec.total_weight}

    @app.delete("/api/results/{result_id}/total-weight")
    def clear_total_weight(result_id: str):
        with store.lock:
            get_record(result_id).total_weight = None
        return {"result_id": result_id, "total_weight": None}

    @app.get("/api/results/{result_id}/plate-analysis")
    def plate_analysis(result_id: str):
        with store.lock:
            return analysis_for(get_record(result_id))

    @app.put("/api/results/{result_id}/regions/{region_id}/entity-count")
    def set_entity_count(result_id: str, region_id: int, body: EntityCountIn):
        with store.lock:
            region = get_region(result_id, region_id)
            region.user_entity_count = body.user_entity_count
            return {"result_id": result_id, **region.to_dict(nutri)}

    @app.put("/api/results/{result_id}/regions/{region_id}/excluded")
    def set_excluded(result_id: str, region_id: int, body: ExcludedIn):
        """Mark a region as not food (its area is then left out of the plate analysis)."""
        with store.lock:
            region = get_region(result_id, region_id)
            region.excluded = body.excluded
            return {"result_id": result_id, **region.to_dict(nutri)}

    @app.post("/api/results/{result_id}/regions/manual", status_code=201)
    def add_manual_region(result_id: str, body: ManualRegionIn):
        """Fallback for a food YOLO missed entirely: the user outlines it. It then goes through the same identity workflow."""
        with store.lock:
            rec = get_record(result_id)
            w, h = rec.image["width"], rec.image["height"]
            if any(not (0 <= x <= w and 0 <= y <= h) for x, y in body.polygon):
                raise ApiError(422, "invalid_polygon", f"Polygon points must lie inside the {w}x{h} image")
            if not plate_session.polygon_mask(body.polygon, h, w).any():
                raise ApiError(422, "invalid_polygon", "The outline encloses no pixels")
            rid = max((r.id for r in rec.regions), default=0) + 1
            region = RegionRecord(rid, 0, [], [], source="manual", manual_polygon=[list(p) for p in body.polygon])
            rec.regions.append(region)
            return {"result_id": result_id, **region.to_dict(nutri)}

    @app.delete("/api/results/{result_id}/regions/{region_id}")
    def delete_manual_region(result_id: str, region_id: int):
        with store.lock:
            rec = get_record(result_id)
            region = get_region(result_id, region_id)
            if region.source != "manual":
                raise ApiError(409, "not_manual_region", "Only user-drawn regions can be deleted; mark a detected region as excluded instead")
            rec.regions.remove(region)
            return {"result_id": result_id, "deleted_region_id": region_id}

    @app.post("/api/results/{result_id}/meal")
    def finalize_meal(result_id: str, body: FinalizeIn | None = None):
        """Save the current analysis as a meal. A complete meal needs allocation_complete AND a USDA record for every food."""
        accept = bool(body and body.accept_incomplete)
        with store.lock:
            rec = get_record(result_id)
            analysis = analysis_for(rec)
            if analysis["allocation_status"] in (None, "allocation_rejected"):
                raise ApiError(409, "cannot_finalize", f"The analysis is {analysis['allocation_status'] or 'missing inputs'}: "
                               + "; ".join(analysis["incomplete_reasons"]) + ". Nothing was saved.")
            if analysis["completeness_status"] != "complete" and not accept:
                raise ApiError(409, "incomplete_meal", "The meal is incomplete (" + "; ".join(analysis["incomplete_reasons"]) +
                               "). Resolve it, or save it explicitly as incomplete with accept_incomplete=true.")
            snap = plate_session.meal_snapshot(rec, analysis, nutri)
        meal, created = meals.save(snap)       # outside the store lock: only touches SQLite
        return JSONResponse({"meal": meal, "created": created, "duplicate": not created}, status_code=201 if created else 200)

    @app.get("/api/meals")
    def list_meals(limit: int = 100):
        return {"meals": meals.list(max(1, min(limit, 500)))}

    @app.get("/api/meals/{meal_id}")
    def get_meal(meal_id: str):
        meal = meals.get(meal_id)
        if meal is None:
            raise ApiError(404, "meal_not_found", f"No saved meal '{meal_id}'")
        return meal

    app.include_router(build_scale_router(coordinator, listener))
    app.include_router(build_camera_router(settings, camera, calibrations, detector, localizer))

    # ----------------------------------------------------------------- UI
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
