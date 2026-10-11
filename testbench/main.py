"""FastAPI app: local testbench API plus the static debugging UI. Model-specific logic lives in the other modules."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings
from .evaluation import FeedbackError, build_summary, make_feedback
from .ingest import ImageValidationError, check_filename
from .model_loader import ModelLoadError, load_models
from .result_store import ResultStore, ResultTooLarge

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


def create_app(settings: Settings | None = None, pipeline=None) -> FastAPI:
    """`pipeline` lets tests inject a fake; when omitted the real checkpoints are loaded once at startup."""
    settings = settings or Settings()
    store = ResultStore(settings.max_results, settings.max_store_bytes)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
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

    app = FastAPI(title="Food Recognition Inference Testbench", version="0.1.0", lifespan=lifespan,
                  docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
    app.state.settings, app.state.store = settings, store

    # ----------------------------------------------------------------- errors
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return JSONResponse(_error_body(exc.code, exc.message), status_code=exc.status)

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
    async def inference(files: list[UploadFile] = File(...)):
        pipe = require_pipeline()
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
                record = await run_in_threadpool(pipe.process_upload, data, name)   # blocking ML work off the event loop
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
            items.append({"filename": name, "status": "completed", "result_id": record.result_id, "error": None, "result": record.to_detail()})

        counts = {"received": len(items), **{s: sum(1 for i in items if i["status"] == s) for s in ("completed", "failed", "rejected")}}
        status = 200 if counts["completed"] else (items[0]["error"]["http_status"] if items else 422)
        return JSONResponse({"items": items, "summary": counts}, status_code=status)

    @app.get("/api/results")
    def list_results():
        return {"results": [r.to_summary() for r in store.list()], "store": store.stats()}

    @app.get("/api/results/{result_id}")
    def result_detail(result_id: str):
        return get_record(result_id).to_detail()

    @app.get("/api/results/{result_id}/images/{image_key}")
    def result_image(result_id: str, image_key: str):
        rec = get_record(result_id)
        blob = rec.images.get(image_key)
        if blob is None:
            raise ApiError(404, "image_not_found", f"Result '{result_id}' has no image '{image_key}'")
        return Response(blob, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=600"})

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
            try:
                seg.feedback = make_feedback(seg, body.verdict, body.actual_label, pipe.class_names)
            except FeedbackError as exc:
                raise ApiError(422, "invalid_feedback", str(exc)) from None
            return {"result_id": result_id, "segment_id": segment_id, "feedback": seg.feedback.to_dict()}

    @app.delete("/api/results/{result_id}/segments/{segment_id}/feedback")
    def delete_feedback(result_id: str, segment_id: int):
        with store.lock:
            get_segment(result_id, segment_id).feedback = None
        return {"result_id": result_id, "segment_id": segment_id, "feedback": None}

    @app.get("/api/evaluation/summary")
    def evaluation_summary():
        with store.lock:
            return build_summary(store.list(), store.evicted_total)

    # ----------------------------------------------------------------- UI
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
