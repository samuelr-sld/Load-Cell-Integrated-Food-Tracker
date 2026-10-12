"""PlateGo local AI gateway: a small authenticated HTTP API between the phone and Ollama.

The phone never talks to Ollama. This gateway exposes exactly two calls (health, chat), pins the model,
the options and the Ollama address (loopback only), and validates every request. It never sees the
phone's database: the app runs the nutrition tools itself and sends only bounded tool results.
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

log = logging.getLogger("ai_gateway")

MAX_BODY_BYTES = 64 * 1024
TOKEN_FILE = Path(__file__).with_name(".pairing_token")


def load_token() -> str:
    """PLATEGO_AI_TOKEN, else a random token created once and kept in ai_gateway/.pairing_token."""
    env = os.environ.get("PLATEGO_AI_TOKEN", "").strip()
    if env:
        return env
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text(encoding="utf-8").strip()
    token = secrets.token_urlsafe(18)
    TOKEN_FILE.write_text(token, encoding="utf-8")
    return token


@dataclass
class Settings:
    token: str = field(default_factory=load_token)
    model: str = field(default_factory=lambda: os.environ.get("PLATEGO_AI_MODEL", "qwen2.5-coder:3b"))
    ollama_url: str = field(default_factory=lambda: os.environ.get("PLATEGO_OLLAMA_URL", "http://127.0.0.1:11434"))
    timeout_s: float = field(default_factory=lambda: float(os.environ.get("PLATEGO_AI_TIMEOUT", "90")))
    max_concurrent: int = field(default_factory=lambda: int(os.environ.get("PLATEGO_AI_CONCURRENCY", "1")))
    num_ctx: int = 4096
    keep_alive: str = "30m"

    def __post_init__(self):
        if urlparse(self.ollama_url).hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("PLATEGO_OLLAMA_URL must be a loopback address; Ollama is never reached over the network")
        if len(self.token) < 16:
            raise ValueError("the pairing token must be at least 16 characters")


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(max_length=16000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requestId: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    messages: list[Message] = Field(min_length=1, max_length=32)
    format: Optional[Literal["json"]] = None


def error(status: int, code: str, message: str, **extra) -> JSONResponse:
    return JSONResponse({"error": code, "message": message, **extra}, status_code=status)


def create_app(settings: Optional[Settings] = None, transport: Optional[httpx.AsyncBaseTransport] = None) -> FastAPI:
    s = settings or Settings()
    app = FastAPI(title="PlateGo AI gateway", docs_url=None, redoc_url=None, openapi_url=None)
    slots = asyncio.Semaphore(s.max_concurrent)
    app.state.settings, app.state.slots = s, slots

    def ollama() -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=s.ollama_url, transport=transport, timeout=httpx.Timeout(s.timeout_s, connect=3.0))

    def authorized(request: Request) -> bool:
        got = request.headers.get("authorization", "")
        return hmac.compare_digest(got.encode(), f"Bearer {s.token}".encode())

    @app.get("/v1/health")
    async def health(request: Request):
        if not authorized(request):
            return error(401, "unauthorized", "Missing or wrong pairing token.")
        try:
            async with ollama() as client:
                r = await client.get("/api/tags", timeout=3.0)
                r.raise_for_status()
                names = {m.get("name") for m in r.json().get("models", [])} | {m.get("model") for m in r.json().get("models", [])}
        except (httpx.HTTPError, ValueError):
            return {"gateway": "ok", "ollama": "unavailable", "model": s.model, "modelInstalled": False, "busy": slots.locked()}
        return {"gateway": "ok", "ollama": "ok", "model": s.model, "modelInstalled": s.model in names, "busy": slots.locked()}

    @app.post("/v1/chat")
    async def chat(request: Request):
        if not authorized(request):
            return error(401, "unauthorized", "Missing or wrong pairing token.")
        if int(request.headers.get("content-length") or 0) > MAX_BODY_BYTES:
            return error(413, "too_large", f"Requests are limited to {MAX_BODY_BYTES} bytes.")
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return error(413, "too_large", f"Requests are limited to {MAX_BODY_BYTES} bytes.")
        try:
            req = ChatRequest.model_validate_json(body)
        except ValidationError as e:
            return error(422, "invalid_request", "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:5]))
        if slots.locked():
            return error(429, "busy", "The laptop AI is answering another request.")

        async with slots:
            payload = {
                # think=False: thinking-capable models (e.g. granite4.2) otherwise spend hundreds of tokens per reply.
                "model": s.model, "stream": False, "think": False, "keep_alive": s.keep_alive,
                "options": {"temperature": 0.2, "num_ctx": s.num_ctx},
                "messages": [m.model_dump() for m in req.messages],
                **({"format": "json"} if req.format else {}),
            }
            started = time.monotonic()

            async def call():
                async with ollama() as client:
                    return await client.post("/api/chat", json=payload)

            task = asyncio.create_task(call())
            # Closing the Ollama connection aborts generation, so a phone that cancels or drops frees the GPU.
            while not task.done():
                await asyncio.wait({task}, timeout=0.5)
                if not task.done() and await request.is_disconnected():
                    task.cancel()
                    log.info("request %s: client disconnected, generation cancelled", req.requestId)
                    return error(499, "cancelled", "Client disconnected.")
            try:
                r = task.result()
            except httpx.TimeoutException:
                return error(504, "timeout", f"The model did not answer within {s.timeout_s:.0f} s.")
            except httpx.HTTPError:
                return error(503, "ollama_unavailable", "Ollama is not running on the laptop.")

            if r.status_code == 404:
                return error(503, "model_missing", f"Model {s.model} is not installed. Run: ollama pull {s.model}", model=s.model)
            if r.status_code != 200:
                log.warning("request %s: ollama returned %s: %s", req.requestId, r.status_code, r.text[:200])
                return error(502, "ollama_error", "Ollama could not answer this request.")
            data = r.json()
            content = (data.get("message") or {}).get("content")
            if not isinstance(content, str):
                return error(502, "ollama_error", "Ollama returned no message.")
            ms = lambda k: round((data.get(k) or 0) / 1e6)  # noqa: E731  (Ollama reports nanoseconds)
            log.info("request %s: %d ms, %s tokens", req.requestId, (time.monotonic() - started) * 1000, data.get("eval_count"))
            return {
                "requestId": req.requestId, "model": s.model, "content": content,
                "timings": {"totalMs": ms("total_duration"), "loadMs": ms("load_duration"),
                            "promptTokens": data.get("prompt_eval_count"), "outputTokens": data.get("eval_count")},
            }

    return app
