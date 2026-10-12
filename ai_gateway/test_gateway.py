"""Gateway contract tests with a fake Ollama (httpx.MockTransport): auth, validation, limits, error mapping."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from ai_gateway.app import MAX_BODY_BYTES, Settings, create_app

TOKEN = "test-token-0123456789"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
BODY = {"requestId": "r-1", "messages": [{"role": "user", "content": "hi"}], "format": "json"}


def client(handler):
    return TestClient(create_app(Settings(token=TOKEN, model="m:1b"), transport=httpx.MockTransport(handler)))


def ok_chat(request: httpx.Request):
    if request.url.path == "/api/tags":
        return httpx.Response(200, json={"models": [{"name": "m:1b", "model": "m:1b"}]})
    sent = json.loads(request.content)
    return httpx.Response(200, json={"message": {"role": "assistant", "content": json.dumps(sent)},
                                     "total_duration": 2_000_000, "load_duration": 1_000_000, "eval_count": 3})


def test_chat_forwards_only_pinned_model_and_options():
    r = client(ok_chat).post("/v1/chat", json=BODY, headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    sent = json.loads(data["content"])
    assert sent["model"] == "m:1b" and sent["stream"] is False and sent["format"] == "json" and sent["think"] is False
    assert sent["messages"] == BODY["messages"]
    assert data["requestId"] == "r-1" and data["timings"]["totalMs"] == 2


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong-token-000000000"}, {"Authorization": TOKEN}])
def test_pairing_token_is_required(headers):
    c = client(ok_chat)
    assert c.post("/v1/chat", json=BODY, headers=headers).status_code == 401
    assert c.get("/v1/health", headers=headers).status_code == 401


@pytest.mark.parametrize("body", [
    {**BODY, "model": "other"},                                         # cannot pick a model
    {**BODY, "options": {"num_ctx": 999999}},                           # cannot pass options
    {**BODY, "messages": []},
    {**BODY, "messages": [{"role": "root", "content": "x"}]},
    {**BODY, "messages": [{"role": "user", "content": "x", "images": ["..."]}]},
    {**BODY, "messages": [{"role": "user", "content": "x"}] * 33},
    {**BODY, "requestId": "../etc"},
    {**BODY, "format": "yaml"},
])
def test_invalid_requests_are_rejected(body):
    assert client(ok_chat).post("/v1/chat", json=body, headers=AUTH).status_code == 422


def test_oversized_payload_is_rejected():
    big = {**BODY, "messages": [{"role": "user", "content": "x" * 15000}] * 5}
    assert len(json.dumps(big)) > MAX_BODY_BYTES
    assert client(ok_chat).post("/v1/chat", json=big, headers=AUTH).status_code == 413


def test_busy_when_all_slots_are_in_use():
    c = client(ok_chat)
    c.app.state.slots._value = 0  # simulate one request in flight
    r = c.post("/v1/chat", json=BODY, headers=AUTH)
    assert r.status_code == 429 and r.json()["error"] == "busy"


@pytest.mark.parametrize("handler,status,code", [
    (lambda req: (_ for _ in ()).throw(httpx.ConnectError("refused")), 503, "ollama_unavailable"),
    (lambda req: (_ for _ in ()).throw(httpx.ReadTimeout("slow")), 504, "timeout"),
    (lambda req: httpx.Response(404, json={"error": "model 'm:1b' not found"}), 503, "model_missing"),
    (lambda req: httpx.Response(500, text="boom"), 502, "ollama_error"),
    (lambda req: httpx.Response(200, json={"message": {}}), 502, "ollama_error"),
])
def test_ollama_failures_map_to_structured_errors(handler, status, code):
    r = client(handler).post("/v1/chat", json=BODY, headers=AUTH)
    assert r.status_code == status and r.json()["error"] == code


def test_health_reports_ollama_and_model_state():
    assert client(ok_chat).get("/v1/health", headers=AUTH).json() == {
        "gateway": "ok", "ollama": "ok", "model": "m:1b", "modelInstalled": True, "busy": False}
    other = client(lambda req: httpx.Response(200, json={"models": [{"name": "x:7b"}]}))
    assert other.get("/v1/health", headers=AUTH).json()["modelInstalled"] is False
    down = client(lambda req: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    assert down.get("/v1/health", headers=AUTH).json()["ollama"] == "unavailable"


def test_only_two_routes_are_exposed():
    c = client(ok_chat)
    for path in ["/api/chat", "/api/pull", "/api/tags", "/docs", "/openapi.json"]:
        assert c.post(path, headers=AUTH).status_code in (404, 405)


def test_ollama_must_be_loopback():
    with pytest.raises(ValueError):
        Settings(token=TOKEN, ollama_url="http://192.168.1.5:11434")
