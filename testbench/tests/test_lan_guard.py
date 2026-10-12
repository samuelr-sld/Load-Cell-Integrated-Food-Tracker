"""LAN guard: the laptop reaches the whole dashboard; other devices on the Wi-Fi reach only the token-protected scan API."""
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from testbench.config import Settings
from testbench.main import create_app

TOKEN = "t" * 24
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def client(tmp_path, monkeypatch, peer, server="192.168.1.8"):
    monkeypatch.setenv("PLATEGO_AI_TOKEN", TOKEN)
    s = replace(Settings(), host="0.0.0.0", meals_db=tmp_path / "m.sqlite", camera_state_file=tmp_path / "c.json")
    return TestClient(create_app(settings=s, pipeline=object()), base_url=f"http://{server}:8000", client=(peer, 50000))


@pytest.mark.parametrize("peer", ["127.0.0.1", "192.168.1.8"])   # localhost, and the laptop using its own Wi-Fi address
def test_laptop_gets_dashboard_assets_and_api(tmp_path, monkeypatch, peer):
    c = client(tmp_path, monkeypatch, peer)
    assert c.get("/").status_code == 200 and "<html" in c.get("/").text.lower()
    assert c.get("/api/camera/status").status_code == 200
    assert c.get("/api/scale/status").status_code == 200           # no token needed locally
    for path in ("/static/app.js", "/static/camera.js", "/static/plate.js", "/static/autoscan.js", "/static/style.css"):
        assert c.get(path).status_code == 200, path

def test_other_device_only_reaches_scan_api_with_token(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch, "192.168.1.50")
    assert c.get("/").status_code == 403
    assert c.get("/api/camera/status").status_code == 403
    assert c.post("/api/camera/capture").status_code == 403
    assert c.get("/api/scans/latest").status_code == 401
    assert c.get("/api/scans/latest", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/api/scans/latest", headers=AUTH).json() == {"scan": None}
    assert c.get("/api/scale/status", headers=AUTH).status_code == 200
