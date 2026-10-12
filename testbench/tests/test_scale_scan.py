"""Scale -> scan: simulated serial lines drive the real FastAPI app (fake camera + fake networks from test_camera_api)."""
import threading
import time

import pytest

from testbench.scale_scan import ScanCoordinator, SerialListener, parse_line
from .test_camera_api import make


class FakeSerial:
    """readline() yields the scripted lines, then raises like an unplugged port."""
    def __init__(self, lines):
        self.lines, self.closed = list(lines), False

    def readline(self):
        if not self.lines:
            raise OSError("device disconnected")
        return self.lines.pop(0).encode()

    def close(self):
        self.closed = True


def feed(coord, lines):
    """Run the lines through a real SerialListener (scripted port) into the coordinator, then wait for the scan to finish."""
    ports = [FakeSerial(lines)]
    def opener(port, baud):
        if not ports:
            raise PermissionError("COM3 busy")
        return ports.pop(0)
    ls = SerialListener("COM3", 115200, coord.handle, opener=opener, retry_s=0.01)
    ls.start()
    deadline = time.time() + 5
    while "busy" not in (ls.last_error or "") and time.time() < deadline:   # disconnect -> reopen -> port busy
        time.sleep(0.01)
    ls.stop()
    coord.join()
    return ls


@pytest.fixture
def app_client(tmp_path):
    c, _, _ = make(tmp_path)
    yield c
    c.__exit__(None, None, None)


def test_parse_line():
    assert parse_line("WEIGHT_STABLE:152.3\r\n") == ("stable", 152.3)
    assert parse_line("SCALE_EMPTY\n") == ("empty", None)
    for bad in ["# state=STABLE weight=1 g", "", "WEIGHT_STABLE:", "WEIGHT_STABLE:abc", "WEIGHT_STABLE:-5", "WEIGHT_STABLE:nan", "garbage\x00"]:
        assert parse_line(bad) is None


def test_stable_triggers_one_scan_and_diagnostics_are_ignored(app_client):
    coord = app_client.app.state.scale
    ls = feed(coord, ["# PlateGo load cell\n", "# Tared. Ready\n", "WEIGHT_STABLE:250.0\n", "# state=STABLE\n", "WEIGHT_STABLE:251.0\n", "noise\n"])
    assert ls.firmware_ready and ls.lines_ignored == 4
    assert ls.last_error == "PermissionError: COM3 busy"                   # reconnect attempt hit a busy port: reported, not fatal
    assert len(coord._scans) == 1                                          # duplicate stable (same placement) did not rescan
    assert coord.last_ignored["reason"] == "duplicate_placement"
    scan = app_client.get("/api/scans/latest").json()["scan"]
    assert scan["status"] == "ready" and scan["measured_total_grams"] == 250.0 and scan["net_grams"] == 250.0
    assert scan["result_id"] and len(scan["foods"]) == 3
    assert scan["foods"][0]["candidates"] and scan["suggested_grams"] is None      # several foods: no invented per-food weights
    assert app_client.get(f"/api/scans/{scan['scan_id']}").json()["scan_id"] == scan["scan_id"]
    assert app_client.get("/api/scale/status").json()["armed"] is False


def test_empty_rearms_without_deleting_pending_scan(app_client):
    coord = app_client.app.state.scale
    coord.handle("stable", 100.0)
    coord.join()
    coord.handle("empty", None)
    coord.handle("stable", 120.0)
    coord.join()
    assert len(coord._scans) == 2
    first = list(coord._scans.values())[0]
    assert first["status"] == "ready" and first["acknowledged_at"] is None
    assert app_client.get("/api/scans/latest").json()["scan"]["measured_total_grams"] == 120.0


def test_acknowledge_is_idempotent_and_hides_scan(app_client):
    coord = app_client.app.state.scale
    feed(coord, ["WEIGHT_STABLE:100.0\n"])
    sid = app_client.get("/api/scans/latest").json()["scan"]["scan_id"]
    r1 = app_client.post(f"/api/scans/{sid}/acknowledge", json={"outcome": "saved"}).json()
    r2 = app_client.post(f"/api/scans/{sid}/acknowledge", json={"outcome": "rejected"}).json()
    assert r1["acknowledged_at"] == r2["acknowledged_at"] and r2["outcome"] == "saved"
    assert app_client.get("/api/scans/latest").json() == {"scan": None}
    assert app_client.post("/api/scans/nope/acknowledge", json={"outcome": "saved"}).status_code == 404
    assert app_client.get("/api/scans/nope").status_code == 404


def test_invalid_weight_is_ignored_and_tare_is_subtracted():
    seen = []
    coord = ScanCoordinator(lambda s: seen.append(s["net_grams"]) or {"foods": []}, tare_g=300.0, min_net_g=5.0)
    coord.handle("stable", 302.0)                                          # net 2 g: noise
    assert coord.last_ignored["reason"] == "invalid_weight" and not coord._scans
    coord.handle("stable", 450.0)
    coord.join()
    assert seen == [150.0]


def test_failed_inference_keeps_error_and_next_placement_works():
    calls = []
    def run(scan):
        calls.append(scan["scan_id"])
        if len(calls) == 1:
            raise RuntimeError("camera unplugged")
        return {"foods": [], "needs_manual_food": True}
    coord = ScanCoordinator(run)
    coord.handle("stable", 200.0)
    coord.join()
    first = coord.latest()
    assert first["status"] == "error" and "camera unplugged" in first["error"]["message"]
    coord.handle("empty", None)
    coord.handle("stable", 200.0)
    coord.join()
    assert coord.latest()["status"] == "ready" and len(calls) == 2


def test_overlapping_scan_is_refused():
    gate, started = threading.Event(), threading.Event()
    def run(scan):
        started.set()
        gate.wait(5)
        return {}
    coord = ScanCoordinator(run)
    coord.handle("stable", 200.0)
    started.wait(5)
    coord.handle("empty", None)
    coord.handle("stable", 300.0)                                          # new placement while the first is still processing
    assert coord.last_ignored["reason"] == "scan_in_progress" and len(coord._scans) == 1
    gate.set()
    coord.join()
    assert coord.latest()["status"] == "ready"


def test_serial_unavailable_reports_error_and_retries():
    n = []
    def opener(port, baud):
        n.append(1)
        raise FileNotFoundError("could not open port 'COM3'")
    ls = SerialListener("COM3", 115200, lambda *a: None, opener=opener, retry_s=0.01)
    ls.start()
    time.sleep(0.2)
    ls.stop()
    assert len(n) > 2 and ls.status()["state"] == "stopped" and "FileNotFoundError" in ls.last_error


def test_dashboard_discovery_is_read_only_and_keeps_the_mobile_contract(app_client):
    from .conftest import jpeg_bytes
    from .synthetic_plate import plate_scene
    coord = app_client.app.state.scale
    pipe = app_client.app.state.pipeline
    feed(coord, ["WEIGHT_STABLE:100.0\n"])
    runs = pipe.segmenter.calls
    assert runs == 1
    scan = app_client.get("/api/scans/latest").json()["scan"]
    assert scan["camera_id"] == "index:1" and scan["result_id"]
    assert set(scan) >= {"scan_id", "timestamp", "measured_total_grams", "foods", "status", "error"}   # fields the phone reads
    for _ in range(5):                                                       # dashboard polling
        app_client.get("/api/scans/latest?include_acknowledged=true")
        app_client.get(f"/api/results/{scan['result_id']}")
    assert pipe.segmenter.calls == runs                                      # no extra inference
    app_client.post(f"/api/scans/{scan['scan_id']}/acknowledge", json={"outcome": "saved"})
    assert app_client.get("/api/scans/latest").json() == {"scan": None}      # the phone no longer sees it
    assert app_client.get("/api/scans/latest?include_acknowledged=true").json()["scan"]["scan_id"] == scan["scan_id"]   # the dashboard still does
    img = app_client.get(f"/api/results/{scan['result_id']}/images/original")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"     # the captured frame
    r = app_client.post("/api/inference", files={"files": ("m.jpg", jpeg_bytes(plate_scene()), "image/jpeg")})   # manual Analyze unaffected
    assert r.status_code == 200 and pipe.segmenter.calls == runs + 1


def test_ready_scan_result_survives_clear_all_and_eviction_until_acknowledged(tmp_path):
    from dataclasses import replace
    from .conftest import jpeg_bytes
    from .synthetic_plate import plate_scene
    c, _, _ = make(tmp_path, max_results=2)                                   # tiny store: eviction is easy to provoke
    try:
        coord = c.app.state.scale
        feed(coord, ["WEIGHT_STABLE:67.7\n"])
        scan = c.get("/api/scans/latest").json()["scan"]
        rid = scan["result_id"]
        assert c.get(f"/api/results/{rid}").status_code == 200
        assert c.get(f"/api/results/{rid}/images/original").headers["content-type"] == "image/jpeg"
        assert c.delete("/api/results").json() == {"cleared": 0}              # "Clear all results" on the dashboard
        for _ in range(4):                                                    # manual analyses overflow max_results=2
            c.post("/api/inference", files={"files": ("m.jpg", jpeg_bytes(plate_scene()), "image/jpeg")})
        assert c.get(f"/api/results/{rid}").status_code == 200                # still retrievable for the pending scan
        assert c.get("/api/results").json()["store"]["count"] <= 3
        c.post(f"/api/scans/{scan['scan_id']}/acknowledge", json={"outcome": "saved"})
        assert c.delete("/api/results").json()["cleared"] >= 1               # once acknowledged it is ordinary again
        assert c.get(f"/api/results/{rid}").status_code == 404
    finally:
        c.__exit__(None, None, None)
