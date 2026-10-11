"""Live preview: camera ownership, stream lifecycle, capture during preview and error recovery.
Fake cameras only (no real device). The MJPEG stream tests run the real app under uvicorn because Starlette's TestClient buffers
whole responses and so cannot consume an endless stream."""
import socket
import threading
import time
from dataclasses import replace

import cv2
import httpx
import numpy as np
import pytest
import torch
import uvicorn

from testbench.camera import CameraError, CameraService
from testbench.config import Settings
from testbench.crops import CropProcessor
from testbench.main import create_app
from testbench.orchestrator import InferencePipeline
from .conftest import FakeClassifier, FakeSegmenter, detection, rect_mask
from .synthetic_plate import plate_scene


def wait_until(pred, timeout=3.0, step=0.02):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(step)
    return pred()


class Rig:
    """Fake USB cameras + bookkeeping: how many opens, how many handles are open AT ONCE, which were released."""
    def __init__(self, per_source=None):
        self.cfg, self.opens, self.live, self.max_live, self.caps = dict(per_source or {}), [], 0, 0, []
        self.lock = threading.Lock()

    def __call__(self, source, flag):
        cfg = dict(self.cfg.get(source, {"opened": False}))
        cap = LiveCap(self, source, **cfg)
        with self.lock:
            self.opens.append(source)
            self.caps.append(cap)
            if cap.opened:
                self.live += 1
                self.max_live = max(self.max_live, self.live)
        return cap

    def opens_of(self, source):
        return self.opens.count(source)

    def all_released(self):
        return all(c.released or not c.opened for c in self.caps)


class LiveCap:
    def __init__(self, rig, source, opened=True, w=640, h=360, fps=60, die_after=None, hang=None, base=None):
        self.rig, self.source, self.opened, self.w, self.h = rig, source, opened, w, h
        self.period, self.die_after, self.hang, self.base = 1.0 / fps, die_after, hang, base
        self.released, self.grabs, self.props = False, 0, {}

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def grab(self):
        if self.hang is not None:
            self.hang.wait(5)
        time.sleep(self.period)
        self.grabs += 1
        return not (self.die_after is not None and self.grabs > self.die_after)

    def retrieve(self):
        if self.base is not None:
            fr = self.base.copy()
            fr[0, 0] = (self.grabs % 250, 0, 0)               # identifies which frame this is
            return True, fr
        return True, np.full((self.h, self.w, 3), self.grabs % 250, np.uint8)

    def read(self):
        return self.grab() and self.retrieve() or (False, None)

    def release(self):
        if not self.released:
            self.released = True
            if self.opened:
                with self.rig.lock:
                    self.rig.live -= 1


def service(rig, idle=0.4, timeout=2.0, fps=30, width=0, height=0, preview_width=320):
    return CameraService("any", width, height, warmup_frames=0, timeout_s=timeout, capture_factory=rig, preview_fps=fps,
                         preview_width=preview_width, preview_idle_timeout_s=idle)


# ============================================================ camera ownership (service level)
def test_preview_opens_the_camera_once_and_stop_releases_it():
    rig = Rig({1: {}})
    svc = service(rig)
    st = svc.start_preview(1)
    assert st["active"] and st["state"] == "running" and (st["width"], st["height"]) == (640, 360) and st["camera_id"] == "index:1"
    assert rig.opens == [1] and rig.live == 1
    assert wait_until(lambda: svc.preview_status()["frames"] >= 3) and svc.preview_status()["measured_fps"] > 0
    stopped = svc.stop_preview()
    assert not stopped["active"] and stopped["state"] == "stopped" and rig.live == 0 and rig.all_released()
    assert svc.stop_preview()["state"] == "stopped"                           # idempotent


def test_start_is_idempotent_for_the_same_camera_and_never_opens_a_second_handle():
    rig = Rig({1: {}})
    svc = service(rig)
    svc.start_preview(1)
    svc.start_preview("1")
    svc.start_preview(1)
    assert rig.opens == [1] and rig.max_live == 1
    svc.stop_preview()


def test_starting_another_camera_releases_the_first_before_opening_the_second():
    rig = Rig({1: {}, 2: {}})
    svc = service(rig)
    svc.start_preview(1)
    st = svc.start_preview(2)
    assert st["camera_id"] == "index:2" and rig.opens == [1, 2] and rig.caps[0].released and rig.max_live == 1
    svc.stop_preview()
    assert rig.live == 0


def test_capture_during_preview_reuses_the_open_handle_and_returns_a_fresh_full_resolution_frame():
    rig = Rig({1: {"w": 1920, "h": 1080}})
    svc = service(rig, preview_width=320)
    svc.start_preview(1)
    a = svc.capture(1)
    time.sleep(0.15)
    b = svc.capture("1")
    assert a.from_preview and b.from_preview and (a.width, a.height) == (1920, 1080)       # full camera resolution, not the 320 px stream copy
    assert int(b.bgr[0, 0, 0]) != int(a.bgr[0, 0, 0])                                       # two captures are two different, new frames
    assert rig.opens == [1] and rig.max_live == 1                                           # no second VideoCapture was ever opened
    assert a.info()["from_preview"] is True
    a.bgr[:] = 0                                                                            # a returned frame is a copy: it cannot corrupt the preview
    assert int(svc.capture(1).bgr.max()) > 0
    svc.stop_preview()


def test_concurrent_captures_during_preview_never_raise_camera_busy():
    rig = Rig({1: {}})
    svc = service(rig)
    svc.start_preview(1)
    errors, frames = [], []

    def grab():
        try:
            frames.append(svc.capture(1))
        except CameraError as exc:
            errors.append(exc.code)
    ts = [threading.Thread(target=grab) for _ in range(8)]
    [t.start() for t in ts]
    [t.join(5) for t in ts]
    assert errors == [] and len(frames) == 8 and rig.opens == [1]
    svc.stop_preview()


def test_capture_from_a_different_camera_while_previewing_is_refused_without_opening_it():
    rig = Rig({1: {}, 2: {}})
    svc = service(rig)
    svc.start_preview(1)
    with pytest.raises(CameraError) as e:
        svc.capture(2)
    assert e.value.code == "camera_in_use" and e.value.http_status == 409 and rig.opens == [1]
    svc.stop_preview()


def test_probe_of_the_previewed_camera_reports_it_without_competing_for_the_device():
    rig = Rig({1: {"w": 1920, "h": 1080}, 2: {}})
    svc = service(rig)
    svc.start_preview(1)
    found = svc.enumerate(3)
    by = {d["camera_id"]: d for d in found}
    assert by["index:1"]["in_preview"] and by["index:1"]["capture_ok"] and (by["index:1"]["width"], by["index:1"]["height"]) == (1920, 1080)
    assert by["index:2"]["capture_ok"] and not by["index:0"]["capture_ok"]                 # other cameras are still probed normally
    assert rig.opens_of(1) == 1
    svc.stop_preview()


def test_one_shot_capture_works_again_after_the_preview_stops():
    rig = Rig({1: {}})
    svc = service(rig)
    svc.start_preview(1)
    svc.stop_preview()
    fr = svc.capture(1)
    assert not fr.from_preview and rig.opens == [1, 1] and rig.live == 0 and rig.max_live == 1


def test_requested_capture_size_is_applied_to_the_preview_handle():
    rig = Rig({1: {}})
    svc = service(rig, width=1920, height=1080)
    st = svc.start_preview(1)
    assert st["requested_size"] == [1920, 1080]
    assert rig.caps[0].props[cv2.CAP_PROP_FRAME_WIDTH] == 1920 and rig.caps[0].props[cv2.CAP_PROP_FRAME_HEIGHT] == 1080
    svc.stop_preview()


# ============================================================ release rules + error recovery
def test_unavailable_camera_fails_cleanly_and_leaves_nothing_owned():
    rig = Rig({})
    svc = service(rig)
    with pytest.raises(CameraError) as e:
        svc.start_preview(3)
    assert e.value.code == "camera_unavailable" and svc.active_preview() is None and rig.live == 0
    rig.cfg[1] = {}
    assert svc.capture(1).width == 640                                                     # the single-flight lock was handed back


def test_invalid_camera_is_rejected_before_anything_is_opened():
    rig = Rig()
    with pytest.raises(CameraError) as e:
        service(rig).start_preview(999)
    assert e.value.code == "invalid_camera" and rig.opens == []


def test_camera_unplugged_mid_preview_is_reported_released_and_recoverable():
    rig = Rig({1: {"die_after": 15}})
    svc = service(rig, timeout=2.0, idle=30)
    svc.start_preview(1)
    assert wait_until(lambda: svc.preview_status()["state"] == "error", timeout=5)
    st = svc.preview_status()
    assert not st["active"] and st["error"]["code"] == "capture_failed" and "stopped delivering" in st["error"]["message"]
    assert rig.live == 0 and rig.caps[0].released                                          # released even though it died
    assert svc.active_preview() is None and not svc.capture(1).from_preview                 # a dead preview no longer owns the camera: normal one-shot capture
    rig.cfg[1] = {}                                                                        # the camera is plugged back in
    assert svc.start_preview(1)["state"] == "running"
    svc.stop_preview()


def test_capture_during_preview_fails_fast_if_the_camera_stalls():
    gate = threading.Event()
    rig = Rig({1: {}})
    svc = service(rig, timeout=0.5)
    svc.start_preview(1)
    rig.caps[0].hang = gate                                                                # the driver now blocks inside grab()
    t0 = time.perf_counter()
    with pytest.raises(CameraError) as e:
        svc.capture(1)
    assert e.value.code == "camera_timeout" and time.perf_counter() - t0 < 2.0
    st = svc.preview_status()
    assert st["state"] == "stalled" or wait_until(lambda: svc.preview_status()["state"] == "stalled", 2)
    gate.set()
    svc.stop_preview()
    assert wait_until(lambda: rig.live == 0)


def test_hung_open_returns_504_and_the_preview_is_not_started():
    gate = threading.Event()
    rig = Rig({1: {"hang": gate}})
    svc = service(rig, timeout=0.4)
    t0 = time.perf_counter()
    with pytest.raises(CameraError) as e:
        svc.start_preview(1)
    assert e.value.code == "camera_timeout" and time.perf_counter() - t0 < 2.0
    with pytest.raises(CameraError) as e2:                                                 # the stuck open still owns the device lock
        svc.capture(1)
    assert e2.value.code in ("camera_busy", "camera_in_use")
    gate.set()
    assert wait_until(lambda: rig.live == 0 and rig.all_released())                         # the worker releases once the driver returns


def test_preview_is_released_automatically_when_no_browser_is_watching():
    rig = Rig({1: {}})
    svc = service(rig, idle=0.3)
    svc.start_preview(1)                                                                   # nobody ever connects a stream
    assert wait_until(lambda: rig.live == 0, timeout=3)
    st = svc.preview_status()
    assert not st["active"] and st["stop_reason"] == "no_viewers"


def test_a_connected_viewer_keeps_the_preview_alive_and_leaving_releases_it():
    rig = Rig({1: {}})
    svc = service(rig, idle=0.3)
    sess = (svc.start_preview(1), svc.active_preview())[1]
    sess.add_viewer()
    time.sleep(0.8)                                                                        # well past the idle timeout
    assert svc.active_preview() is sess and rig.live == 1
    sess.remove_viewer()
    assert wait_until(lambda: rig.live == 0, timeout=3)


def test_service_shutdown_releases_the_camera():
    rig = Rig({1: {}})
    svc = service(rig, idle=30)
    svc.start_preview(1)
    svc.shutdown()
    assert rig.live == 0 and svc.active_preview() is None


def test_start_stop_start_cycles_do_not_leak_handles_or_the_lock():
    rig = Rig({1: {}})
    svc = service(rig, idle=30)
    for _ in range(4):
        svc.start_preview(1)
        svc.stop_preview()
    assert rig.live == 0 and rig.max_live == 1 and rig.opens == [1] * 4
    assert svc.capture(1).width == 640


def test_encoding_work_only_happens_while_someone_is_watching():
    rig = Rig({1: {}})
    svc = service(rig, idle=30)
    svc.start_preview(1)
    sess = svc.active_preview()
    assert wait_until(lambda: sess.frames_published >= 5)
    assert sess.latest_jpeg() == (0, None)                                                 # frames decoded for capture, but no JPEG encoded for nobody
    sess.add_viewer()
    assert wait_until(lambda: sess.latest_jpeg()[1] is not None)
    seq, jpeg = sess.latest_jpeg()
    assert jpeg[:2] == b"\xff\xd8"
    svc.stop_preview()


# ============================================================ HTTP: stream lifecycle (real server)
class LiveServer:
    def __init__(self, app):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error", lifespan="on"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        assert wait_until(lambda: self.server.started, timeout=10)
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{self.port}", timeout=10)

    def close(self):
        self.client.close()
        self.server.should_exit = True
        self.thread.join(10)


def _chunks(resp):
    it = getattr(resp, "_test_iter", None)
    if it is None:
        it = resp._test_iter = resp.iter_bytes()                    # httpx allows one iteration per response: keep it
    return it


def drain(resp):
    for _ in _chunks(resp):
        pass


def read_frames(resp, n, timeout=5.0):
    """Parse n JPEGs out of a multipart/x-mixed-replace stream."""
    buf, frames, end = b"", [], time.time() + timeout
    for chunk in _chunks(resp):
        buf += chunk
        while True:
            i = buf.find(b"--frame\r\n")
            j = buf.find(b"\r\n\r\n", i) if i >= 0 else -1
            if j < 0:
                break
            head = buf[i:j].decode()
            length = int(head.split("Content-Length:")[1].split()[0])
            if len(buf) < j + 4 + length:
                break
            frames.append(buf[j + 4:j + 4 + length])
            buf = buf[j + 4 + length:]
        if len(frames) >= n or time.time() > end:
            break
    return frames


@pytest.fixture
def live(tmp_path):
    made = []

    def build(rig, idle=0.5, timeout=2.0, base_frame=None, **settings):
        masks_shape = (480, 640)
        masks = [rect_mask(masks_shape, *rc) for rc in [(200, 170, 260, 250), (270, 170, 330, 250)]]
        dets = [detection(i, m, class_id=cid, name=n, conf=0.9 - 0.1 * i) for i, (m, (cid, n)) in enumerate(zip(masks, [(48, "sausage"), (23, "egg")]))]
        s = replace(Settings(), meals_db=tmp_path / "meals.sqlite", camera_state_file=tmp_path / "camera_state.json", camera_source="1",
                    camera_timeout_s=timeout, preview_idle_timeout_s=idle, **settings)
        pipe = InferencePipeline(FakeSegmenter(dets), CropProcessor(s.min_mask_pixels, s.min_crop_side), FakeClassifier(["a"] * 101), s, torch.device("cpu"),
                                 {"device": {"resolved": "cpu"}}, {-1: "background", 23: "egg", 48: "sausage"})
        cam = service(rig, idle=idle, timeout=timeout, fps=30, preview_width=320)
        srv = LiveServer(create_app(settings=s, pipeline=pipe, camera=cam))
        made.append(srv)
        return srv, cam
    yield build
    for srv in made:
        srv.close()


def test_stream_serves_downscaled_mjpeg_frames_while_capture_stays_full_resolution(live):
    rig = Rig({1: {"w": 1920, "h": 1080}})
    srv, cam = live(rig)
    st = srv.client.post("/api/camera/preview/start", json={}).json()
    assert st["active"] and (st["width"], st["height"]) == (1920, 1080) and st["stream_url"] == "/api/camera/preview/stream"
    with srv.client.stream("GET", "/api/camera/preview/stream") as r:
        assert r.status_code == 200 and r.headers["content-type"] == "multipart/x-mixed-replace; boundary=frame" and r.headers["cache-control"] == "no-store"
        frames = read_frames(r, 4)
    assert len(frames) >= 3
    img = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 320 and abs(img.shape[0] - 180) <= 1                              # stream copy is downscaled to preview_width
    cap = srv.client.post("/api/camera/capture", json={}).json()
    assert (cap["width"], cap["height"], cap["from_preview"]) == (1920, 1080, True)
    full = cv2.imdecode(np.frombuffer(srv.client.get(cap["image_url"]).content, np.uint8), cv2.IMREAD_COLOR)
    assert full.shape[:2] == (1080, 1920)                                                    # the captured still is full resolution
    assert rig.max_live == 1 and rig.opens == [1]


def test_closing_the_stream_releases_the_camera_after_the_grace_period(live):
    rig = Rig({1: {}})
    srv, cam = live(rig, idle=0.5)
    srv.client.post("/api/camera/preview/start", json={})
    with srv.client.stream("GET", "/api/camera/preview/stream") as r:
        assert len(read_frames(r, 2)) == 2
        assert wait_until(lambda: srv.client.get("/api/camera/preview/status").json()["viewers"] == 1)
        assert rig.live == 1                                                                 # watching: camera stays open past the idle timeout
        time.sleep(0.8)
        assert rig.live == 1
    assert wait_until(lambda: srv.client.get("/api/camera/preview/status").json()["viewers"] == 0, timeout=3)     # browser disconnected
    assert wait_until(lambda: rig.live == 0, timeout=4)
    st = srv.client.get("/api/camera/preview/status").json()
    assert not st["active"] and st["stop_reason"] == "no_viewers"


def test_stream_without_a_preview_is_a_clean_409_and_stop_ends_an_open_stream(live):
    rig = Rig({1: {}})
    srv, cam = live(rig, idle=30)
    r = srv.client.get("/api/camera/preview/stream")
    assert r.status_code == 409 and r.json()["error"]["code"] == "no_preview"
    srv.client.post("/api/camera/preview/start", json={})
    with srv.client.stream("GET", "/api/camera/preview/stream") as r:
        assert len(read_frames(r, 2)) == 2
        stopped = srv.client.post("/api/camera/preview/stop").json()
        assert stopped["state"] == "stopped" and rig.live == 0
        t0 = time.time()
        drain(r)                                                                             # the stream terminates instead of hanging
        assert time.time() - t0 < 3
    assert srv.client.get("/api/camera/preview/status").json()["viewers"] == 0


def test_camera_unplugged_while_streaming_ends_the_stream_and_reports_the_error(live):
    rig = Rig({1: {"die_after": 20}})
    srv, cam = live(rig, idle=30, timeout=2.0)
    srv.client.post("/api/camera/preview/start", json={})
    with srv.client.stream("GET", "/api/camera/preview/stream") as r:
        read_frames(r, 2)
        assert wait_until(lambda: srv.client.get("/api/camera/preview/status").json()["state"] == "error", timeout=6)
        drain(r)                                                                             # drains to the end: the server closed the stream
    st = srv.client.get("/api/camera/preview/status").json()
    assert st["error"]["code"] == "capture_failed" and rig.live == 0
    rig.cfg[1] = {}
    assert srv.client.post("/api/camera/preview/start", json={}).json()["state"] == "running"       # recovers without restarting the server
    assert srv.client.post("/api/camera/preview/stop").status_code == 200


def test_a_stalled_camera_ends_the_stream_instead_of_hanging_the_browser(live):
    gate = threading.Event()
    rig = Rig({1: {}})
    srv, cam = live(rig, idle=30, timeout=0.6)
    srv.client.post("/api/camera/preview/start", json={})
    with srv.client.stream("GET", "/api/camera/preview/stream") as r:
        read_frames(r, 2)
        rig.caps[0].hang = gate                                                              # driver freezes inside grab()
        t0 = time.time()
        drain(r)
        assert time.time() - t0 < 6
    gate.set()
    assert wait_until(lambda: rig.live == 0, timeout=4)


def test_start_validation_and_error_mapping(live):
    rig = Rig({1: {}})
    srv, cam = live(rig, idle=30)
    assert srv.client.post("/api/camera/preview/start", json={"fps": 99}).status_code == 422
    assert srv.client.post("/api/camera/preview/start", json={"preview_width": 50}).status_code == 422
    assert srv.client.post("/api/camera/preview/start", json={"source": 500}).status_code == 422
    bad = srv.client.post("/api/camera/preview/start", json={"source": 3})
    assert bad.status_code == 409 and bad.json()["error"]["code"] == "camera_unavailable"
    ok = srv.client.post("/api/camera/preview/start", json={"fps": 5, "preview_width": 200}).json()
    assert ok["target_fps"] == 5 and ok["preview_width"] == 200
    assert srv.client.get("/api/camera/status").json()["preview"]["active"] is True
    other = srv.client.post("/api/camera/capture", json={"source": 2})
    assert other.status_code == 409 and other.json()["error"]["code"] == "camera_in_use"


def test_server_shutdown_releases_a_running_preview(live):
    rig = Rig({1: {}})
    srv, cam = live(rig, idle=60)
    srv.client.post("/api/camera/preview/start", json={})
    assert rig.live == 1
    srv.close()
    assert rig.live == 0


# ============================================================ capture during preview -> existing inference workflow
def test_frame_captured_during_preview_goes_through_inference_and_automatic_plate_localization(live):
    rig = Rig({1: {"w": 640, "h": 480, "base": np.ascontiguousarray(plate_scene()[..., ::-1])}})
    srv, cam = live(rig, idle=30)
    c = srv.client
    c.post("/api/camera/preview/start", json={})
    with c.stream("GET", "/api/camera/preview/stream") as r:                                  # a browser is watching
        assert len(read_frames(r, 2)) == 2
        cap = c.post("/api/camera/capture", json={}).json()
        assert cap["from_preview"] is True
        jpg = c.get(cap["image_url"]).content
        res = c.post("/api/inference", files=[("files", (cap["filename"], jpg, "image/jpeg"))], data={"mode": "yolo_only"}).json()["items"][0]["result"]
        auto = c.post(f"/api/results/{res['result_id']}/plate/auto", json={"camera_id": cap["camera_id"]}).json()
        assert len(res["segments"]) == 2 and len(res["regions"]) == 2                          # YOLO (fake) segmentation ran on the captured frame
        assert auto["applied"] and auto["source"] == "automatic_detection"
        assert abs(auto["plate"]["center_x"] - 320) < 3 and abs(auto["plate"]["radius_px"] - 201.6) < 3
        assert c.get(f"/api/results/{res['result_id']}/plate-analysis").json()["plate_source"] == "automatic_detection"
        # the preview was never disturbed by any of that
        assert c.get("/api/camera/preview/status").json()["state"] == "running" and rig.opens == [1] and rig.max_live == 1
