"""CameraService with a fake cv2.VideoCapture: no real device is ever opened by these tests."""
import threading
import time
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from testbench.camera import CameraError, CameraService, camera_id, parse_camera_id, parse_source
from testbench.config import Settings
from testbench.main import create_app


class FakeCap:
    def __init__(self, frame=None, opened=True, reads=None, read_delay=0.0, hang=None):
        self.frame = np.full((480, 640, 3), 90, np.uint8) if frame is None else frame
        self.opened, self.read_delay, self.hang = opened, read_delay, hang
        self.reads_left = reads            # None = unlimited good reads; int = that many, then failures (disconnect)
        self.released, self.reads, self.props = False, 0, {}

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def read(self):
        if self.hang is not None:
            self.hang.wait(5)              # a driver stuck inside read()
        if self.read_delay:
            time.sleep(self.read_delay)
        self.reads += 1
        if self.reads_left is not None and self.reads > self.reads_left:
            return False, None
        return True, self.frame.copy()

    def release(self):
        self.released = True


class Devices:
    """capture_factory: source -> FakeCap. Records every open so tests can assert on opens and releases."""
    def __init__(self, caps: dict):
        self.caps, self.opened = caps, []

    def __call__(self, source, backend_flag):
        cap = self.caps.get(source)
        if cap is None:
            cap = FakeCap(opened=False)
        self.opened.append((source, cap))
        return cap


def service(devices, **kw):
    kw.setdefault("warmup_frames", 2)
    kw.setdefault("timeout_s", 2.0)
    return CameraService("any", 0, 0, capture_factory=devices, **kw)


def test_capture_success_reports_actual_dimensions_and_releases():
    cap = FakeCap(np.full((720, 1280, 3), 50, np.uint8))
    dev = Devices({1: cap})
    fr = service(dev).capture(1)
    assert (fr.width, fr.height) == (1280, 720) and fr.bgr.shape == (720, 1280, 3)
    assert fr.info()["camera_id"] == "index:1"
    assert cap.released and cap.reads == 3               # 2 warm-up frames discarded + the kept one


def test_requested_size_is_set_as_a_capture_property_but_actual_size_is_reported():
    import cv2
    cap = FakeCap(np.full((480, 640, 3), 50, np.uint8))     # the "driver" ignores the request and keeps 640x480
    svc = CameraService("any", 1920, 1080, warmup_frames=0, timeout_s=2, capture_factory=Devices({0: cap}))
    fr = svc.capture(0)
    assert cap.props[cv2.CAP_PROP_FRAME_WIDTH] == 1920 and cap.props[cv2.CAP_PROP_FRAME_HEIGHT] == 1080
    assert (fr.width, fr.height) == (640, 480) and fr.requested == (1920, 1080)


def test_grayscale_frame_is_returned_as_three_channels():
    fr = service(Devices({0: FakeCap(np.full((100, 120), 7, np.uint8))})).capture(0)
    assert fr.bgr.shape == (100, 120, 3)


def test_camera_unavailable_is_an_error_not_a_crash_and_is_released():
    cap = FakeCap(opened=False)
    with pytest.raises(CameraError) as e:
        service(Devices({3: cap})).capture(3)
    assert e.value.code == "camera_unavailable"
    assert cap.released


def test_factory_exception_becomes_camera_unavailable():
    def boom(source, flag):
        raise RuntimeError("driver exploded")
    with pytest.raises(CameraError) as e:
        CameraService("any", capture_factory=boom, timeout_s=2).capture(0)
    assert e.value.code == "camera_unavailable" and "driver exploded" not in e.value.message


@pytest.mark.parametrize("bad", [-1, 64, 999, "abc", "", "1.5", "http://cam/stream", "/etc/passwd", "/dev/video0/../sda", None, True, 1.5])
def test_invalid_camera_sources_are_rejected_before_any_device_is_touched(bad):
    dev = Devices({})
    with pytest.raises(CameraError) as e:
        service(dev).capture(bad)
    assert e.value.code == "invalid_camera" and e.value.http_status == 422
    assert dev.opened == []


def test_source_parsing_accepts_indices_and_linux_device_paths():
    assert parse_source("2") == 2 and parse_source(0) == 0
    assert parse_source("/dev/video0") == "/dev/video0"
    assert parse_source("/dev/v4l/by-id/usb-Cam_1234-video-index0") == "/dev/v4l/by-id/usb-Cam_1234-video-index0"
    assert camera_id(1) == "index:1" and camera_id("/dev/video0") == "path:/dev/video0"
    assert parse_camera_id("path:/dev/video2") == "path:/dev/video2" and parse_camera_id("index:3") == "index:3"
    for bad in ("3", "index:", "index:x", "foo:1", "path:/etc/passwd"):
        with pytest.raises(CameraError):
            parse_camera_id(bad)


def test_disconnect_mid_capture_is_capture_failed_and_released():
    cap = FakeCap(reads=1)                                   # first read ok, then the device vanishes
    with pytest.raises(CameraError) as e:
        service(Devices({0: cap})).capture(0)
    assert e.value.code == "capture_failed" and cap.released


def test_probe_reports_failure_instead_of_raising():
    svc = service(Devices({0: FakeCap()}))
    ok, missing = svc.probe(0), svc.probe(4)
    assert ok["connected"] and ok["capture_ok"] and (ok["width"], ok["height"]) == (640, 480) and ok["error"] is None
    assert not missing["connected"] and not missing["capture_ok"] and missing["error_code"] == "camera_unavailable" and missing["width"] is None


def test_enumerate_probes_each_index_and_never_assumes_index_zero():
    dev = Devices({0: FakeCap(), 2: FakeCap(np.full((720, 1280, 3), 1, np.uint8))})
    found = service(dev).enumerate(4)
    assert [d["camera_id"] for d in found] == ["index:0", "index:1", "index:2", "index:3"]
    assert [d["capture_ok"] for d in found] == [True, False, True, False]
    assert all(c.released for _, c in dev.opened)


def test_hung_driver_cannot_block_the_request_and_marks_camera_busy():
    gate = threading.Event()
    cap = FakeCap(hang=gate)
    svc = service(Devices({0: cap, 1: FakeCap()}), timeout_s=0.3)
    t0 = time.perf_counter()
    with pytest.raises(CameraError) as e:
        svc.capture(0)
    assert e.value.code == "camera_timeout" and time.perf_counter() - t0 < 2.0
    with pytest.raises(CameraError) as e2:                   # the stuck operation still owns the device lock
        svc.capture(1)
    assert e2.value.code == "camera_busy"
    gate.set()                                               # driver finally returns: the worker releases the camera and the lock
    deadline = time.time() + 3
    while not cap.released and time.time() < deadline:
        time.sleep(0.02)
    assert cap.released
    time.sleep(0.1)
    assert svc.capture(1).width == 640                       # service is usable again


def test_camera_is_not_opened_by_creating_the_service_or_the_app(tmp_path):
    dev = Devices({0: FakeCap()})
    s = replace(Settings(), meals_db=tmp_path / "m.sqlite", camera_state_file=tmp_path / "cam.json")
    with TestClient(create_app(settings=s, pipeline=object(), camera=service(dev))) as c:
        assert c.get("/api/camera/status").status_code == 200 and c.get("/api/camera/calibration").status_code == 200
    assert dev.opened == []
