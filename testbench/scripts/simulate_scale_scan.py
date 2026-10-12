"""TEST-ONLY scale-scan simulator. Serves the real testbench app (real models, ResultStore, /api/scans/*, dashboard, LAN guard)
with ONE substitution: the camera returns a still photo. Firmware lines are fed into the same ScanCoordinator.handle the
SerialListener calls, via the same parse_line. Nothing is sent to COM3 (the serial listener is forced off) and no route is added.

Drive it by appending firmware lines to the inbox file (printed at start), e.g. from PowerShell:
    Set-Content $env:TEMP\\platego_scale_inbox.txt "WEIGHT_STABLE:250.0"     # a plate placed
    Set-Content $env:TEMP\\platego_scale_inbox.txt "SCALE_EMPTY"             # plate removed: re-arms the next scan
Run:  venv\\Scripts\\python -m testbench.scripts.simulate_scale_scan IMAGE [--host 0.0.0.0] [--port 8010] [--capture-s 3]"""
import argparse
import dataclasses
import logging
import tempfile
import threading
import time
from pathlib import Path

import cv2
import uvicorn

from testbench.camera import CameraService
from testbench.config import Settings
from testbench.main import create_app
from testbench.scale_scan import parse_line

log = logging.getLogger("simulate_scale_scan")


class StillCapture:
    """cv2.VideoCapture stand-in that always returns the same frame, after `delay_s` (a real USB camera's open + warm-up)."""
    def __init__(self, bgr, delay_s=0.0):
        self.bgr, self.delay_s = bgr, delay_s

    def isOpened(self): return True
    def set(self, *_): return True
    def get(self, *_): return 0.0

    def read(self):
        time.sleep(self.delay_s)
        return True, self.bgr.copy()
    def release(self): pass


def read_inbox(path: Path) -> list[str]:
    raw = path.read_bytes()
    path.unlink()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")   # PowerShell 5.1 `>` writes UTF-16
    return [ln for ln in text.splitlines() if ln.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--capture-s", type=float, default=0.0, help="simulated camera warm-up before the frame (keeps 'processing' visible)")
    ap.add_argument("--inbox", type=Path, default=Path(tempfile.gettempdir()) / "platego_scale_inbox.txt")
    args = ap.parse_args()
    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"cannot read image {args.image}")

    settings = dataclasses.replace(Settings(), host=args.host, port=args.port, scale_enabled=False)
    camera = CameraService("any", 0, 0, warmup_frames=0, capture_factory=lambda _src, _flag: StillCapture(bgr, args.capture_s))
    app = create_app(settings, camera=camera)
    args.inbox.unlink(missing_ok=True)

    def pump():
        while True:
            time.sleep(0.3)
            if args.inbox.exists():
                for line in read_inbox(args.inbox):
                    ev = parse_line(line)
                    log.info("inbox %r -> %s", line, ev)
                    if ev:
                        app.state.scale.handle(*ev)

    threading.Thread(target=pump, daemon=True, name="sim-inbox").start()
    log.info("TEST HARNESS (still image %s) on http://%s:%d  inbox: %s", args.image, args.host, args.port, args.inbox)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
