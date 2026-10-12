"""ESP32 load cell -> automatic food scan. Three small parts, none of them a second CV pipeline:
  SerialListener   reads the firmware protocol from a serial port on its own thread and reconnects when the port drops.
  ScanCoordinator  turns WEIGHT_STABLE into exactly one scan per plate placement (re-armed by SCALE_EMPTY) and runs the injected
                   `run_scan` (camera capture + the existing InferencePipeline) off the serial thread.
  build_router     /api/scale/status and /api/scans/* for the phone. The phone owns the canonical meal; nothing is saved here.
Firmware protocol: `WEIGHT_STABLE:<grams>`, `SCALE_EMPTY`, `# diagnostics`. This module only reads: it never sends tare/calibration."""
from __future__ import annotations

import copy
import logging
import math
import re
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Callable, Literal, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .regions import MAX_WEIGHT_G

log = logging.getLogger("testbench.scale")
_STABLE = re.compile(r"^WEIGHT_STABLE:\s*(\d+(?:\.\d+)?)$")
MAX_CANDIDATES = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_line(raw: str) -> Optional[tuple[str, Optional[float]]]:
    """('stable', grams) | ('empty', None) | None for diagnostics (#...), blanks and anything malformed."""
    line = raw.strip()
    if line == "SCALE_EMPTY":
        return ("empty", None)
    m = _STABLE.match(line)
    return ("stable", float(m.group(1))) if m else None


def foods_from_regions(regions: list[dict], nutri) -> list[dict]:
    """Detected foods (one per region, excluded ones dropped) with ranked candidates and, where USDA maps unambiguously, per-100 g data.
    Unmapped / ambiguous labels carry per_100g=None: unknown stays unknown, it is never turned into zero."""
    foods = []
    for r in regions:
        if r.get("excluded"):
            continue
        cands = []
        for c in r["candidates"][:MAX_CANDIDATES]:
            look = nutri.lookup(c["taxonomy"], c["label"])
            sel = look["selected"]
            cands.append({"label": c["label"], "taxonomy": c["taxonomy"], "source": c["source"], "confidence": c["confidence"],
                          "per_100g": sel["per_100g"] if sel else None,
                          "usda": ({k: sel[k] for k in ("fdc_id", "description", "data_source", "preparation")} if sel else None)})
        foods.append({"region_id": r["region_id"], "candidates": cands})
    return foods


class ScanCoordinator:
    def __init__(self, run_scan: Callable[[dict], dict], tare_g: float = 0.0, min_net_g: float = 5.0, keep: int = 50):
        self._run_scan, self.tare_g, self.min_net_g, self.keep = run_scan, tare_g, min_net_g, keep
        self._lock = threading.Lock()
        self._scans: "OrderedDict[str, dict]" = OrderedDict()
        self.armed, self.busy = True, False          # armed: a new plate placement may start a scan
        self.last_ignored: Optional[dict] = None
        self._thread: Optional[threading.Thread] = None

    def handle(self, kind: str, grams: Optional[float]) -> None:
        with self._lock:
            if kind == "empty":
                self.armed = True                    # a finished scan stays pending; only the next placement is enabled
                return
            net = None if grams is None else round(grams - self.tare_g, 1)
            reason = ("duplicate_placement" if not self.armed else "scan_in_progress" if self.busy
                      else "invalid_weight" if net is None or not math.isfinite(net) or net < self.min_net_g or net > MAX_WEIGHT_G else None)
            if reason:
                self.last_ignored = {"reason": reason, "grams": grams, "at": _now()}
                if reason == "scan_in_progress":
                    self.armed = False               # this placement is lost; require the plate to be removed first
                log.info("WEIGHT_STABLE %s ignored: %s", grams, reason)
                return
            self.armed, self.busy = False, True
            scan = {"scan_id": uuid.uuid4().hex[:12], "timestamp": _now(), "status": "processing", "measured_total_grams": grams,
                    "tare_g": self.tare_g, "net_grams": net, "foods": [], "needs_manual_food": False, "result_id": None,
                    "error": None, "acknowledged_at": None, "outcome": None}
            self._scans[scan["scan_id"]] = scan
            while len(self._scans) > self.keep:
                self._scans.popitem(last=False)
            self._thread = threading.Thread(target=self._work, args=(scan,), daemon=True, name="scale-scan")
            self._thread.start()

    def _work(self, scan: dict) -> None:
        try:
            out = self._run_scan(dict(scan))
            update = {**out, "status": "ready"}
        except Exception as exc:                     # keep the failure on the scan; the next placement is still allowed
            log.exception("scan %s failed", scan["scan_id"])
            update = {"status": "error", "error": {"code": "scan_failed", "message": str(exc) or type(exc).__name__}}
        with self._lock:
            scan.update(update)
            self.busy = False

    def join(self, timeout: float = 10.0) -> None:
        t = self._thread
        if t is not None:
            t.join(timeout)

    def get(self, scan_id: str) -> Optional[dict]:
        with self._lock:
            s = self._scans.get(scan_id)
            return copy.deepcopy(s) if s else None

    def latest(self, include_acknowledged: bool = False) -> Optional[dict]:
        """Newest scan that has not been acknowledged (processing, ready or error). The dashboard also wants acknowledged ones."""
        with self._lock:
            for s in reversed(self._scans.values()):
                if include_acknowledged or s["acknowledged_at"] is None:
                    return copy.deepcopy(s)
        return None

    def active_result_ids(self) -> set:
        """Results the phone may still be about to review: their scans are ready/failed and not yet acknowledged."""
        with self._lock:
            return {s["result_id"] for s in self._scans.values() if s["result_id"] and s["acknowledged_at"] is None}

    def acknowledge(self, scan_id: str, outcome: str) -> Optional[dict]:
        """Idempotent. Raises ValueError for a scan that is still processing."""
        with self._lock:
            s = self._scans.get(scan_id)
            if s is None:
                return None
            if s["status"] == "processing":
                raise ValueError("scan is still processing")
            if s["acknowledged_at"] is None:
                s["acknowledged_at"], s["outcome"] = _now(), outcome
            return copy.deepcopy(s)


class SerialListener:
    def __init__(self, port: str, baud: int, on_event: Callable[[str, Optional[float]], None], opener=None, retry_s: float = 3.0):
        self.port, self.baud, self.on_event, self.retry_s = port, baud, on_event, retry_s
        self._opener = opener or self._open_serial
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.state: Literal["stopped", "connecting", "connected", "error"] = "stopped"
        self.last_error: Optional[str] = None
        self.firmware_ready = False                  # saw "# Tared. Ready" since the last open
        self.connected_at = self.last_line_at = self.last_event_at = None
        self.last_event: Optional[dict] = None
        self.lines_ignored = 0

    @staticmethod
    def _open_serial(port: str, baud: int):
        import serial
        ser = serial.Serial()
        ser.port, ser.baudrate, ser.timeout = port, baud, 1.0
        ser.dtr = ser.rts = False                    # keep the ESP32 from auto-resetting when the port opens (it still copes if it does)
        ser.open()
        return ser

    def start(self) -> None:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name="scale-serial")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(3.0)
            self._thread = None
        self.state = "stopped"

    def _run(self) -> None:
        while not self._stop.is_set():
            self.state = "connecting"
            ser = None
            try:
                ser = self._opener(self.port, self.baud)
                self.state, self.last_error, self.firmware_ready, self.connected_at = "connected", None, False, _now()
                log.info("serial %s opened at %d baud", self.port, self.baud)
                while not self._stop.is_set():
                    raw = ser.readline()
                    if raw:
                        self._line(raw.decode("utf-8", errors="ignore"))
            except Exception as exc:                 # port missing / in use (PermissionError) / unplugged mid-read
                self.state, self.last_error = "error", f"{type(exc).__name__}: {exc}"
                log.warning("serial %s: %s; retrying in %.0f s", self.port, self.last_error, self.retry_s)
            finally:
                if ser is not None:
                    try:
                        ser.close()
                    except Exception:
                        pass
            self._stop.wait(self.retry_s)

    def _line(self, text: str) -> None:
        self.last_line_at = _now()
        if text.strip() == "# Tared. Ready":
            self.firmware_ready = True
        ev = parse_line(text)
        if ev is None:
            self.lines_ignored += 1
            return
        self.last_event, self.last_event_at = {"kind": ev[0], "grams": ev[1]}, _now()
        self.on_event(*ev)

    def status(self) -> dict:
        return {"port": self.port, "baud": self.baud, "state": self.state, "firmware_ready": self.firmware_ready,
                "last_error": self.last_error, "connected_at": self.connected_at, "last_line_at": self.last_line_at,
                "last_event": self.last_event, "last_event_at": self.last_event_at}


class AckIn(BaseModel):
    outcome: Literal["saved", "rejected"]


def build_router(coord: ScanCoordinator, listener: Optional[SerialListener]) -> APIRouter:
    router = APIRouter(tags=["scale"])

    def not_found(scan_id):
        return JSONResponse({"error": {"code": "scan_not_found", "message": f"No scan '{scan_id}'"}}, status_code=404)

    @router.get("/api/scale/status")
    def status():
        return {"enabled": listener is not None, "serial": listener.status() if listener else None, "armed": coord.armed,
                "busy": coord.busy, "last_ignored": coord.last_ignored}

    @router.get("/api/scans/latest")
    def latest(include_acknowledged: bool = False):
        return {"scan": coord.latest(include_acknowledged)}

    @router.get("/api/scans/{scan_id}")
    def get(scan_id: str):
        return coord.get(scan_id) or not_found(scan_id)

    @router.post("/api/scans/{scan_id}/acknowledge")
    def acknowledge(scan_id: str, body: AckIn):
        try:
            scan = coord.acknowledge(scan_id, body.outcome)
        except ValueError as exc:
            return JSONResponse({"error": {"code": "scan_processing", "message": str(exc)}}, status_code=409)
        return scan or not_found(scan_id)

    return router
