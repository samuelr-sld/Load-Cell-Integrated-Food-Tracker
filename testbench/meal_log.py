"""Minimal SQLite meal log. One finalized meal = one `meals` row plus one `meal_items` row per allocated food, written in a
single transaction. Rows are self-contained snapshots (weights, USDA record, per-100 g values, nutrition), so a saved meal
never depends on in-memory results or on later changes to the USDA cache. The file is opened lazily: creating the app (or
running the unrelated tests) never creates it. It is separate from nutrition.sqlite, which the server never writes."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS meals (
    meal_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, result_id TEXT NOT NULL, filename TEXT,
    fingerprint TEXT NOT NULL UNIQUE,
    completeness_status TEXT NOT NULL CHECK (completeness_status IN ('complete','incomplete')),
    allocation_status TEXT NOT NULL, validation_status TEXT NOT NULL, incomplete_reasons TEXT NOT NULL,
    weight_source TEXT NOT NULL, total_weight_g REAL NOT NULL, allocated_weight_g REAL NOT NULL, unresolved_weight_g REAL NOT NULL,
    allocation_method TEXT NOT NULL, plate_summary TEXT NOT NULL, unresolved TEXT NOT NULL,
    totals TEXT, warnings TEXT NOT NULL, disclaimer TEXT NOT NULL, schema_version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS meal_items (
    meal_id TEXT NOT NULL REFERENCES meals(meal_id) ON DELETE CASCADE, region_id INTEGER NOT NULL, region_source TEXT NOT NULL,
    label TEXT NOT NULL, taxonomy TEXT, detected_segments INTEGER NOT NULL, user_entity_count INTEGER,
    estimated_weight_g REAL NOT NULL, weight_per_entity_g REAL, area_px REAL, food_area_pct REAL,
    fdc_id INTEGER, usda_description TEXT, usda_data_source TEXT, usda_preparation TEXT, per_100g TEXT, nutrition TEXT,
    PRIMARY KEY (meal_id, region_id));
"""


def fingerprint(snapshot: dict) -> str:
    """Identity of a finalized analysis: same photo result, same foods/weights/total/status => same meal. Excludes timestamps."""
    core = {"result_id": snapshot["result_id"], "status": snapshot["completeness_status"], "total": snapshot["total_weight_g"],
            "source": snapshot["weight_source"], "plate": snapshot["plate_summary"],
            "items": [[i["region_id"], i["label"], i["taxonomy"], i["user_entity_count"], i["estimated_weight_g"],
                       (i["usda"] or {}).get("fdc_id")] for i in snapshot["items"]]}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()


class MealLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    def _db(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)   # explicit BEGIN/COMMIT below
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA journal_mode = WAL")
                conn.executescript(SCHEMA)
                self._conn = conn
            return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def save(self, snap: dict) -> tuple[dict, bool]:
        """Returns (meal, created). An identical snapshot returns the earlier meal with created=False instead of inserting."""
        fp = fingerprint(snap)
        with self._lock:
            db = self._db()
            db.execute("BEGIN IMMEDIATE")
            try:
                existing = db.execute("SELECT meal_id FROM meals WHERE fingerprint = ?", (fp,)).fetchone()
                if existing:
                    db.execute("ROLLBACK")
                    return self.get(existing["meal_id"]), False
                meal_id = "meal_" + uuid.uuid4().hex[:12]
                db.execute("INSERT INTO meals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    meal_id, datetime.now(timezone.utc).isoformat(timespec="seconds"), snap["result_id"], snap["filename"], fp,
                    snap["completeness_status"], snap["allocation_status"], snap["validation_status"], json.dumps(snap["incomplete_reasons"]),
                    snap["weight_source"], snap["total_weight_g"], snap["allocated_weight_g"], snap["unresolved_weight_g"],
                    snap["allocation_method"], json.dumps(snap["plate_summary"]), json.dumps(snap["unresolved"]),
                    json.dumps(snap["totals"]), json.dumps(snap["warnings"]), snap["disclaimer"], SCHEMA_VERSION))
                for i in snap["items"]:
                    u = i["usda"] or {}
                    db.execute("INSERT INTO meal_items VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                        meal_id, i["region_id"], i["region_source"], i["label"], i["taxonomy"], i["detected_segments"], i["user_entity_count"],
                        i["estimated_weight_g"], i["weight_per_entity_g"], i["area_px"], i["food_area_pct"],
                        u.get("fdc_id"), u.get("description"), u.get("data_source"), u.get("preparation"),
                        json.dumps(i["per_100g"]), json.dumps(i["nutrition"])))
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise
        return self.get(meal_id), True

    def _meal(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        for k in ("incomplete_reasons", "plate_summary", "unresolved", "totals", "warnings"):
            d[k] = json.loads(d[k]) if d[k] is not None else None
        d.pop("fingerprint")
        return d

    def get(self, meal_id: str) -> dict | None:
        with self._lock:
            db = self._db()
            row = db.execute("SELECT * FROM meals WHERE meal_id = ?", (meal_id,)).fetchone()
            if row is None:
                return None
            meal = self._meal(row)
            meal["items"] = [{**dict(r), "per_100g": json.loads(r["per_100g"]) if r["per_100g"] else None,
                              "nutrition": json.loads(r["nutrition"]) if r["nutrition"] else None}
                             for r in db.execute("SELECT * FROM meal_items WHERE meal_id = ? ORDER BY region_id", (meal_id,))]
            for i in meal["items"]:
                i.pop("meal_id")
            return meal

    def list(self, limit: int = 100) -> list[dict]:
        with self._lock:
            ids = [r["meal_id"] for r in self._db().execute("SELECT meal_id FROM meals ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,))]
            return [self.get(i) for i in ids]
