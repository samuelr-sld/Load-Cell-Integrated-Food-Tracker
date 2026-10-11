"""Local USDA FoodData Central reference: SQLite cache of real FDC records plus label -> record mappings.
Nothing here calls the network and nothing is estimated: a nutrient that USDA did not report stays NULL and is returned as null."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional

NUTRIENT_FIELDS = ("energy_kcal", "protein_g", "carbs_g", "fat_g")
TAXONOMIES = ("foodseg103", "food101")   # label vocabularies the models emit; anything else is "custom" and never mapped
PREPARATIONS = ("raw", "cooked", "prepared")

SCHEMA = """
CREATE TABLE IF NOT EXISTS foods (
    fdc_id INTEGER PRIMARY KEY, description TEXT NOT NULL, data_source TEXT NOT NULL,
    preparation TEXT NOT NULL CHECK (preparation IN ('raw','cooked','prepared')),
    energy_kcal REAL, protein_g REAL, carbs_g REAL, fat_g REAL,
    energy_basis TEXT, imported_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS label_map (
    taxonomy TEXT NOT NULL, label TEXT NOT NULL, fdc_id INTEGER NOT NULL REFERENCES foods(fdc_id),
    PRIMARY KEY (taxonomy, label, fdc_id));
"""


class NutritionError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def norm_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())


def connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


class NutritionService:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._conn = connect(self.db_path)
        self._lock = threading.Lock()

    def _all(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    @staticmethod
    def _food(row: sqlite3.Row) -> dict:
        return {"fdc_id": row["fdc_id"], "description": row["description"], "data_source": row["data_source"],
                "preparation": row["preparation"], "energy_basis": row["energy_basis"],
                "per_100g": {k: row[k] for k in NUTRIENT_FIELDS}}

    def stats(self) -> dict:
        return {"usda_records": self._all("SELECT COUNT(*) n FROM foods")[0]["n"], "mapped_labels": self._all("SELECT COUNT(*) n FROM (SELECT 1 FROM label_map GROUP BY taxonomy, label)")[0]["n"]}

    def candidates(self, taxonomy: str, label: str) -> list[dict]:
        rows = self._all("SELECT f.* FROM label_map m JOIN foods f ON f.fdc_id = m.fdc_id WHERE m.taxonomy = ? AND m.label = ? ORDER BY f.preparation, f.fdc_id",
                         (taxonomy, norm_label(label)))
        return [self._food(r) for r in rows]

    def food(self, fdc_id: int) -> Optional[dict]:
        rows = self._all("SELECT * FROM foods WHERE fdc_id = ?", (fdc_id,))
        return self._food(rows[0]) if rows else None

    def mapped_labels(self) -> list[dict]:
        rows = self._all("SELECT taxonomy, label, COUNT(*) n FROM label_map GROUP BY taxonomy, label ORDER BY taxonomy, label")
        return [{"taxonomy": r["taxonomy"], "label": r["label"], "num_usda_records": r["n"]} for r in rows]

    def lookup(self, taxonomy: str, label: str, fdc_id: Optional[int] = None) -> dict:
        """Resolve a confirmed label to USDA data. Never guesses: several candidate records means the caller must choose one.
        status: ok | needs_usda_selection | unmapped | invalid_selection."""
        cands = self.candidates(taxonomy, label) if taxonomy in TAXONOMIES else []
        base = {"taxonomy": taxonomy, "label": norm_label(label), "candidates": cands, "selected": None}
        if not cands:
            return {**base, "status": "unmapped", "message": f"No USDA record is mapped to '{label}'. No nutrition data is available for it."}
        if fdc_id is not None:
            chosen = next((c for c in cands if c["fdc_id"] == fdc_id), None)
            if chosen is None:
                return {**base, "status": "invalid_selection", "message": f"FDC ID {fdc_id} is not a mapped option for '{label}'."}
        elif len(cands) == 1:
            chosen = cands[0]
        else:
            return {**base, "status": "needs_usda_selection",
                    "message": f"'{label}' maps to {len(cands)} USDA records (e.g. raw vs cooked). Choose one by fdc_id."}
        return {**base, "status": "ok", "selected": chosen, "message": None}


def serving_nutrition(per_100g: dict, weight_g: float) -> dict:
    """Scale per-100 g values to a manually entered weight. A nutrient USDA did not report stays None."""
    return {k: (None if per_100g.get(k) is None else round(per_100g[k] * weight_g / 100.0, 2)) for k in NUTRIENT_FIELDS}
