"""One-time USDA FoodData Central import into the local SQLite cache.

    python -m testbench.scripts.usda_import search "broccoli raw"      # list real FDC records to curate from
    python -m testbench.scripts.usda_import import                     # fetch every record in usda_curated.json and cache it

The API key is read from the USDA_API_KEY environment variable (falls back to the public, rate-limited DEMO_KEY).
It is used only here, on the backend; the HTTP API never sees or returns it. Branded foods are refused because
their nutrients are not per 100 g of a defined food.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from testbench.config import Settings
from testbench.nutrition import PREPARATIONS, TAXONOMIES, connect, norm_label

BASE = "https://api.nal.usda.gov/fdc/v1"
CURATED = Path(__file__).resolve().parent.parent / "usda_curated.json"
# FDC nutrient ids. Foundation foods often lack 1008 and report Atwater-computed energy instead (2047 general / 2048 specific).
PROTEIN, FAT, CARBS, ENERGY, ATWATER_GENERAL, ATWATER_SPECIFIC = 1003, 1004, 1005, 1008, 2047, 2048


def _get(path: str, params: dict) -> dict:
    params = {**params, "api_key": os.environ.get("USDA_API_KEY") or "DEMO_KEY"}
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 3:
                time.sleep(20)
                continue
            raise SystemExit(f"USDA API error {exc.code} for {path} (key redacted)") from None
    raise SystemExit("USDA API rate limit; set USDA_API_KEY to your own key and retry")


def extract_nutrients(food: dict) -> tuple[dict, str]:
    by_id: dict[int, float] = {}
    for n in food.get("foodNutrients", []):
        nid = (n.get("nutrient") or {}).get("id", n.get("nutrientId"))
        amount = n.get("amount", n.get("value"))
        if nid is not None and amount is not None:
            by_id[int(nid)] = float(amount)
    basis, energy = "reported", by_id.get(ENERGY)
    if energy is None:
        for nid, name in ((ATWATER_GENERAL, "atwater_general"), (ATWATER_SPECIFIC, "atwater_specific")):
            if nid in by_id:
                basis, energy = name, by_id[nid]
                break
    return {"energy_kcal": energy, "protein_g": by_id.get(PROTEIN), "carbs_g": by_id.get(CARBS), "fat_g": by_id.get(FAT)}, basis


def cmd_search(args) -> None:
    for dtype in ("Foundation", "SR Legacy"):
        data = _get("/foods/search", {"query": args.query, "dataType": dtype, "pageSize": args.n})
        for f in data.get("foods", []):
            print(f"{f['fdcId']}\t{dtype}\t{f['description']}")


def cmd_import(args) -> None:
    spec = json.loads(CURATED.read_text(encoding="utf-8"))
    conn = connect(Settings().nutrition_db)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cache: dict[int, dict] = {}
    bad = 0
    for entry in spec["mappings"]:
        taxonomy, label = entry["taxonomy"], norm_label(entry["label"])
        if taxonomy not in TAXONOMIES:
            raise SystemExit(f"unknown taxonomy {taxonomy!r}")
        for rec in entry["records"]:
            fdc_id, prep = int(rec["fdc_id"]), rec["preparation"]
            if prep not in PREPARATIONS:
                raise SystemExit(f"fdc {fdc_id}: preparation must be one of {PREPARATIONS}")
            if fdc_id not in cache:
                have = conn.execute("SELECT 1 FROM foods WHERE fdc_id = ?", (fdc_id,)).fetchone()
                if have:   # already cached from USDA: keep it untouched, only add the label mapping
                    conn.execute("INSERT OR REPLACE INTO label_map VALUES (?,?,?)", (taxonomy, label, fdc_id))
                    continue
                food = _get(f"/food/{fdc_id}", {})
                if food.get("dataType") == "Branded":
                    raise SystemExit(f"fdc {fdc_id} is a Branded food; refusing (not per 100 g of a defined food)")
                if rec["expect"].lower() not in food.get("description", "").lower():
                    print(f"SKIPPED fdc {fdc_id}: USDA says {food.get('description')!r}, expected it to contain {rec['expect']!r}", file=sys.stderr)
                    bad += 1
                    continue
                nutrients, basis = extract_nutrients(food)
                cache[fdc_id] = {"description": food["description"], "data_source": food.get("dataType", "unknown"), "nutrients": nutrients, "basis": basis}
                missing = [k for k, v in nutrients.items() if v is None]
                print(f"{fdc_id}\t{food.get('dataType')}\t{food['description']}\t{nutrients}" + (f"  MISSING {missing}" if missing else ""))
                time.sleep(0.3)
            if fdc_id not in cache:
                continue
            c = cache[fdc_id]
            conn.execute("INSERT OR REPLACE INTO foods VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (fdc_id, c["description"], c["data_source"], prep, *(c["nutrients"][k] for k in ("energy_kcal", "protein_g", "carbs_g", "fat_g")), c["basis"], now))
            conn.execute("INSERT OR REPLACE INTO label_map VALUES (?,?,?)", (taxonomy, label, fdc_id))
    conn.commit()
    print(f"cached {len(cache)} USDA records -> {Settings().nutrition_db}")
    if bad:
        raise SystemExit(f"{bad} record(s) skipped: fix usda_curated.json (use the `search` command to find the right FDC IDs)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-n", type=int, default=8)
    s.set_defaults(fn=cmd_search)
    sub.add_parser("import").set_defaults(fn=cmd_import)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
