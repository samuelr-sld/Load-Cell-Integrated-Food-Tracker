// Laptop scan (testbench /api/scans) -> meal form draft -> canonical meal input. Pure (no React, no network).
// Unknown nutrition stays blank/null, never 0. The scan is only a suggestion until the user confirms the form.
import { toDraft, emptyItem } from './mealForm.js';

const numStr = (v) => (v == null ? '' : String(v));

// USDA per-100g fields (testbench names) -> the app's nutrient names. Missing stays blank.
export const candidateNutrients = (cand) => ({
  calories: numStr(cand?.per_100g?.energy_kcal),
  protein: numStr(cand?.per_100g?.protein_g),
  carbs: numStr(cand?.per_100g?.carbs_g),
  fat: numStr(cand?.per_100g?.fat_g),
});

export const applyCandidate = (item, cand) => ({ ...item, name: cand.label, ...candidateNutrients(cand) });

// One item per detected food using its top candidate. Grams are prefilled only when the plate holds exactly one food:
// with several, the user must allocate the portions (blank grams fails validation until they do).
export const scanToDraft = (scan, nowMs = Date.now()) => {
  const items = (scan.foods ?? []).map((f) => applyCandidate(
    { ...emptyItem(), regionId: f.region_id, grams: scan.foods.length === 1 ? numStr(scan.suggested_grams ?? scan.net_grams) : '' }, f.candidates[0]));
  return { ...toDraft(null, nowMs), items: items.length ? items : [{ ...emptyItem(), grams: numStr(scan.net_grams) }] };
};

// Stored unchanged in meals.original_scan_json as provenance (candidates, USDA record ids, measured weight).
export const scanSnapshot = (scan) => ({
  scanId: scan.scan_id, resultId: scan.result_id, scannedAt: scan.timestamp, measuredTotalGrams: scan.measured_total_grams,
  tareG: scan.tare_g, netGrams: scan.net_grams, foods: scan.foods,
});
