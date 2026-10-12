// Canonical meal domain (pure, no persistence). Schema is a working reading of the
// Phase 1B spec; the spec doc is not in the repo, so verify field names before P3.
//
// Meal: { id, loggedAt (epoch ms | null = undated), dayKey 'YYYY-MM-DD' (local day) | null, category,
//         items: FoodItem[], source: { kind, ref? }, revision: { rev, updatedAt }, deletedAt: ms|null }
// FoodItem: { id, name, grams, per100g: { calories, protein, carbs, fat } } (each nutrient number|null)

export const NUTRIENTS = ['calories', 'protein', 'carbs', 'fat'];
export const CATEGORIES = ['breakfast', 'lunch', 'dinner', 'snack', 'unknown'];
export const SOURCE_KINDS = ['camera', 'manual', 'legacy', 'sample']; // sample = on-device demo data, never synced

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

export const roundTo = (x, dp = 1) => {
  const f = 10 ** dp;
  return Math.round((x + Number.EPSILON) * f) / f;
};

// Local day from an instant. Offset defaults to the device offset at that instant (DST-safe).
export const localDayKey = (ms, offsetMinutes = -new Date(ms).getTimezoneOffset()) =>
  new Date(ms + offsetMinutes * 60000).toISOString().slice(0, 10);

export const addDays = (dayKey, n) => {
  const d = new Date(`${dayKey}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
};

// A nutrient must be a finite number >= 0, or null/undefined (= unknown, never 0).
const knownOrNull = (v, label) => {
  if (v == null) return null;
  if (!isNum(v) || v < 0) throw new RangeError(`${label} must be a finite number >= 0 or null`);
  return v;
};

const scale = (per100g, grams) => {
  if (!isNum(grams) || grams < 0) throw new RangeError('grams must be a finite number >= 0');
  const out = {};
  for (const n of NUTRIENTS) {
    const v = knownOrNull(per100g?.[n], n);
    out[n] = v === null ? null : (v * grams) / 100;
  }
  return out;
};

const roundAll = (o) =>
  Object.fromEntries(NUTRIENTS.map((n) => [n, o[n] === null ? null : roundTo(o[n])]));

export const nutrientsForServing = (per100g, grams) => roundAll(scale(per100g, grams));

// Sums unrounded per-item values and rounds once. A nutrient unknown in any item is unknown for the meal.
export const mealTotals = (items) => {
  if (!items?.length) return Object.fromEntries(NUTRIENTS.map((n) => [n, null]));
  const parts = items.map((it) => scale(it.per100g, it.grams));
  const out = {};
  for (const n of NUTRIENTS) {
    out[n] = parts.some((p) => p[n] === null) ? null : parts.reduce((s, p) => s + p[n], 0);
  }
  return roundAll(out);
};

export const nutritionCompleteness = (totals) => {
  const missing = NUTRIENTS.filter((n) => totals?.[n] == null);
  return { complete: missing.length === 0, missing };
};

export const validateMeal = (m) => {
  const e = [];
  if (!m || typeof m !== 'object') return ['meal must be an object'];
  if (typeof m.id !== 'string' || !m.id) e.push('id required');
  if (m.loggedAt === null) {
    if (m.dayKey !== null) e.push('undated meal must have null dayKey');
  } else {
    if (!isNum(m.loggedAt)) e.push('loggedAt must be epoch ms or null');
    if (!/^\d{4}-\d{2}-\d{2}$/.test(m.dayKey ?? '')) e.push('dayKey must be YYYY-MM-DD');
  }
  if (!CATEGORIES.includes(m.category)) e.push('invalid category');
  if (!SOURCE_KINDS.includes(m.source?.kind)) e.push('invalid source.kind');
  if (!Number.isInteger(m.revision?.rev) || m.revision.rev < 1) e.push('revision.rev must be integer >= 1');
  if (!isNum(m.revision?.updatedAt)) e.push('revision.updatedAt must be epoch ms');
  if (m.deletedAt != null && !isNum(m.deletedAt)) e.push('deletedAt must be null or epoch ms');
  if (!Array.isArray(m.items)) e.push('items must be an array');
  else {
    m.items.forEach((it, i) => {
      if (typeof it?.id !== 'string' || !it.id) e.push(`items[${i}].id required`);
      if (typeof it?.name !== 'string' || !it.name) e.push(`items[${i}].name required`);
      try { scale(it?.per100g, it?.grams); } catch (err) { e.push(`items[${i}]: ${err.message}`); }
    });
  }
  return e;
};

export const softDelete = (meal, now) => ({
  ...meal,
  deletedAt: now,
  revision: { rev: meal.revision.rev + 1, updatedAt: now },
});
