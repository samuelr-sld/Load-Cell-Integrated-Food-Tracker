// Manual meal form <-> repository input. Pure (no React), so validation is unit-tested.
// Nutrients are entered per 100 g, exactly as stored; a blank field means unknown (null), never 0.
import { NUTRIENTS, mealTotals } from './meal.js';

export const FORM_CATEGORIES = ['breakfast', 'lunch', 'dinner', 'snack'];
const MAX = { calories: 900, protein: 100, carbs: 100, fat: 100 }; // per 100 g, physical upper bounds
const LABEL = { calories: 'calories', protein: 'protein', carbs: 'carbs', fat: 'fat' };

const pad = (n) => String(n).padStart(2, '0');
// Suggestion only: pre-selects a chip for a NEW meal; the user can always pick another.
export const suggestCategory = (hour) =>
  hour >= 5 && hour < 11 ? 'breakfast' : hour >= 11 && hour < 15 ? 'lunch' : hour >= 17 && hour < 22 ? 'dinner' : 'snack';

// '2:05 PM' (12-hour, shown to the user). 24-hour '14:05' is also accepted when typed.
const fmtTime = (d) => `${d.getHours() % 12 || 12}:${pad(d.getMinutes())} ${d.getHours() < 12 ? 'AM' : 'PM'}`;
const parseTime = (str) => {
  const t = /^(\d{1,2}):(\d{2})\s*([ap]m)?$/i.exec(str.trim());
  if (!t) return null;
  let h = +t[1];
  const m = +t[2];
  if (m > 59) return null;
  if (t[3]) {
    if (h < 1 || h > 12) return null;
    h = (h % 12) + (t[3].toLowerCase() === 'pm' ? 12 : 0);
  } else if (h > 23) return null;
  return [h, m];
};
const numStr = (v) => (v == null ? '' : String(v));
const parseNum = (s) => (String(s).trim() === '' ? null : Number(String(s).trim().replace(',', '.')));

export const emptyItem = () => ({ id: null, name: '', grams: '', calories: '', protein: '', carbs: '', fat: '' });

// Meal (or null for a new one) -> editable strings. Times are shown in the device's local time.
// A new meal captures `nowMs` at the moment the form is opened (call once per open, never on re-render).
export const toDraft = (meal, nowMs = Date.now()) => {
  const at = new Date(meal?.loggedAt ?? nowMs);
  return {
    category: meal ? (FORM_CATEGORIES.includes(meal.category) ? meal.category : null) : suggestCategory(at.getHours()),
    undated: meal ? meal.loggedAt === null : false,
    date: `${at.getFullYear()}-${pad(at.getMonth() + 1)}-${pad(at.getDate())}`,
    time: fmtTime(at),
    items: meal
      ? meal.items.map((it) => ({
        id: it.id, name: it.name, grams: numStr(it.grams),
        ...Object.fromEntries(NUTRIENTS.map((n) => [n, numStr(it.per100g?.[n])])),
      }))
      : [emptyItem()],
  };
};

// Returns { errors, totals, input, patch }. input is for createMeal; patch (when editing `original`)
// carries only editable fields, and the time only if the user changed it (re-parsing the shown
// minute would otherwise drop seconds and could move the meal's day). Unknown item fields
// (e.g. from a camera scan) are kept.
export const parseDraft = (draft, original = null, nowMs = Date.now()) => {
  const errors = [];
  if (!FORM_CATEGORIES.includes(draft.category)) errors.push('Choose breakfast, lunch, dinner or snack.');

  let loggedAt = null;
  if (!draft.undated) {
    const d = /^(\d{4})-(\d{2})-(\d{2})$/.exec(draft.date.trim());
    const t = parseTime(draft.time);
    const at = d && t ? new Date(+d[1], +d[2] - 1, +d[3], t[0], t[1]) : null;
    if (!at || at.getDate() !== +d[3] || at.getMonth() !== +d[2] - 1) {
      errors.push('Enter the date as YYYY-MM-DD and the time as h:mm AM/PM (for example 2:30 PM).');
    } else if (at.getTime() > nowMs + 5 * 60000) {
      errors.push('The meal time cannot be in the future.');
    } else {
      loggedAt = at.getTime();
    }
  }

  if (!draft.items.length) errors.push('Add at least one food item.');
  const byId = new Map((original?.items ?? []).map((it) => [it.id, it]));
  let itemsOk = true;
  const items = draft.items.map((f, i) => {
    const n = `Item ${i + 1}`;
    const before = errors.length;
    const name = f.name.trim();
    if (!name) errors.push(`${n}: enter a food name.`);
    else if (name.length > 80) errors.push(`${n}: the name is too long (80 characters max).`);
    const grams = parseNum(f.grams);
    if (grams === null || !Number.isFinite(grams) || grams <= 0 || grams > 5000) {
      errors.push(`${n}: quantity must be a number of grams between 0 and 5000.`);
    }
    const per100g = {};
    for (const k of NUTRIENTS) {
      const v = parseNum(f[k]);
      if (v !== null && (!Number.isFinite(v) || v < 0 || v > MAX[k])) {
        errors.push(`${n}: ${LABEL[k]} per 100 g must be between 0 and ${MAX[k]} (leave blank if unknown).`);
      }
      per100g[k] = v;
    }
    if (errors.length > before) itemsOk = false;
    const orig = f.id ? byId.get(f.id) : null;
    return { ...(orig ?? {}), ...(f.id ? { id: f.id } : {}), name, grams, per100g: { ...(orig?.per100g ?? {}), ...per100g } };
  });

  const tzOffsetMin = loggedAt === null ? null : -new Date(loggedAt).getTimezoneOffset();
  const input = { loggedAt, tzOffsetMin, category: draft.category, items };
  let patch = null;
  if (original) {
    // Compared at minute precision, so re-saving an untouched time keeps the original instant, offset and dayKey.
    const timeChanged = draft.undated !== (original.loggedAt === null)
      || (!draft.undated && Math.floor(loggedAt / 60000) !== Math.floor(original.loggedAt / 60000));
    patch = { category: draft.category, items, ...(timeChanged ? { loggedAt, tzOffsetMin } : {}) };
  }
  return { errors, totals: itemsOk && items.length ? mealTotals(items) : null, input, patch };
};
