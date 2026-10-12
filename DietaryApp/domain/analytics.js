// Deterministic nutrition engine over canonical meals (the SQLite diary, as returned by the repository).
// Pure: no I/O and no clock (callers pass `today`). Never fabricates data: an unknown nutrient is never
// counted as 0, a day with no logged meals has no value (not 0 kcal), and results name the meals they used.
// Read-only Firestore history is deliberately not an input: SQLite is the source for all calculations.
import { NUTRIENTS, CATEGORIES, addDays, roundTo } from './meal.js';

export const PERIODS = ['day', 'week', 'month'];
export const AVERAGE_BASES = ['logged_days', 'calendar_days'];
const MACRO_KCAL = { carbs: 4, protein: 4, fat: 9 };
const MACROS = Object.keys(MACRO_KCAL);

const utc = (k) => Date.parse(`${k}T00:00:00Z`);
export const isDayKey = (k) =>
  typeof k === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(k) && Number.isFinite(utc(k))
  && new Date(utc(k)).toISOString().slice(0, 10) === k;
export const daysBetween = (from, to) => Math.round((utc(to) - utc(from)) / 86400000);
const minKey = (a, b) => (a < b ? a : b);
const maxKey = (a, b) => (a > b ? a : b);

// Calendar periods in local-day keys. Weeks run Monday to Sunday (ISO 8601).
export const periodRange = (period, dayKey) => {
  if (period === 'day') return { from: dayKey, to: dayKey };
  if (period === 'week') {
    const from = addDays(dayKey, -((new Date(utc(dayKey)).getUTCDay() + 6) % 7));
    return { from, to: addDays(from, 6) };
  }
  if (period === 'month') {
    const d = new Date(utc(`${dayKey.slice(0, 7)}-01`));
    const from = d.toISOString().slice(0, 10);
    d.setUTCMonth(d.getUTCMonth() + 1);
    return { from, to: addDays(d.toISOString().slice(0, 10), -1) };
  }
  throw new RangeError(`unknown period ${period}`);
};

// The period just before `range`: the previous calendar day/week/month, or for a custom range
// the same number of days immediately before it.
export const previousRange = (period, { from, to }) =>
  PERIODS.includes(period)
    ? periodRange(period, addDays(from, -1))
    : { from: addDays(from, -(daysBetween(from, to) + 1)), to: addDays(from, -1) };

// What the stored record says about where its values came from. Camera values are scan estimates;
// `edited` means the record changed after creation on any device. shortcut: meal-level only (the
// schema does not record which field changed), upgrade when scan import stores per-field corrections.
export const provenance = (m) => ({
  source: m.source.kind,
  edited: m.revision.rev > 1 || (m.remoteRev ?? 1) > 1,
});

const missingOf = (m) => NUTRIENTS.filter((n) => m.totals[n] == null);

// Per nutrient: the sum of known values and how many meals lacked it (total is then a lower bound).
const nutrientStats = (meals) => Object.fromEntries(NUTRIENTS.map((n) => {
  const known = meals.filter((m) => m.totals[n] != null);
  return [n, {
    total: known.length ? roundTo(known.reduce((s, m) => s + m.totals[n], 0)) : null,
    knownMeals: known.length,
    missingMeals: meals.length - known.length,
  }];
}));

// Largest-remainder rounding so the shares always add up to exactly 100.
const percentages = (parts) => {
  const sum = Object.values(parts).reduce((a, b) => a + b, 0);
  const raw = Object.entries(parts).map(([k, v]) => [k, (v / sum) * 100]);
  const out = Object.fromEntries(raw.map(([k, v]) => [k, Math.floor(v)]));
  let left = 100 - Object.values(out).reduce((a, b) => a + b, 0);
  for (const [k] of [...raw].sort((a, b) => (b[1] % 1) - (a[1] % 1))) if (left-- > 0) out[k]++;
  return out;
};

// Energy share of carbs/protein/fat (4/4/9 kcal per g), only from meals where all three are known,
// so a meal missing fat cannot inflate the carb share.
export const macroDistribution = (meals) => {
  const used = meals.filter((m) => MACROS.every((n) => m.totals[n] != null));
  const grams = Object.fromEntries(MACROS.map((n) => [n, roundTo(used.reduce((s, m) => s + m.totals[n], 0))]));
  const kcal = Object.fromEntries(MACROS.map((n) => [n, roundTo(grams[n] * MACRO_KCAL[n])]));
  const any = Object.values(kcal).some((v) => v > 0);
  return { basedOnMeals: used.length, excludedMeals: meals.length - used.length, grams, kcal, pct: any ? percentages(kcal) : null };
};

const liveInRange = (meals, from, to) => meals
  .filter((m) => m.deletedAt == null && m.dayKey != null && m.dayKey >= from && m.dayKey <= to)
  .sort((a, b) => a.loggedAt - b.loggedAt);

// Summary of [from, to] (inclusive local-day keys). `meals` may include undated meals; they belong to
// no period and are only counted. Averages divide by logged days or by elapsed calendar days (days
// after `today` have not happened yet and are not counted as empty).
export const summarize = (meals, { from, to, today, averageBasis = 'logged_days', goals = null }) => {
  if (!AVERAGE_BASES.includes(averageBasis)) throw new RangeError(`unknown averageBasis ${averageBasis}`);
  const inRange = liveInRange(meals, from, to);
  const logged = [...new Set(inRange.map((m) => m.dayKey))].sort();
  const end = minKey(to, maxKey(today, logged[logged.length - 1] ?? today));
  const elapsedDays = end < from ? 0 : daysBetween(from, end) + 1;
  const nutrients = nutrientStats(inRange);
  const divisor = averageBasis === 'calendar_days' ? elapsedDays : logged.length;
  const per = (d) => Object.fromEntries(NUTRIENTS.map((n) => [n, nutrients[n].total == null || !d ? null : roundTo(nutrients[n].total / d)]));
  const perDay = per(divisor);
  const unloggedDays = [];
  for (let k = from; k <= end; k = addDays(k, 1)) if (!logged.includes(k)) unloggedDays.push(k);
  const prov = inRange.map((m) => [m.id, provenance(m)]);
  const ids = (pred) => prov.filter(([, p]) => pred(p)).map(([id]) => id);

  const summary = {
    from, to, elapsedDays, loggedDays: logged.length, mealCount: inRange.length,
    nutrients,
    averages: {
      basis: averageBasis,
      perDay,
      perWeek: elapsedDays >= 7 ? per(elapsedDays / 7) : null, // calendar weeks elapsed
    },
    macros: macroDistribution(inRange),
    byMealType: CATEGORIES.map((c) => {
      const ms = inRange.filter((m) => m.category === c);
      return { category: c, mealCount: ms.length, nutrients: nutrientStats(ms) };
    }).filter((b) => b.mealCount > 0),
    completeness: {
      completeMeals: inRange.filter((m) => !missingOf(m).length).length,
      incompleteMeals: inRange.filter((m) => missingOf(m).length).map((m) => ({ id: m.id, dayKey: m.dayKey, missing: missingOf(m) })),
      unloggedDays,
      undatedMealsExcluded: meals.filter((m) => m.deletedAt == null && m.dayKey == null).length,
    },
    evidence: {
      mealIds: inRange.map((m) => m.id),
      estimatedMealIds: ids((p) => p.source === 'camera' && !p.edited),
      userEditedScanMealIds: ids((p) => p.source === 'camera' && p.edited),
      sampleMealIds: ids((p) => p.source === 'sample'),
    },
  };
  summary.goals = goals ? goalProgress(summary, goals) : null;
  return summary;
};

// Goal vs actual: the day's total for a single day, otherwise the average per day. `lowerBound`
// means some meals lacked the nutrient, so the real figure may be higher. A comparison, not advice.
export const goalProgress = (summary, goals) => Object.fromEntries(NUTRIENTS.filter((n) => goals[n] != null).map((n) => {
  const single = summary.from === summary.to;
  const actual = single ? summary.nutrients[n].total : summary.averages.perDay[n];
  return [n, {
    goal: goals[n],
    actual,
    basis: single ? 'day_total' : `average_per_${summary.averages.basis === 'calendar_days' ? 'calendar' : 'logged'}_day`,
    pctOfGoal: actual == null ? null : Math.round((actual / goals[n]) * 100),
    lowerBound: actual != null && summary.nutrients[n].missingMeals > 0,
  }];
}));

// Per-day averages of two summaries. Not comparable when either side lacks data or is a lower bound.
export const comparePeriods = (current, previous) => ({
  current: { from: current.from, to: current.to, mealCount: current.mealCount },
  previous: { from: previous.from, to: previous.to, mealCount: previous.mealCount },
  basis: current.averages.basis,
  nutrients: Object.fromEntries(NUTRIENTS.map((n) => {
    const a = current.averages.perDay[n];
    const b = previous.averages.perDay[n];
    const comparable = a != null && b != null && !current.nutrients[n].missingMeals && !previous.nutrients[n].missingMeals;
    return [n, {
      current: a, previous: b, comparable,
      change: comparable ? roundTo(a - b) : null,
      changePct: comparable && b > 0 ? Math.round(((a - b) / b) * 100) : null,
    }];
  })),
});

// ---- Patterns: only claims the stored record supports, each with the meals/days behind it.

export const PATTERN_TYPES = ['frequent_foods', 'meal_type_logging', 'goal_adherence', 'exclusion_matches'];
export const UNSUPPORTED_PATTERNS = {
  preparation_method: 'Preparation method (fried, grilled, steamed …) is not recorded and is never inferred from fat content or food names.',
  food_groups: 'Food groups and ingredients are not recorded.',
  micronutrients: 'Only calories, protein, carbohydrates and fat are recorded.',
  diet_compliance: 'Ingredients are not recorded, so meals cannot be checked against a diet.',
};
const MIN_SUPPORT = 2; // one meal is an observation, not a pattern
const norm = (s) => String(s).trim().toLowerCase().replace(/\s+/g, ' ');
// ' peanut butter toast ' style word list, so matching is whole-word and works for any alphabet.
const words = (s) => ` ${String(s).toLowerCase().split(/[\s.,;:!?()[\]{}"'/\\&+*_-]+/).filter(Boolean).join(' ')} `;

const frequentFoods = (meals) => {
  const byFood = new Map();
  for (const m of meals) for (const f of new Set(m.items.map((i) => norm(i.name)))) byFood.set(f, [...(byFood.get(f) ?? []), m]);
  return [...byFood].filter(([, ms]) => ms.length >= MIN_SUPPORT)
    .sort((a, b) => b[1].length - a[1].length || (a[0] < b[0] ? -1 : 1)).slice(0, 10)
    .map(([food, ms]) => ({
      type: 'frequent_foods', food, mealCount: ms.length, dayCount: new Set(ms.map((m) => m.dayKey)).size,
      basis: 'stored_food_names', evidence: { mealIds: ms.map((m) => m.id) },
    }));
};

// "Logged", never "eaten" or "skipped": a missing log is not evidence of a skipped meal.
const mealTypeLogging = (meals) => {
  const loggedDays = new Set(meals.map((m) => m.dayKey)).size;
  return CATEGORIES.filter((c) => c !== 'unknown').map((c) => {
    const ms = meals.filter((m) => m.category === c);
    const days = [...new Set(ms.map((m) => m.dayKey))].sort();
    return { type: 'meal_type_logging', category: c, daysLogged: days.length, loggedDays, basis: 'stored_meal_category', evidence: { dayKeys: days, mealIds: ms.map((m) => m.id) } };
  }).filter((p) => loggedDays > 0);
};

// Days above vs at-or-below a goal, judged only on days where every meal has the nutrient.
const goalAdherence = (meals, goals) => NUTRIENTS.filter((n) => goals?.[n] != null).map((n) => {
  const above = []; const within = []; const excluded = [];
  for (const k of [...new Set(meals.map((m) => m.dayKey))].sort()) {
    const day = meals.filter((m) => m.dayKey === k);
    if (day.some((m) => m.totals[n] == null)) excluded.push(k);
    else (day.reduce((s, m) => s + m.totals[n], 0) > goals[n] ? above : within).push(k);
  }
  return {
    type: 'goal_adherence', nutrient: n, goal: goals[n], daysAbove: above.length, daysAtOrBelow: within.length,
    daysExcludedIncomplete: excluded.length, basis: 'stored_daily_totals',
    evidence: { aboveDayKeys: above, atOrBelowDayKeys: within, excludedDayKeys: excluded },
  };
}).filter((p) => p.daysAbove + p.daysAtOrBelow > 0);

// Whole-word match of the user's own exclusion terms against stored food names. It reports a name
// match only; it does not claim the food contains (or does not contain) the excluded ingredient.
const exclusionMatches = (meals, exclusions = []) => exclusions.map((term) => {
  const hit = (name) => words(term).trim() !== '' && words(name).includes(words(term));
  const ms = meals.filter((m) => m.items.some((i) => hit(i.name)));
  return {
    type: 'exclusion_matches', term: norm(term), mealCount: ms.length, basis: 'food_name_contains_term',
    evidence: { mealIds: ms.map((m) => m.id), matchedNames: [...new Set(ms.flatMap((m) => m.items.filter((i) => hit(i.name)).map((i) => norm(i.name))))] },
  };
}).filter((p) => p.mealCount > 0);

export const detectPatterns = (meals, { from, to, types = PATTERN_TYPES, preferences = null }) => {
  const bad = types.filter((t) => !PATTERN_TYPES.includes(t));
  if (bad.length) throw new RangeError(`unsupported pattern type: ${bad.join(', ')}`);
  const ms = liveInRange(meals, from, to);
  const run = { frequent_foods: () => frequentFoods(ms), meal_type_logging: () => mealTypeLogging(ms),
    goal_adherence: () => goalAdherence(ms, preferences?.goals), exclusion_matches: () => exclusionMatches(ms, preferences?.exclusions) };
  return types.flatMap((t) => run[t]());
};

// ---- Charts: every value comes from summarize(), the same engine as the text summaries.

export const CHART_TYPES = ['bar', 'line', 'pie'];
export const GROUP_BY = ['day', 'week', 'month'];
const UNIT = { calories: 'kcal', protein: 'g', carbs: 'g', fat: 'g' };

// Bar/line: one point per day (that day's total) or per week/month (average per day, so partial
// buckets stay comparable). A bucket with no logged meals has value null, never 0.
// Pie: macro energy distribution for the whole range.
export const chartData = (meals, { type, metric, groupBy = 'day', from, to, today, averageBasis, goals = null }) => {
  if (!CHART_TYPES.includes(type)) throw new RangeError(`unknown chart type ${type}`);
  if (type === 'pie') {
    const { macros } = summarize(meals, { from, to, today, averageBasis });
    return {
      type, from, to, unit: 'kcal', basedOnMeals: macros.basedOnMeals, excludedMeals: macros.excludedMeals,
      slices: macros.pct && MACROS.map((k) => ({ key: k, grams: macros.grams[k], kcal: macros.kcal[k], pct: macros.pct[k] })),
    };
  }
  if (!NUTRIENTS.includes(metric)) throw new RangeError(`unknown metric ${metric}`);
  if (!GROUP_BY.includes(groupBy)) throw new RangeError(`unknown groupBy ${groupBy}`);
  const points = [];
  for (let k = from; k <= to;) {
    const r = groupBy === 'day' ? { from: k, to: k } : periodRange(groupBy, k);
    const b = { from: maxKey(r.from, from), to: minKey(r.to, to) };
    const s = summarize(meals, { ...b, today, averageBasis });
    const value = groupBy === 'day' ? s.nutrients[metric].total : s.averages.perDay[metric];
    points.push({ key: b.from, ...b, value, mealCount: s.mealCount, lowerBound: value != null && s.nutrients[metric].missingMeals > 0 });
    k = addDays(r.to, 1);
  }
  return {
    type, metric, unit: UNIT[metric], groupBy, valueKind: groupBy === 'day' ? 'total' : 'daily_average', from, to,
    points, goal: goals?.[metric] ?? null,
  };
};
