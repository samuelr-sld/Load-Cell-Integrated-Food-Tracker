import test from 'node:test';
import assert from 'node:assert/strict';
import { mealTotals, localDayKey } from './meal.js';
import {
  periodRange, previousRange, summarize, comparePeriods, detectPatterns, chartData, macroDistribution, isDayKey,
} from './analytics.js';
import { normalizePreferences, DEFAULT_PREFERENCES } from './preferences.js';

const P = { calories: 100, protein: 10, carbs: 10, fat: 5 }; // per 100 g
let n = 0;
// Canonical meal as the repository returns it (totals derived from items, like the stored columns).
const meal = (dayKey, { grams = 100, per100g = P, name = 'rice', category = 'lunch', kind = 'manual', rev = 1, remoteRev = null, deletedAt = null, items } = {}) => {
  const its = items ?? [{ id: `i${++n}`, name, grams, per100g }];
  return {
    id: `m${++n}`, dayKey, loggedAt: dayKey ? Date.parse(`${dayKey}T12:00:00Z`) + n : null, tzOffsetMin: 0,
    category, items: its, totals: mealTotals(its), source: { kind }, revision: { rev, updatedAt: 1 }, remoteRev, deletedAt,
  };
};
const S = (meals, from, to, opts = {}) => summarize(meals, { from, to, today: '2026-10-11', ...opts });

test('periods: ISO weeks (Mon-Sun), calendar months, leap years, year boundaries', () => {
  assert.deepEqual(periodRange('day', '2026-10-11'), { from: '2026-10-11', to: '2026-10-11' });
  assert.deepEqual(periodRange('week', '2026-10-11'), { from: '2026-10-05', to: '2026-10-11' }); // Sunday
  assert.deepEqual(periodRange('week', '2026-10-05'), { from: '2026-10-05', to: '2026-10-11' }); // Monday
  assert.deepEqual(periodRange('week', '2027-01-01'), { from: '2026-12-28', to: '2027-01-03' });
  assert.deepEqual(periodRange('month', '2028-02-15'), { from: '2028-02-01', to: '2028-02-29' });
  assert.deepEqual(periodRange('month', '2026-02-15'), { from: '2026-02-01', to: '2026-02-28' });
  assert.deepEqual(periodRange('month', '2026-12-31'), { from: '2026-12-01', to: '2026-12-31' });
  assert.deepEqual(previousRange('month', periodRange('month', '2026-03-31')), { from: '2026-02-01', to: '2026-02-28' });
  assert.deepEqual(previousRange('week', periodRange('week', '2026-10-11')), { from: '2026-09-28', to: '2026-10-04' });
  assert.deepEqual(previousRange('custom', { from: '2026-10-08', to: '2026-10-10' }), { from: '2026-10-05', to: '2026-10-07' });
  assert.ok(isDayKey('2028-02-29'));
  assert.ok(!isDayKey('2026-02-29') && !isDayKey('2026-13-01') && !isDayKey('26-1-1') && !isDayKey(null));
});

test('daily, weekly, monthly and custom summaries sum only meals inside the range', () => {
  const ms = [meal('2026-10-04', { grams: 900 }), meal('2026-10-05'), meal('2026-10-05', { grams: 200 }), meal('2026-10-11'), meal('2026-09-30', { grams: 50 })];
  assert.equal(S(ms, '2026-10-05', '2026-10-05').nutrients.calories.total, 300);
  const w = S(ms, ...Object.values(periodRange('week', '2026-10-11')));
  assert.equal(w.mealCount, 3);
  assert.equal(w.nutrients.calories.total, 400);
  assert.deepEqual(w.evidence.mealIds, [ms[1].id, ms[2].id, ms[3].id]); // which meals made the total
  assert.equal(S(ms, '2026-10-01', '2026-10-31').nutrients.calories.total, 1300);
  assert.equal(S(ms, '2026-09-30', '2026-10-04').nutrients.calories.total, 950);
});

test('averages: per logged day or per elapsed calendar day; future days are not counted as empty', () => {
  const ms = [meal('2026-10-05', { grams: 300 }), meal('2026-10-07', { grams: 100 })];
  const logged = S(ms, '2026-10-05', '2026-10-11');
  assert.equal(logged.averages.perDay.calories, 200); // 400 / 2 logged days
  assert.equal(logged.averages.perWeek.calories, 400);
  const cal = S(ms, '2026-10-05', '2026-10-11', { averageBasis: 'calendar_days' });
  assert.equal(cal.averages.perDay.calories, 57.1); // 400 / 7
  const month = S(ms, '2026-10-01', '2026-10-31', { averageBasis: 'calendar_days' });
  assert.equal(month.elapsedDays, 11); // Oct 1..11 (today), not 31
  assert.equal(month.averages.perDay.calories, 36.4);
  assert.deepEqual(S(ms, '2026-10-05', '2026-10-11').completeness.unloggedDays, ['2026-10-06', '2026-10-08', '2026-10-09', '2026-10-10', '2026-10-11']);
  assert.throws(() => S(ms, '2026-10-05', '2026-10-11', { averageBasis: 'weird' }), RangeError);
});

test('empty dataset: nothing is fabricated', () => {
  const s = S([], '2026-10-05', '2026-10-11');
  assert.equal(s.mealCount, 0);
  assert.equal(s.nutrients.calories.total, null);
  assert.equal(s.averages.perDay.calories, null);
  assert.equal(s.macros.pct, null);
  assert.deepEqual(s.byMealType, []);
  assert.equal(s.completeness.unloggedDays.length, 7);
  const future = S([], '2026-11-01', '2026-11-30');
  assert.equal(future.elapsedDays, 0);
  assert.deepEqual(future.completeness.unloggedDays, []);
});

test('partial nutrition: unknown is never zero, known zero stays zero, totals become lower bounds', () => {
  const partial = meal('2026-10-10', { per100g: { calories: 200, protein: null, carbs: 0, fat: null } });
  const full = meal('2026-10-10');
  const s = S([partial, full], '2026-10-10', '2026-10-10');
  assert.deepEqual(s.nutrients.calories, { total: 300, knownMeals: 2, missingMeals: 0 });
  assert.deepEqual(s.nutrients.protein, { total: 10, knownMeals: 1, missingMeals: 1 });
  assert.equal(s.nutrients.carbs.total, 10);
  assert.deepEqual(s.completeness.incompleteMeals, [{ id: partial.id, dayKey: '2026-10-10', missing: ['protein', 'fat'] }]);
  assert.equal(s.completeness.completeMeals, 1);
  const allUnknown = S([meal('2026-10-10', { per100g: { calories: null, protein: null, carbs: null, fat: null } })], '2026-10-10', '2026-10-10');
  assert.equal(allUnknown.nutrients.protein.total, null);
  assert.equal(allUnknown.averages.perDay.protein, null);
  const zero = S([meal('2026-10-10', { per100g: { calories: 0, protein: 0, carbs: 0, fat: 0 } })], '2026-10-10', '2026-10-10');
  assert.equal(zero.nutrients.calories.total, 0);
  assert.equal(zero.macros.pct, null); // 0 kcal has no distribution
});

test('macro distribution uses only meals with all three macros and always sums to 100', () => {
  const m = macroDistribution([meal('d', { per100g: { calories: 1, protein: 10, carbs: 10, fat: 10 } }), meal('d', { per100g: { calories: 1, protein: null, carbs: 90, fat: 0 } })]);
  assert.equal(m.basedOnMeals, 1);
  assert.equal(m.excludedMeals, 1);
  assert.deepEqual(m.kcal, { carbs: 40, protein: 40, fat: 90 });
  assert.deepEqual(m.pct, { carbs: 24, protein: 23, fat: 53 }); // 23.53/23.53/52.94 -> largest remainder
  const thirds = macroDistribution([meal('d', { per100g: { calories: 1, protein: 1, carbs: 1, fat: 4 / 9 } })]);
  assert.equal(Object.values(thirds.pct).reduce((a, b) => a + b), 100);
});

test('meal-type breakdown and provenance: estimates, user-edited scans, samples', () => {
  const scan = meal('2026-10-10', { kind: 'camera', category: 'dinner' });
  const editedScan = meal('2026-10-10', { kind: 'camera', rev: 3, category: 'dinner' });
  const pulledEdited = meal('2026-10-10', { kind: 'camera', remoteRev: 2, category: 'dinner' }); // edited on another device
  const sample = meal('2026-10-10', { kind: 'sample', category: 'breakfast' });
  const s = S([scan, editedScan, pulledEdited, sample, meal('2026-10-10')], '2026-10-10', '2026-10-10');
  assert.deepEqual(s.byMealType.map((b) => [b.category, b.mealCount, b.nutrients.calories.total]), [['breakfast', 1, 100], ['lunch', 1, 100], ['dinner', 3, 300]]);
  assert.deepEqual(s.evidence.estimatedMealIds, [scan.id]);
  assert.deepEqual(s.evidence.userEditedScanMealIds, [editedScan.id, pulledEdited.id]);
  assert.deepEqual(s.evidence.sampleMealIds, [sample.id]); // samples are included but always flagged
});

test('deleted and undated meals are excluded; undated are counted', () => {
  const s = S([meal('2026-10-10'), meal('2026-10-10', { deletedAt: 5 }), meal(null), meal(null, { deletedAt: 5 })], '2026-10-01', '2026-10-31');
  assert.equal(s.mealCount, 1);
  assert.equal(s.completeness.undatedMealsExcluded, 1);
});

test('midnight rollover and timezone: meals sit on the local day of their stored offset', () => {
  const at = (iso, off) => ({ ...meal(localDayKey(Date.parse(iso), off)), loggedAt: Date.parse(iso), tzOffsetMin: off });
  const late = at('2026-10-10T15:59:00Z', 480); // 23:59 at +08:00
  const early = at('2026-10-10T16:00:00Z', 480); // 00:00 next day at +08:00
  assert.equal(S([late, early], '2026-10-10', '2026-10-10').mealCount, 1);
  assert.equal(S([late, early], '2026-10-11', '2026-10-11').mealCount, 1);
  // Sunday 23:59 belongs to the old week, Monday 00:00 to the new one.
  const sun = at('2026-10-04T15:59:00Z', 480);
  const mon = at('2026-10-04T16:00:00Z', 480);
  assert.equal(S([sun, mon], ...Object.values(periodRange('week', '2026-10-05'))).mealCount, 1);
  // Month boundary in a negative offset: 2026-10-01T03:00Z is still Sept 30 at -05:00.
  const ny = at('2026-10-01T03:00:00Z', -300);
  assert.equal(ny.dayKey, '2026-09-30');
  assert.equal(S([ny], '2026-10-01', '2026-10-31').mealCount, 0);
});

test('goals: day total vs goal, average vs goal, lower bound flagged', () => {
  const goals = { ...DEFAULT_PREFERENCES.goals, calories: 2000, protein: 50 };
  const ms = [meal('2026-10-10', { grams: 1500 }), meal('2026-10-11', { grams: 500, per100g: { ...P, protein: null } })];
  const day = S(ms, '2026-10-10', '2026-10-10', { goals });
  assert.deepEqual(day.goals.calories, { goal: 2000, actual: 1500, basis: 'day_total', pctOfGoal: 75, lowerBound: false });
  assert.equal(day.goals.carbs, undefined); // no goal set, nothing reported
  const wk = S(ms, '2026-10-05', '2026-10-11', { goals });
  assert.equal(wk.goals.calories.actual, 1000);
  assert.equal(wk.goals.calories.basis, 'average_per_logged_day');
  assert.equal(wk.goals.protein.lowerBound, true);
  assert.equal(S([], '2026-10-10', '2026-10-10', { goals }).goals.calories.actual, null);
});

test('comparison: per-day averages; incomplete or empty sides are not comparable', () => {
  const cur = S([meal('2026-10-10', { grams: 300 })], '2026-10-05', '2026-10-11');
  const prev = S([meal('2026-10-01', { grams: 200 })], '2026-09-28', '2026-10-04');
  const c = comparePeriods(cur, prev);
  assert.deepEqual(c.nutrients.calories, { current: 300, previous: 200, comparable: true, change: 100, changePct: 50 });
  const partial = S([meal('2026-10-01', { per100g: { ...P, fat: null } })], '2026-09-28', '2026-10-04');
  assert.equal(comparePeriods(cur, partial).nutrients.fat.comparable, false);
  assert.equal(comparePeriods(cur, partial).nutrients.fat.change, null);
  assert.equal(comparePeriods(cur, S([], '2026-09-28', '2026-10-04')).nutrients.calories.comparable, false);
});

test('patterns are backed by evidence and need at least two meals', () => {
  const r1 = meal('2026-10-06', { name: ' Fried  Chicken ', items: undefined });
  const r2 = meal('2026-10-07', { name: 'fried chicken' });
  const once = meal('2026-10-07', { name: 'adobo', category: 'dinner' });
  const ps = detectPatterns([r1, r2, once], { from: '2026-10-05', to: '2026-10-11' });
  const ff = ps.filter((p) => p.type === 'frequent_foods');
  assert.deepEqual(ff.map((p) => [p.food, p.mealCount, p.dayCount]), [['fried chicken', 2, 2]]);
  assert.deepEqual(ff[0].evidence.mealIds, [r1.id, r2.id]);
  const lunch = ps.find((p) => p.type === 'meal_type_logging' && p.category === 'lunch');
  assert.deepEqual([lunch.daysLogged, lunch.loggedDays], [2, 2]);
  // A name containing "fried" yields a frequency fact only, never a preparation-method claim.
  assert.ok(ps.every((p) => !/prep|cook|fried/.test(p.type)));
  assert.throws(() => detectPatterns([r1], { from: 'a', to: 'z', types: ['preparation_method'] }), /unsupported pattern/);
});

test('goal adherence ignores incomplete days; exclusion matches whole words of stored names', () => {
  const preferences = { goals: { ...DEFAULT_PREFERENCES.goals, calories: 150 }, exclusions: ['peanut', 'rice', 'café'] };
  const ms = [
    meal('2026-10-06', { grams: 200, name: 'Peanut-butter toast' }), // 200 > 150
    meal('2026-10-07', { grams: 100, name: 'licorice' }), // 100 <= 150, and "rice" must not match "licorice"
    meal('2026-10-08', { per100g: { ...P, calories: null }, name: 'fried rice' }),
  ];
  ms[1].items.push({ id: 'cafe', name: 'Café latte', grams: 0, per100g: P }); // non-ASCII whole word
  const ps = detectPatterns(ms, { from: '2026-10-05', to: '2026-10-11', types: ['goal_adherence', 'exclusion_matches'], preferences });
  const g = ps.find((p) => p.type === 'goal_adherence');
  assert.deepEqual([g.daysAbove, g.daysAtOrBelow, g.daysExcludedIncomplete], [1, 1, 1]);
  assert.deepEqual(g.evidence.excludedDayKeys, ['2026-10-08']);
  const ex = ps.filter((p) => p.type === 'exclusion_matches');
  assert.deepEqual(ex.map((p) => [p.term, p.evidence.mealIds, p.evidence.matchedNames]), [
    ['peanut', [ms[0].id], ['peanut-butter toast']],
    ['rice', [ms[2].id], ['fried rice']],
    ['café', [ms[1].id], ['café latte']],
  ]);
  assert.deepEqual(detectPatterns(ms, { from: '2026-10-05', to: '2026-10-11', types: ['goal_adherence'] }), []); // no goals
});

test('chart datasets come from the summary engine; empty buckets are null, not zero', () => {
  const ms = [meal('2026-10-05', { grams: 300 }), meal('2026-10-07', { per100g: { ...P, protein: null } }), meal('2026-10-13')];
  const bar = chartData(ms, { type: 'bar', metric: 'calories', from: '2026-10-05', to: '2026-10-11', today: '2026-10-11', goals: { calories: 2000 } });
  assert.equal(bar.points.length, 7);
  assert.deepEqual(bar.points.map((p) => p.value), [300, null, 100, null, null, null, null]);
  assert.equal(bar.valueKind, 'total');
  assert.equal(bar.goal, 2000);
  const prot = chartData(ms, { type: 'line', metric: 'protein', from: '2026-10-05', to: '2026-10-07', today: '2026-10-11' });
  assert.deepEqual(prot.points.map((p) => [p.value, p.lowerBound]), [[30, false], [null, false], [null, false]]);
  const wk = chartData(ms, { type: 'bar', metric: 'calories', groupBy: 'week', from: '2026-10-01', to: '2026-10-31', today: '2026-10-31' });
  assert.deepEqual(wk.points.map((p) => [p.from, p.to]), [['2026-10-01', '2026-10-04'], ['2026-10-05', '2026-10-11'], ['2026-10-12', '2026-10-18'], ['2026-10-19', '2026-10-25'], ['2026-10-26', '2026-10-31']]);
  assert.equal(wk.points[1].value, S(ms, '2026-10-05', '2026-10-11').averages.perDay.calories); // same engine
  const mo = chartData(ms, { type: 'bar', metric: 'fat', groupBy: 'month', from: '2026-09-15', to: '2026-10-31', today: '2026-10-31' });
  assert.equal(mo.points.length, 2);
  const pie = chartData(ms, { type: 'pie', from: '2026-10-05', to: '2026-10-11', today: '2026-10-11' });
  assert.deepEqual(pie.slices.map((x) => x.key), ['carbs', 'protein', 'fat']);
  assert.equal(pie.slices.reduce((a, x) => a + x.pct, 0), 100);
  assert.equal(pie.excludedMeals, 1);
  assert.equal(chartData([], { type: 'pie', from: '2026-10-05', to: '2026-10-11', today: '2026-10-11' }).slices, null);
  assert.throws(() => chartData(ms, { type: 'bar', metric: 'sugar', from: 'a', to: 'b' }), RangeError);
});

test('preferences validation and normalization', () => {
  const ok = normalizePreferences({ goals: { calories: 2000 }, diets: ['vegan'], exclusions: [' Peanut  Butter ', 'peanut butter', ''], averageBasis: 'calendar_days' });
  assert.deepEqual(ok.errors, []);
  assert.deepEqual(ok.value, { goals: { calories: 2000, protein: null, carbs: null, fat: null }, diets: ['vegan'], exclusions: ['peanut butter'], averageBasis: 'calendar_days' });
  assert.deepEqual(normalizePreferences({}).value, DEFAULT_PREFERENCES);
  for (const bad of [
    { goals: { calories: -1 } }, { goals: { calories: 0 } }, { goals: { calories: '2000' } }, { goals: { sugar: 5 } },
    { diets: ['keto_cure'] }, { exclusions: 'nuts' }, { exclusions: ['x'.repeat(41)] }, { exclusions: Array.from({ length: 31 }, (_, i) => `f${i}`) },
    { averageBasis: 'weekly' }, { medicalCondition: 'diabetes' }, null,
  ]) assert.ok(normalizePreferences(bad).errors.length, JSON.stringify(bad));
});
