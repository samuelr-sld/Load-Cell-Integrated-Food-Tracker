// Deterministic callers for the nutrition tools, against real SQLite (node:sqlite) and the real repository.
import test from 'node:test';
import assert from 'node:assert/strict';
import { openNodeDb } from '../data/nodeSqliteAdapter.mjs';
import { migrate } from '../data/schema.js';
import { createMealRepository } from '../data/mealRepository.js';
import { createNutritionTools, TOOL_DEFINITIONS } from './nutritionTools.js';

const TODAY = '2026-10-11'; // a Sunday
const food = (name, grams, calories, protein, carbs, fat) => ({ name, grams, per100g: { calories, protein, carbs, fat } });
const at = (day, hourUtc = 12) => Date.parse(`${day}T${String(hourUtc).padStart(2, '0')}:00:00Z`);

// Fixture: one user's week plus the previous week, with a partial meal, a deleted meal, an undated meal,
// a scan, a sample, and a second user whose meals must never appear.
const setup = async () => {
  const db = openNodeDb();
  await migrate(db);
  const repo = createMealRepository(db);
  await repo.setIdentity('uidB');
  await repo.createMeal({ loggedAt: at('2026-10-10'), tzOffsetMin: 0, items: [food('secret B food', 100, 999, 1, 1, 1)], source: { kind: 'manual' } });
  await repo.setIdentity('uidA');
  const mk = (day, category, items, source = { kind: 'manual' }, extra = {}) =>
    repo.createMeal({ loggedAt: day ? at(day) : null, tzOffsetMin: 0, category, items, source, ...extra });
  const ids = {
    rice1: (await mk('2026-10-06', 'lunch', [food('Rice', 200, 130, 2.7, 28, 0.3)])).id,
    rice2: (await mk('2026-10-08', 'dinner', [food('rice', 150, 130, 2.7, 28, 0.3), food('Fried chicken', 100, 250, 20, 10, 15)])).id,
    partial: (await mk('2026-10-09', 'breakfast', [food('mystery bowl', 300, 120, null, null, null)])).id,
    scan: (await mk('2026-10-10', 'dinner', [food('adobo', 250, 200, 15, 3, 14)], { kind: 'camera', ref: 'scan-1' }, { originalScan: { label: 'adobo' } })).id,
    sample: (await mk('2026-10-10', 'snack', [food('SAMPLE oatmeal', 250, 70, 2.5, 12, 1.5)], { kind: 'sample' })).id,
    prev: (await mk('2026-09-30', 'lunch', [food('rice', 100, 130, 2.7, 28, 0.3)])).id,
  };
  const gone = await mk('2026-10-07', 'lunch', [food('deleted cake', 500, 400, 5, 50, 20)]);
  await repo.deleteMeal(gone.id, 1);
  await mk(null, 'snack', [food('undated apple', 100, 52, 0.3, 14, 0.2)]);
  return { db, repo, ids, tools: createNutritionTools({ repo, today: () => TODAY }) };
};

test('definitions are complete, JSON-schema shaped and closed to extra arguments', () => {
  assert.deepEqual(TOOL_DEFINITIONS.map((d) => d.name), [
    'get_nutrition_summary', 'get_food_patterns', 'compare_periods', 'get_meal_history', 'render_nutrition_chart', 'get_user_preferences',
  ]);
  for (const d of TOOL_DEFINITIONS) assert.equal(d.parameters.type, 'object');
});

test('get_nutrition_summary: weekly totals with evidence, completeness and provenance; user isolation', async () => {
  const { tools, ids } = await setup();
  const r = await tools.call('get_nutrition_summary', { period: 'week' });
  assert.equal(r.ok, true, JSON.stringify(r.error));
  const s = r.data;
  assert.deepEqual([s.from, s.to], ['2026-10-05', '2026-10-11']);
  assert.deepEqual(new Set(s.evidence.mealIds), new Set([ids.rice1, ids.rice2, ids.partial, ids.scan, ids.sample]));
  assert.equal(s.nutrients.calories.total, 260 + 195 + 250 + 360 + 500 + 175); // deleted cake and user B excluded
  assert.deepEqual(s.nutrients.protein.missingMeals, 1);
  assert.deepEqual(s.completeness.incompleteMeals, [{ id: ids.partial, dayKey: '2026-10-09', missing: ['protein', 'carbs', 'fat'] }]);
  assert.equal(s.completeness.undatedMealsExcluded, 1);
  assert.deepEqual(s.evidence.estimatedMealIds, [ids.scan]);
  assert.deepEqual(s.evidence.sampleMealIds, [ids.sample]);
  assert.doesNotMatch(JSON.stringify(r), /secret B food|uidA|uidB|scan-1/); // no other user, owner id or scan ref
  assert.equal((await tools.call('get_nutrition_summary', { period: 'day', date: '2026-10-09' })).data.nutrients.protein.total, null);
  assert.equal((await tools.call('get_nutrition_summary', { period: 'month' })).data.mealCount, 5);
  assert.equal((await tools.call('get_nutrition_summary', { period: 'custom', from: '2026-09-30', to: '2026-10-06' })).data.mealCount, 2);
});

test('summary applies stored goals and averaging preference', async () => {
  const { tools, repo } = await setup();
  await repo.savePreferences(0, { goals: { calories: 2000 }, averageBasis: 'calendar_days' });
  const s = (await tools.call('get_nutrition_summary', { period: 'week' })).data;
  assert.equal(s.averages.basis, 'calendar_days');
  assert.equal(s.goals.calories.actual, s.averages.perDay.calories);
  assert.equal(s.goals.calories.basis, 'average_per_calendar_day');
});

test('get_food_patterns: evidence-backed only; unsupported claims are refused with a reason', async () => {
  const { tools, repo, ids } = await setup();
  await repo.savePreferences(0, { exclusions: ['chicken'] });
  const r = await tools.call('get_food_patterns', { period: 'week' });
  assert.equal(r.ok, true);
  const rice = r.data.patterns.find((p) => p.type === 'frequent_foods');
  assert.deepEqual([rice.food, rice.evidence.mealIds], ['rice', [ids.rice1, ids.rice2]]);
  const ex = r.data.patterns.find((p) => p.type === 'exclusion_matches');
  assert.deepEqual([ex.term, ex.evidence.mealIds, ex.basis], ['chicken', [ids.rice2], 'food_name_contains_term']);
  for (const type of ['preparation_method', 'diet_compliance']) {
    const bad = await tools.call('get_food_patterns', { period: 'week', types: [type] });
    assert.equal(bad.error.code, 'unsupported_claim');
  }
  assert.equal((await tools.call('get_food_patterns', { period: 'week', types: ['fried_food_lover'] })).error.code, 'invalid_arguments');
});

test('compare_periods: defaults to the previous period and refuses incomplete comparisons', async () => {
  const { tools } = await setup();
  const r = await tools.call('compare_periods', { current: { period: 'week' } });
  assert.equal(r.ok, true);
  assert.deepEqual([r.data.previous.from, r.data.previous.to, r.data.previous.mealCount], ['2026-09-28', '2026-10-04', 1]);
  assert.equal(r.data.nutrients.calories.comparable, true);
  assert.equal(r.data.nutrients.protein.comparable, false); // this week has a meal with unknown protein
  const custom = await tools.call('compare_periods', { current: { period: 'day', date: '2026-10-08' }, previous: { period: 'day', date: '2026-10-06' } });
  assert.deepEqual(custom.data.nutrients.calories, { current: 445, previous: 260, comparable: true, change: 185, changePct: 71 });
});

test('get_meal_history: bounded, filtered, minimal fields', async () => {
  const { tools, ids } = await setup();
  const r = await tools.call('get_meal_history', { from: '2026-10-01', to: TODAY, limit: 2 });
  assert.deepEqual([r.data.totalMeals, r.data.returned, r.data.meals.length], [5, 2, 2]);
  const scan = r.data.meals.find((m) => m.id === ids.scan);
  assert.deepEqual(Object.keys(scan).sort(), ['category', 'dayKey', 'edited', 'foods', 'id', 'missing', 'source', 'totals']);
  const dinners = await tools.call('get_meal_history', { from: '2026-10-01', to: TODAY, category: 'dinner' });
  assert.deepEqual(dinners.data.meals.map((m) => m.id).sort(), [ids.rice2, ids.scan].sort());
  assert.equal((await tools.call('get_meal_history', { from: '2026-10-01', to: TODAY, limit: 500 })).error.code, 'invalid_arguments');
});

test('render_nutrition_chart: engine numbers, null for unlogged days, bounded buckets', async () => {
  const { tools } = await setup();
  const bar = await tools.call('render_nutrition_chart', { type: 'bar', metric: 'calories', period: 'week' });
  const summary = await tools.call('get_nutrition_summary', { period: 'day', date: '2026-10-08' });
  assert.equal(bar.data.points.find((p) => p.key === '2026-10-08').value, summary.data.nutrients.calories.total);
  assert.equal(bar.data.points.find((p) => p.key === '2026-10-05').value, null);
  const pie = await tools.call('render_nutrition_chart', { type: 'pie', period: 'week' });
  assert.equal(pie.data.slices.reduce((a, x) => a + x.pct, 0), 100);
  assert.equal(pie.data.excludedMeals, 1);
  const wk = await tools.call('render_nutrition_chart', { type: 'line', metric: 'protein', groupBy: 'week', period: 'month' });
  assert.equal(wk.data.valueKind, 'daily_average');
  const year = await tools.call('render_nutrition_chart', { type: 'bar', metric: 'fat', period: 'custom', from: '2025-10-12', to: TODAY });
  assert.equal(year.error.code, 'unsupported_range');
  assert.equal((await tools.call('render_nutrition_chart', { type: 'bar', metric: 'fat', groupBy: 'month', period: 'custom', from: '2025-10-12', to: TODAY })).ok, true);
  assert.equal((await tools.call('render_nutrition_chart', { type: 'pie', metric: 'fat', period: 'week' })).error.code, 'invalid_arguments');
  assert.equal((await tools.call('render_nutrition_chart', { type: 'bar', period: 'week' })).error.code, 'invalid_arguments');
  assert.equal((await tools.call('render_nutrition_chart', { type: 'bar', metric: 'sodium', period: 'week' })).error.code, 'invalid_arguments');
});

test('get_user_preferences returns only the preference document', async () => {
  const { tools, repo } = await setup();
  await repo.savePreferences(0, { goals: { fat: 70 }, diets: ['pescatarian'] });
  const r = await tools.call('get_user_preferences', {});
  assert.deepEqual(r.data, { goals: { calories: null, protein: null, carbs: null, fat: 70 }, diets: ['pescatarian'], exclusions: [], averageBasis: 'logged_days' });
});

test('invalid requests are rejected without touching data', async () => {
  const { tools } = await setup();
  const cases = [
    ['drop_table', {}, 'unknown_tool'],
    ['get_nutrition_summary', { period: 'week', userId: 'uidB' }, 'invalid_arguments'], // no way to name another user
    ['get_nutrition_summary', { period: 'week', sql: 'DELETE FROM meals' }, 'invalid_arguments'],
    ['get_nutrition_summary', { period: 'year' }, 'invalid_arguments'],
    ['get_nutrition_summary', {}, 'invalid_arguments'],
    ['get_nutrition_summary', null, 'invalid_arguments'],
    ['get_nutrition_summary', { period: 'day', date: '2026-02-30' }, 'invalid_arguments'],
    ['get_nutrition_summary', { period: 'day', date: "2026-10-10'; --" }, 'invalid_arguments'],
    ['get_nutrition_summary', { period: 'week', from: '2026-10-01', to: '2026-10-02' }, 'invalid_arguments'],
    ['get_nutrition_summary', { period: 'custom', from: '2026-10-05' }, 'invalid_arguments'],
    ['get_nutrition_summary', { period: 'custom', from: '2026-10-09', to: '2026-10-01' }, 'invalid_arguments'],
    ['get_nutrition_summary', { period: 'custom', from: '2024-01-01', to: '2026-10-01' }, 'unsupported_range'],
    ['get_nutrition_summary', { period: 'day', date: '2026-10-12' }, 'unsupported_range'], // future
    ['get_meal_history', { from: '2026-10-01', to: TODAY, limit: 1.5 }, 'invalid_arguments'],
    ['get_user_preferences', { goals: { calories: 1 } }, 'invalid_arguments'], // read-only: no way to change goals
  ];
  for (const [name, args, code] of cases) {
    const r = await tools.call(name, args);
    assert.equal(r.ok, false, `${name} ${JSON.stringify(args)}`);
    assert.equal(r.error.code, code, `${name} ${JSON.stringify(args)}: ${r.error.message}`);
  }
});

test('tools never mutate meals, the outbox or preferences', async () => {
  const { db, tools } = await setup();
  const snap = async () => JSON.stringify(await Promise.all(['meals', 'outbox', 'user_preferences', 'meal_conflicts']
    .map((t) => db.getAllAsync(`SELECT * FROM ${t} ORDER BY 1`))));
  const before = await snap();
  for (const d of TOOL_DEFINITIONS) {
    await tools.call(d.name, d.name === 'get_meal_history' ? { from: '2026-09-01', to: TODAY }
      : d.name === 'compare_periods' ? { current: { period: 'month' } }
        : d.name === 'render_nutrition_chart' ? { type: 'bar', metric: 'calories', period: 'month' }
          : d.name === 'get_user_preferences' ? {} : { period: 'month' });
  }
  assert.equal(await snap(), before);
});

test('large results are capped and the cut is reported', async () => {
  const db = openNodeDb();
  await migrate(db);
  const repo = createMealRepository(db);
  for (let i = 0; i < 60; i++) {
    await repo.createMeal({ loggedAt: at('2026-10-10') + i * 1000, tzOffsetMin: 0, items: [food('x', 1, null, 1, 1, 1)], source: { kind: 'manual' } });
  }
  const r = await createNutritionTools({ repo, today: () => TODAY }).call('get_nutrition_summary', { period: 'day', date: '2026-10-10' });
  assert.equal(r.data.mealCount, 60);
  assert.equal(r.data.evidence.mealIds.length, 50);
  assert.deepEqual(r.truncated.map((t) => [t.path, t.total]), [['completeness.incompleteMeals', 60], ['evidence.mealIds', 60]]);
});
