import test from 'node:test';
import assert from 'node:assert/strict';
import {
  nutrientsForServing, mealTotals, nutritionCompleteness, validateMeal, softDelete,
  localDayKey, addDays, roundTo,
} from './meal.js';

const P = { calories: 200, protein: 10, carbs: 30, fat: 5 };
const item = (grams, per100g = P, name = 'rice') => ({ id: `i${grams}${name}`, name, grams, per100g });
const meal = (over = {}) => ({
  id: 'm1', loggedAt: Date.UTC(2026, 9, 10, 12), dayKey: '2026-10-10', category: 'lunch',
  items: [item(150)], source: { kind: 'manual' }, revision: { rev: 1, updatedAt: 1 }, deletedAt: null, ...over,
});

test('serving nutrients scale per 100g and round to 1 dp', () => {
  assert.deepEqual(nutrientsForServing(P, 150), { calories: 300, protein: 15, carbs: 45, fat: 7.5 });
  assert.equal(nutrientsForServing({ calories: 33.33 }, 100).calories, 33.3);
  assert.equal(roundTo(1.005, 2), 1.01);
});

test('null stays null; zero stays zero; undefined is unknown', () => {
  const r = nutrientsForServing({ calories: 0, protein: null }, 50);
  assert.deepEqual(r, { calories: 0, protein: null, carbs: null, fat: null });
  assert.equal(nutrientsForServing(P, 0).calories, 0);
});

test('invalid input throws', () => {
  assert.throws(() => nutrientsForServing(P, -1), RangeError);
  assert.throws(() => nutrientsForServing(P, NaN), RangeError);
  assert.throws(() => nutrientsForServing({ calories: -5 }, 10), RangeError);
  assert.throws(() => nutrientsForServing({ calories: '5' }, 10), RangeError);
});

test('meal totals sum items; unknown in any item propagates null', () => {
  assert.equal(mealTotals([item(100), item(50)]).calories, 300);
  const t = mealTotals([item(100), item(100, { calories: 100, protein: null, carbs: 1, fat: 1 }, 'x')]);
  assert.equal(t.protein, null);
  assert.equal(t.calories, 300);
  assert.deepEqual(nutritionCompleteness(t), { complete: false, missing: ['protein'] });
  assert.equal(mealTotals([]).calories, null);
});

test('validateMeal accepts canonical meal and reports problems', () => {
  assert.deepEqual(validateMeal(meal()), []);
  const errs = validateMeal(meal({ id: '', category: 'brunch', dayKey: '10/10', items: [item(-1)], deletedAt: 'x' }));
  assert.equal(errs.length, 5);
  assert.ok(validateMeal(null).length);
});

test('softDelete bumps revision and keeps the record', () => {
  const d = softDelete(meal(), 99);
  assert.equal(d.deletedAt, 99);
  assert.deepEqual(d.revision, { rev: 2, updatedAt: 99 });
  assert.equal(d.items.length, 1);
});

test('day key respects the timezone offset at the boundary', () => {
  const t = Date.UTC(2026, 9, 10, 16, 30); // 16:30Z
  assert.equal(localDayKey(t, 0), '2026-10-10');
  assert.equal(localDayKey(t, 8 * 60), '2026-10-11'); // 00:30 next day at +08:00
  assert.equal(localDayKey(t, -5 * 60), '2026-10-10');
  assert.equal(addDays('2026-03-01', -1), '2026-02-28');
  assert.equal(addDays('2026-12-31', 1), '2027-01-01');
});

test('AI prompts say "unknown" for missing nutrition, never "null" or a fabricated 0', async () => {
  const { generateWeeklyInsightsAdvice, generateSingleMealAdvice, generateDietaryAdvice } = await import('../geminiService.js');
  const prompts = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (_url, opts) => {
    prompts.push(JSON.parse(opts.body).contents[0].parts[0].text);
    return { json: async () => ({}) };
  };
  try {
    await generateWeeklyInsightsAdvice({ avgCalories: 500, avgProtein: null, avgCarbs: null, avgFat: 0, topFoods: [] });
    await generateSingleMealAdvice({ mealType: 'x', totalCalories: null, totalProtein: 3 });
    await generateDietaryAdvice([{ mealType: 'x', timestamp: null, totalCalories: null, totalFat: 0 }]);
  } finally {
    globalThis.fetch = realFetch;
  }
  for (const p of prompts) assert.doesNotMatch(p, /null/);
  assert.match(prompts[0], /Daily Average Protein: unknown\n/);
  assert.match(prompts[0], /Daily Average Fat: 0 g/);
  assert.match(prompts[1], /Total Calories: unknown\n/);
  assert.match(prompts[2], /\[undated\].*unknown \| .*Fat: 0g/);
});
