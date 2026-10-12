// UNIT tests: manual meal form parsing/validation (pure), plus one SQLite round trip through the repository.
import test from 'node:test';
import assert from 'node:assert/strict';
import { toDraft, parseDraft, emptyItem, suggestCategory } from './mealForm.js';
import { openNodeDb } from '../data/nodeSqliteAdapter.mjs';
import { migrate } from '../data/schema.js';
import { createMealRepository } from '../data/mealRepository.js';

const NOW = new Date(2026, 9, 11, 20, 0).getTime(); // local time
const item = (over = {}) => ({ ...emptyItem(), name: 'rice', grams: '150', calories: '130', protein: '2.7', carbs: '28', fat: '0.3', ...over });
const draft = (over = {}) => ({ category: 'lunch', undated: false, date: '2026-10-11', time: '12:30 PM', items: [item()], ...over });

test('valid draft -> createMeal input with local time, offset and derived totals', () => {
  const r = parseDraft(draft(), null, NOW);
  assert.deepEqual(r.errors, []);
  assert.equal(r.input.loggedAt, new Date(2026, 9, 11, 12, 30).getTime());
  assert.equal(r.input.tzOffsetMin, -new Date(r.input.loggedAt).getTimezoneOffset());
  assert.deepEqual(r.totals, { calories: 195, protein: 4.1, carbs: 42, fat: 0.5 });
  assert.equal(r.patch, null);
});

test('blank nutrients stay unknown (null), zero stays zero, comma decimals accepted', () => {
  const r = parseDraft(draft({ items: [item({ calories: '0', protein: '', carbs: ' ', fat: '1,5' })] }), null, NOW);
  assert.deepEqual(r.errors, []);
  assert.deepEqual(r.input.items[0].per100g, { calories: 0, protein: null, carbs: null, fat: 1.5 });
  assert.equal(r.totals.protein, null);
  assert.equal(r.totals.calories, 0);
});

test('helpful errors: category, name, grams, nutrient range, date, future time, no items', () => {
  const r = parseDraft(draft({
    category: null, date: '2026-02-30',
    items: [item({ name: ' ', grams: '0' }), item({ protein: '150', calories: 'abc' })],
  }), null, NOW);
  assert.deepEqual(r.errors, [
    'Choose breakfast, lunch, dinner or snack.',
    'Enter the date as YYYY-MM-DD and the time as h:mm AM/PM (for example 2:30 PM).',
    'Item 1: enter a food name.',
    'Item 1: quantity must be a number of grams between 0 and 5000.',
    'Item 2: calories per 100 g must be between 0 and 900 (leave blank if unknown).',
    'Item 2: protein per 100 g must be between 0 and 100 (leave blank if unknown).',
  ]);
  assert.equal(r.totals, null);
  assert.deepEqual(parseDraft(draft({ time: '9:00 PM' }), null, NOW).errors, ['The meal time cannot be in the future.']);
  assert.deepEqual(parseDraft(draft({ time: '13:00 PM' }), null, NOW).errors.length, 1);
  assert.deepEqual(parseDraft(draft({ items: [] }), null, NOW).errors, ['Add at least one food item.']);
});

test('undated meals have no time', () => {
  const r = parseDraft(draft({ undated: true, date: 'garbage' }), null, NOW);
  assert.deepEqual(r.errors, []);
  assert.equal(r.input.loggedAt, null);
  assert.equal(r.input.tzOffsetMin, null);
});

test('edit patch: only editable fields, time only when changed, scan item fields kept', () => {
  const original = {
    loggedAt: new Date(2026, 9, 11, 12, 30, 45).getTime(), category: 'unknown',
    items: [{ id: 'i1', name: 'adobo', grams: 200, label: 'chicken_adobo', confidence: 0.9, per100g: { calories: 180, protein: 15, carbs: 3, fat: 12 } }],
  };
  const d = toDraft(original);
  assert.equal(d.category, null); // 'unknown' scan category must be chosen explicitly
  assert.equal(d.time, '12:30 PM');
  d.category = 'dinner';
  d.items[0].grams = '250';
  const r = parseDraft(d, original, NOW);
  assert.deepEqual(r.errors, []);
  assert.deepEqual(Object.keys(r.patch).sort(), ['category', 'items']); // seconds preserved: time untouched
  assert.equal(r.patch.items[0].label, 'chicken_adobo');
  assert.equal(r.patch.items[0].id, 'i1');
  d.time = '1:00 PM';
  assert.equal(parseDraft(d, original, NOW).patch.loggedAt, new Date(2026, 9, 11, 13, 0).getTime());
  d.undated = true;
  assert.equal(parseDraft(d, original, NOW).patch.loggedAt, null);
});

test('SQLite round trip: create from the form, edit from the form, totals and nulls preserved', async () => {
  const db = openNodeDb();
  await migrate(db);
  const repo = createMealRepository(db);
  const created = await repo.createMeal({
    ...parseDraft(draft({ items: [item(), item({ name: 'egg', grams: '50', fat: '' })] }), null, NOW).input,
    source: { kind: 'manual' },
  });
  assert.equal(created.items.length, 2);
  assert.equal(created.totals.fat, null);
  const d = toDraft(created);
  d.items[1].fat = '10';
  const { patch } = parseDraft(d, created, NOW);
  const saved = await repo.updateMeal(created.id, created.revision.rev, patch);
  assert.equal(saved.totals.fat, 5.5); // 0.45 + 5, rounded once
  assert.equal(saved.loggedAt, created.loggedAt);
  assert.deepEqual(saved.items.map((i) => i.id), created.items.map((i) => i.id));
});

const at = (...a) => new Date(...a).getTime();

test('new form captures the current local time in 12-hour format and suggests a category', () => {
  const d = toDraft(null, at(2026, 9, 11, 8, 5));
  assert.deepEqual([d.date, d.time, d.category, d.undated], ['2026-10-11', '8:05 AM', 'breakfast', false]);
  assert.equal(toDraft(null, at(2026, 9, 11, 12, 0)).time, '12:00 PM'); // noon
  assert.equal(toDraft(null, at(2026, 9, 11, 0, 7)).time, '12:07 AM'); // just after midnight
  assert.deepEqual([6, 12, 16, 19, 23, 2].map(suggestCategory), ['breakfast', 'lunch', 'snack', 'dinner', 'snack', 'snack']);
});

test('reopening the form later takes a fresh timestamp each time; an open draft is never rewritten', () => {
  const first = toDraft(null, at(2026, 9, 11, 9, 0));
  const second = toDraft(null, at(2026, 9, 11, 13, 45));
  assert.equal(first.time, '9:00 AM');
  assert.equal(second.time, '1:45 PM');
  assert.equal(second.category, 'lunch');
  // user edits survive: parsing never consults the clock except for the future-time check
  const edited = { ...first, time: '7:15 AM', category: 'snack' };
  assert.equal(parseDraft(edited, null, at(2026, 9, 11, 23, 0)).input.loggedAt, at(2026, 9, 11, 7, 15));
  assert.equal(parseDraft(edited, null, at(2026, 9, 11, 23, 0)).input.category, 'snack');
});

test('midnight rollover: 11:59 PM and 12:00 AM land on different local days', () => {
  const late = toDraft(null, at(2026, 9, 11, 23, 59));
  const early = toDraft(null, at(2026, 9, 12, 0, 1));
  assert.deepEqual([late.date, late.time], ['2026-10-11', '11:59 PM']);
  assert.deepEqual([early.date, early.time], ['2026-10-12', '12:01 AM']);
  // opened at 11:59 PM, saved after midnight: the captured date is kept, not silently moved
  assert.equal(parseDraft(late, null, at(2026, 9, 12, 0, 3)).input.loggedAt, at(2026, 9, 11, 23, 59));
});

test('manual 12h and 24h entry parse the same instant; bad times are rejected', () => {
  const t = (time) => parseDraft(draft({ time }), null, NOW);
  assert.equal(t('2:30 pm').input.loggedAt, at(2026, 9, 11, 14, 30));
  assert.equal(t('14:30').input.loggedAt, at(2026, 9, 11, 14, 30));
  assert.equal(t('12:00 AM').input.loggedAt, at(2026, 9, 11, 0, 0));
  assert.equal(t('12:00 PM').input.loggedAt, at(2026, 9, 11, 12, 0));
  for (const bad of ['0:30 AM', '13:00 AM', '2:60 PM', '2 PM', '']) assert.equal(t(bad).errors.length, 1, bad);
});

test('editing keeps the original instant, offset and day; a changed time recomputes them', () => {
  const original = {
    loggedAt: Date.UTC(2026, 9, 10, 17, 0, 30), tzOffsetMin: 480, dayKey: '2026-10-11', category: 'lunch',
    items: [{ id: 'i1', name: 'rice', grams: 100, per100g: { calories: 1, protein: 1, carbs: 1, fat: 1 } }],
  };
  const d = toDraft(original, at(2030, 0, 1)); // "now" must be irrelevant when editing
  assert.notEqual(d.date, '2030-01-01');
  d.items[0].grams = '120';
  const same = parseDraft(d, original, NOW);
  assert.deepEqual(Object.keys(same.patch).sort(), ['category', 'items']); // no loggedAt/tzOffsetMin => stored values kept
  d.time = d.time.toLowerCase(); // retyped identically
  assert.deepEqual(Object.keys(parseDraft(d, original, NOW).patch).sort(), ['category', 'items']);
  d.time = '3:15 AM';
  const moved = parseDraft(d, original, NOW).patch;
  assert.equal(moved.tzOffsetMin, -new Date(moved.loggedAt).getTimezoneOffset()); // device offset at that instant
});

test('timezone: day key follows the stored offset, not the offset of whoever reads it later', async () => {
  const db = openNodeDb();
  await migrate(db);
  const repo = createMealRepository(db);
  const m = await repo.createMeal({ ...parseDraft(draft(), null, NOW).input, loggedAt: Date.UTC(2026, 9, 10, 17), tzOffsetMin: 480, source: { kind: 'manual' } });
  assert.equal(m.dayKey, '2026-10-11'); // 01:00 next day at +08:00
  const d = toDraft(m);
  const { patch } = parseDraft({ ...d, category: 'dinner' }, m, NOW);
  const saved = await repo.updateMeal(m.id, 1, patch);
  assert.equal(saved.dayKey, '2026-10-11');
  assert.equal(saved.tzOffsetMin, 480);
  assert.equal(saved.loggedAt, m.loggedAt);
});
