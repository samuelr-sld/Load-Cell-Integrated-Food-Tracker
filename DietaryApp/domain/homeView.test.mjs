import test from 'node:test';
import assert from 'node:assert/strict';
import { summarize } from './analytics.js';
import { todayOverview, syncPill, scaleIndicator, mealTag } from './homeView.js';

const meal = (id, totals, extra = {}) => ({
  id, dayKey: '2026-10-12', loggedAt: 1, deletedAt: null, category: 'lunch', totals,
  source: { kind: 'manual' }, revision: { rev: 1 }, ...extra,
});
const sumOf = (meals, goals = null) => summarize(meals, { from: '2026-10-12', to: '2026-10-12', today: '2026-10-12', goals });

test('overview totals equal the analytics engine and fraction is capped at 1', () => {
  const o = todayOverview(sumOf([meal('a', { calories: 600, protein: 30, carbs: 50, fat: 20 }), meal('b', { calories: 900, protein: 10, carbs: 40, fat: 5 })], { calories: 1200, protein: 100, carbs: null, fat: null }));
  assert.equal(o.calories.total, 1500);
  assert.equal(o.calories.fraction, 1);
  assert.equal(o.calories.over, true);
  assert.equal(o.macros[0].total, 40);
  assert.equal(o.macros[0].fraction, 0.4);
  assert.equal(o.macros[1].goal, null);
  assert.equal(o.macros[1].fraction, 0);
});

test('unknown nutrition stays null and a partial total is flagged as a lower bound', () => {
  const o = todayOverview(sumOf([meal('a', { calories: 500, protein: null, carbs: 10, fat: 5 }), meal('b', { calories: null, protein: null, carbs: 10, fat: 5 })]));
  assert.equal(o.macros[0].total, null);
  assert.equal(o.macros[0].lowerBound, false);
  assert.equal(o.calories.total, 500);
  assert.equal(o.calories.lowerBound, true);
});

test('empty day has zero meals and no totals', () => {
  const o = todayOverview(sumOf([]));
  assert.equal(o.mealCount, 0);
  assert.equal(o.calories.total, null);
});

test('syncPill never shows raw errors and prioritises conflicts', () => {
  assert.equal(syncPill({ uid: null }).label, 'Saved on this phone');
  assert.equal(syncPill({ uid: 'u', conflicts: 2, status: { state: 'error', error: 'boom' } }).label, '2 meals need your choice');
  assert.equal(syncPill({ uid: 'u', conflicts: 1 }).label, '1 meal needs your choice');
  const err = syncPill({ uid: 'u', conflicts: 0, status: { state: 'error', error: 'permission-denied' } });
  assert.equal(err.label, 'Sync paused · will retry');
  assert.equal(syncPill({ uid: 'u', unsynced: 3, status: { state: 'idle' } }).label, '3 waiting to sync');
  assert.equal(syncPill({ uid: 'u', unsynced: 0, status: { state: 'idle' } }).label, 'Up to date');
  assert.equal(syncPill({ uid: 'u', unsynced: 0, status: null }), null);
});

test('scaleIndicator maps every link state and defaults to checking', () => {
  assert.equal(scaleIndicator('online').tone, 'ok');
  assert.equal(scaleIndicator('offline').tone, 'warn');
  assert.equal(scaleIndicator('unpaired').title, 'Scale not paired');
  assert.equal(scaleIndicator(undefined).title, 'Looking for your scale…');
});

test('mealTag separates estimates from corrected, manual and sample meals', () => {
  assert.equal(mealTag(meal('a', {}, { source: { kind: 'camera' } })).label, 'Estimated');
  assert.equal(mealTag(meal('a', {}, { source: { kind: 'camera' }, revision: { rev: 2 } })).label, 'Corrected');
  assert.equal(mealTag(meal('a', {}, { source: { kind: 'sample' } })).label, 'Sample');
  assert.equal(mealTag(meal('a', {})), null);
});
