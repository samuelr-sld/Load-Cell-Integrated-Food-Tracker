import test from 'node:test';
import assert from 'node:assert/strict';
import { openNodeDb } from '../data/nodeSqliteAdapter.mjs';
import { migrate } from '../data/schema.js';
import { createMealRepository, DuplicateScanError } from '../data/mealRepository.js';
import { createScanClient, scanBaseUrl, startScanPolling } from '../data/scanClient.js';
import { createNutritionTools } from './nutritionTools.js';
import { parseDraft, toDraft } from './mealForm.js';
import { scanToDraft, scanSnapshot, applyCandidate } from './scan.js';
import { localDayKey } from './meal.js';

const cand = (label, per_100g) => ({ label, taxonomy: 'foodseg103', source: 'yolo', confidence: 0.8, per_100g, usda: per_100g && { fdc_id: 1 } });
const scan = (over = {}) => ({
  scan_id: 'abc123', timestamp: '2026-10-12T10:00:00+00:00', status: 'ready', measured_total_grams: 250, tare_g: 0, net_grams: 250,
  result_id: 'r1', suggested_grams: 250, error: null,
  foods: [{ region_id: 1, candidates: [cand('rice', { energy_kcal: 130, protein_g: 2.7, carbs_g: null, fat_g: 0.3 }), cand('egg', null)] }], ...over,
});
const fresh = async () => {
  const db = openNodeDb();
  await migrate(db);
  return createMealRepository(db);
};

test('single-food scan prefills grams; unknown USDA nutrients stay unknown, not 0', () => {
  const d = scanToDraft(scan());
  assert.deepEqual([d.items[0].name, d.items[0].grams, d.items[0].calories, d.items[0].carbs], ['rice', '250', '130', '']);
  const { errors, input } = parseDraft(d);
  assert.deepEqual(errors, []);
  assert.equal(input.items[0].per100g.carbs, null);
  assert.equal(applyCandidate(d.items[0], scan().foods[0].candidates[1]).calories, '');   // unmapped food: all nutrients blank
});

test('several foods get no invented weights: the draft is invalid until the user allocates grams', () => {
  const s = scan({ suggested_grams: null, foods: [scan().foods[0], { region_id: 2, candidates: [cand('egg', null)] }] });
  const d = scanToDraft(s);
  assert.deepEqual(d.items.map((i) => i.grams), ['', '']);
  assert.ok(parseDraft(d).errors.some((e) => /quantity/.test(e)));
  d.items[0].grams = '150'; d.items[1].grams = '100';
  assert.deepEqual(parseDraft(d).errors, []);
});

test('no foods detected: one blank item with the measured grams, for manual food selection', () => {
  const d = scanToDraft(scan({ foods: [], needs_manual_food: true }));
  assert.deepEqual([d.items.length, d.items[0].name, d.items[0].grams], [1, '', '250']);
});

test('confirming a scan creates one canonical camera meal; confirming again does not duplicate; manual entry still works', async () => {
  const repo = await fresh();
  const s = scan();
  const save = () => repo.createMeal({ ...parseDraft(scanToDraft(s)).input, source: { kind: 'camera', ref: s.scan_id }, originalScan: scanSnapshot(s) });
  const meal = await save();
  await assert.rejects(save(), DuplicateScanError);
  const today = localDayKey(Date.now());
  const all = await repo.listMeals({ fromDayKey: today, toDayKey: today });
  assert.equal(all.length, 1);
  assert.deepEqual([all[0].source, all[0].items[0].grams, all[0].totals.carbs], [{ kind: 'camera', ref: 'abc123' }, 250, null]);
  assert.equal((await repo.getMeal(meal.id)).originalScan.measuredTotalGrams, 250);

  const manual = { ...toDraft(null), items: [{ id: null, name: 'toast', grams: '40', calories: '265', protein: '9', carbs: '49', fat: '3' }] };
  await repo.createMeal({ ...parseDraft(manual).input, source: { kind: 'manual' } });
  assert.equal((await repo.listMeals({ fromDayKey: today, toDayKey: today })).length, 2);

  const hist = await createNutritionTools({ repo, today: () => today }).call('get_meal_history', { from: today, to: today });
  assert.ok(hist.ok && hist.data.totalMeals === 2, JSON.stringify(hist));
});

test('outbox gets the scanned meal so the existing sync uploads it', async () => {
  const db = openNodeDb();
  await migrate(db);
  const repo = createMealRepository(db);
  await repo.createMeal({ ...parseDraft(scanToDraft(scan())).input, source: { kind: 'camera', ref: 'abc123' } });
  assert.equal((await db.getFirstAsync("SELECT count(*) AS n FROM outbox WHERE op = 'upsert'")).n, 1);
});

test('scanBaseUrl swaps the gateway port for the testbench port', () => {
  assert.equal(scanBaseUrl('http://192.168.1.20:8100/'), 'http://192.168.1.20:8000');
  assert.equal(scanBaseUrl('nonsense'), null);
});

test('client sends the pairing token; polling delivers each scan once, waits out processing, and defers while a screen is open', async () => {
  const calls = [];
  const queue = [scan({ status: 'processing' }), scan(), scan(), scan({ scan_id: 'next' })];
  const fetchImpl = async (url, init) => {
    calls.push([url, init.headers.Authorization]);
    return { ok: true, json: async () => ({ scan: queue.shift() ?? null }) };
  };
  const client = createScanClient({ baseUrl: 'http://laptop:8000', token: 'tok', fetchImpl });
  const got = [];
  let open = false;
  const stop = startScanPolling({ getClient: async () => client, onScan: (s) => got.push(s.scan_id), canShow: () => !open, intervalMs: 5 });
  await new Promise((r) => setTimeout(r, 60));
  open = true; // a form is open: 'next' must wait
  queue.push(scan({ scan_id: 'later' }));
  await new Promise((r) => setTimeout(r, 40));
  assert.deepEqual(got, ['abc123', 'next']);
  open = false;
  queue.push(scan({ scan_id: 'later' }), scan({ scan_id: 'later' }));
  await new Promise((r) => setTimeout(r, 40));
  stop();
  assert.deepEqual(got, ['abc123', 'next', 'later']);
  assert.deepEqual(calls[0], ['http://laptop:8000/api/scans/latest', 'Bearer tok']);
});

test('an unreachable laptop never throws out of the poller', async () => {
  const stop = startScanPolling({ getClient: async () => createScanClient({ baseUrl: 'http://x:1', token: 't', fetchImpl: async () => { throw new Error('offline'); } }), onScan: () => assert.fail(), intervalMs: 5 });
  await new Promise((r) => setTimeout(r, 30));
  stop();
});
