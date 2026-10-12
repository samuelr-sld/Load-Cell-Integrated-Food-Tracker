// Dashboard auto-scan logic (static/autoscan.js) run in a vm with stubbed DOM helpers. `node --test testbench/tests/autoscan.test.mjs`
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const src = readFileSync(new URL('../static/autoscan.js', import.meta.url), 'utf8');

const setup = (results = {}) => {
  const box = { hidden: true, className: '', kids: [], replaceChildren(...k) { this.kids = k; } };
  const cards = [], fetched = [];
  const text = (n) => (n && n.t !== undefined ? n.t + (n.kids ?? []).map(text).join('') : String(n ?? ''));
  const ctx = {
    window: { __AUTOSCAN_TEST__: true }, document: {}, console,
    $: () => box, fmtPct: (v) => `${(v * 100).toFixed(1)}%`,
    h: (tag, attrs, ...kids) => ({ t: '', tag, kids: kids.flat().filter((k) => k != null && k !== false) }),
    api: async (path) => { fetched.push(path); if (path in results) return results[path]; throw new Error('404'); },
    finishUpload: async (res, cam) => cards.push([res.result_id, cam]),
  };
  vm.createContext(ctx);
  vm.runInContext(src + '\nthis.apply = autoscanApply; this.S = autoscan;', ctx);
  const dump = (n) => (typeof n === 'string' ? n : n?.kids ? n.kids.map(dump).join(' ') : String(n ?? ''));
  return { ctx, box, cards, fetched, apply: ctx.apply, S: ctx.S, shown: () => box.kids.map(dump).join(' | ') };
};
const scan = (o = {}) => ({ scan_id: 's1', timestamp: '2026-10-12T10:00:05+00:00', status: 'ready', measured_total_grams: 250, tare_g: 0, net_grams: 250,
  result_id: 'r1', camera_id: 'index:0', needs_manual_food: false, suggested_grams: 250, error: null,
  foods: [{ region_id: 1, candidates: [{ label: 'rice', confidence: 0.81, source: 'yolo', per_100g: { energy_kcal: 130 } }] }], ...o });

test('a new processing scan shows "Analyzing meal" with the measured weight, and no card yet', async () => {
  const t = setup();
  await t.apply(scan({ status: 'processing', foods: [] }), 1);   // first poll, still processing: shown
  assert.equal(t.box.hidden, false);
  assert.match(t.shown(), /Analyzing meal/);
  assert.match(t.shown(), /250 g on the scale/);
  assert.equal(t.cards.length, 0);
});

test('processing -> ready renders the stored result once via the shared card path, with confidence and kcal', async () => {
  const t = setup({ '/api/results/r1': { result_id: 'r1' } });
  await t.apply(scan({ status: 'processing', foods: [] }), 1);
  await t.apply(scan(), 2);
  await t.apply(scan(), 3);            // duplicate poll
  await t.apply(scan(), 4);
  assert.deepEqual(t.cards, [['r1', 'index:0']]);
  assert.equal(t.fetched.filter((p) => p === '/api/results/r1').length, 1);
  assert.match(t.shown(), /rice 81\.0% \(yolo · 130 kcal\/100 g\)/);
});

test('failed scan shows the error and the manual fallback, renders no card, and is not retried', async () => {
  const t = setup();
  await t.apply(scan({ status: 'processing', foods: [] }), 1);
  await t.apply(scan({ status: 'error', result_id: null, foods: [], error: { message: 'camera unplugged' } }), 2);
  await t.apply(scan({ status: 'error', result_id: null, foods: [], error: { message: 'camera unplugged' } }), 3);
  assert.match(t.shown(), /Automatic scan failed/);
  assert.match(t.shown(), /camera unplugged/);
  assert.match(t.shown(), /manually/);
  assert.equal(t.cards.length, 0);
  assert.deepEqual(t.fetched, []);                       // discovery alone never calls anything but the scan API
});

test('an old response cannot overwrite a newer scan, and an older scan id is dropped', async () => {
  const t = setup({ '/api/results/r2': { result_id: 'r2' } });
  await t.apply(scan({ status: 'processing', scan_id: 's2', timestamp: '2026-10-12T10:05:00+00:00', result_id: null, foods: [] }), 1);
  await t.apply(scan({ scan_id: 's1' }), 2);             // late response for the earlier scan
  assert.equal(t.S.current.scan_id, 's2');
  await t.apply(scan({ scan_id: 's2', timestamp: '2026-10-12T10:05:00+00:00', result_id: 'r2' }), 4);
  await t.apply(scan({ status: 'processing', scan_id: 's2', timestamp: '2026-10-12T10:05:00+00:00', foods: [] }), 3);   // stale seq
  assert.equal(t.S.current.status, 'ready');
  assert.deepEqual(t.cards, [['r2', 'index:0']]);
});

test('page reload: a scan that finished earlier is not displayed again (its card comes from the stored results)', async () => {
  const t = setup({ '/api/results/r1': { result_id: 'r1' } });
  await t.apply(scan(), 1);
  assert.equal(t.cards.length, 0);
  assert.equal(t.box.hidden, true);
});

test('the banner stays after the plate is removed and the phone acknowledges (no scan => no change)', async () => {
  const t = setup({ '/api/results/r1': { result_id: 'r1' } });
  await t.apply(scan({ status: 'processing', foods: [] }), 1);
  await t.apply(scan(), 2);
  await t.apply(null, 3);
  assert.match(t.shown(), /Automatic scan/);
  assert.equal(t.cards.length, 1);
});

test('a result that vanished from server memory (404) is reported once and not re-fetched every poll', async () => {
  const t = setup();
  t.ctx.api = async (p) => { t.fetched.push(p); throw Object.assign(new Error('404'), { status: 404 }); };
  t.box.append = function (n) { this.kids.push(n); };
  t.S.first = false;                                          // not a page reload: the scan finished while the page was open
  await t.apply(scan(), 1);
  await t.apply(scan(), 2);
  await t.apply(scan(), 3);
  assert.equal(t.fetched.filter((p) => p === '/api/results/r1').length, 1);
  assert.match(t.shown(), /no longer in server memory/);
});
