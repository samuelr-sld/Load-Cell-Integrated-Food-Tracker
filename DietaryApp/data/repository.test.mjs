import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { openNodeDb } from './nodeSqliteAdapter.mjs';
import { migrate, MIGRATIONS } from './schema.js';
import {
  createMealRepository, RevisionConflictError, DuplicateScanError, MealNotFoundError,
} from './mealRepository.js';

const T = Date.UTC(2026, 9, 10, 12);
const rice = { name: 'rice', grams: 150, per100g: { calories: 130, protein: 2.7, carbs: 28, fat: 0.3 } };
const input = (over = {}) => ({
  loggedAt: T, tzOffsetMin: 480, category: 'lunch', items: [rice], source: { kind: 'manual' }, ...over,
});

const fresh = async (path) => {
  const db = openNodeDb(path);
  await migrate(db);
  return { db, repo: createMealRepository(db) };
};
const count = async (db, table) => (await db.getFirstAsync(`SELECT count(*) AS n FROM ${table}`)).n;

test('init creates all tables at the latest version; rerunning migrations is a no-op', async () => {
  const { db } = await fresh();
  const names = (await db.getAllAsync("SELECT name FROM sqlite_master WHERE type = 'table'")).map((r) => r.name);
  for (const t of ['meals', 'outbox', 'meal_conflicts', 'sync_state']) assert.ok(names.includes(t), t);
  assert.equal((await db.getFirstAsync('PRAGMA user_version')).user_version, MIGRATIONS.length);
  await migrate(db);
  await migrate(db);
  assert.equal((await db.getFirstAsync('PRAGMA user_version')).user_version, MIGRATIONS.length);
});

test('v1 -> v2 upgrade keeps real meals and their outbox, and retags only demo samples', async () => {
  const db = openNodeDb();
  await db.withExclusiveTransactionAsync(async (t) => {
    await t.execAsync(MIGRATIONS[0]);
    await t.execAsync('PRAGMA user_version = 1');
  });
  const v1 = createMealRepository(db); // v1 code path wrote samples as source kind 'manual'
  const sample = await v1.createMeal(input({ items: [{ ...rice, name: 'SAMPLE oatmeal' }] }));
  const real = await v1.createMeal(input());
  const mixed = await v1.createMeal(input({ items: [{ ...rice, name: 'SAMPLE x' }, rice] }));
  assert.equal(await count(db, 'outbox'), 3);
  await migrate(db);
  const repo = createMealRepository(db);
  assert.equal((await repo.getMeal(sample.id)).source.kind, 'sample');
  assert.equal((await repo.getMeal(real.id)).source.kind, 'manual');
  assert.equal((await repo.getMeal(mixed.id)).source.kind, 'manual'); // user-edited, not a pure sample
  assert.deepEqual((await repo.listOutbox()).map((o) => o.meal_id).sort(), [real.id, mixed.id].sort());
  await assert.rejects(db.runAsync("UPDATE meals SET source_kind = 'manual' WHERE id = ?", [sample.id]), /immutable/);
  const d = await repo.deleteMeal(sample.id, 1); // samples stay deletable, without queueing a cloud delete
  assert.ok(d.deletedAt);
  assert.equal(await count(db, 'outbox'), 2);
});

test('parallel creates all commit through the transaction queue', async () => {
  const { repo } = await fresh();
  const made = await Promise.all(Array.from({ length: 10 }, () => repo.createMeal(input())));
  assert.equal(new Set(made.map((m) => m.id)).size, 10);
  assert.equal((await repo.listMeals()).length, 10);
});

test('create writes meal and outbox atomically with derived totals', async () => {
  const { db, repo } = await fresh();
  const m = await repo.createMeal({ ...input(), totals: { calories: 99999 } });
  assert.match(m.id, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  assert.equal(m.totals.calories, 195); // derived, input totals ignored
  assert.equal(m.dayKey, '2026-10-10');
  const ob = await db.getAllAsync('SELECT * FROM outbox');
  assert.equal(ob.length, 1);
  assert.equal(ob[0].meal_id, m.id);
  assert.equal(ob[0].op, 'upsert');
  assert.equal(ob[0].meal_rev, 1);
});

test('update bumps rev and coalesces the outbox row with a new edit id', async () => {
  const { db, repo } = await fresh();
  const m = await repo.createMeal(input());
  const before = await db.getFirstAsync('SELECT * FROM outbox');
  const u = await repo.updateMeal(m.id, 1, { items: [{ ...rice, grams: 300 }] });
  assert.equal(u.revision.rev, 2);
  assert.equal(u.totals.calories, 390);
  const ob = await db.getAllAsync('SELECT * FROM outbox');
  assert.equal(ob.length, 1);
  assert.equal(ob[0].meal_rev, 2);
  assert.notEqual(ob[0].edit_id, before.edit_id);
  assert.equal(ob[0].created_at, before.created_at);
});

test('a failure mid-transaction rolls back the meal write', async () => {
  const { db, repo } = await fresh();
  await db.execAsync("CREATE TEMP TRIGGER fail_outbox BEFORE INSERT ON outbox BEGIN SELECT RAISE(ABORT, 'boom'); END;");
  await assert.rejects(repo.createMeal(input()), /boom/);
  assert.equal(await count(db, 'meals'), 0);
  await db.execAsync('DROP TRIGGER fail_outbox');

  const m = await repo.createMeal(input());
  await db.execAsync("CREATE TEMP TRIGGER fail_outbox2 BEFORE UPDATE ON outbox BEGIN SELECT RAISE(ABORT, 'boom'); END;");
  await assert.rejects(repo.updateMeal(m.id, 1, { category: 'dinner' }), /boom/);
  const after = await repo.getMeal(m.id);
  assert.equal(after.revision.rev, 1);
  assert.equal(after.category, 'lunch');
});

test('stale revision is rejected', async () => {
  const { repo } = await fresh();
  const m = await repo.createMeal(input());
  await repo.updateMeal(m.id, 1, { category: 'dinner' });
  await assert.rejects(repo.updateMeal(m.id, 1, { category: 'snack' }), RevisionConflictError);
  await assert.rejects(repo.deleteMeal(m.id, 1), RevisionConflictError);
});

test('soft delete keeps the row, hides it, queues a delete, and blocks hard delete', async () => {
  const { db, repo } = await fresh();
  const m = await repo.createMeal(input());
  const d = await repo.deleteMeal(m.id, 1);
  assert.ok(d.deletedAt);
  assert.equal((await repo.listMeals()).length, 0);
  assert.ok((await repo.getMeal(m.id)).deletedAt);
  assert.equal((await db.getFirstAsync('SELECT op FROM outbox')).op, 'delete');
  await assert.rejects(repo.updateMeal(m.id, 2, { category: 'dinner' }), RevisionConflictError);
  await assert.rejects(db.runAsync('DELETE FROM meals WHERE id = ?', [m.id]), /unsynchronized/);
});

test('null nutrients survive the round trip; zero stays zero', async () => {
  const { repo } = await fresh();
  const item = { name: 'mystery', grams: 100, per100g: { calories: 0, protein: null, carbs: null, fat: null } };
  const m = await repo.createMeal(input({ items: [item] }));
  const back = await repo.getMeal(m.id);
  assert.deepEqual(back.totals, { calories: 0, protein: null, carbs: null, fat: null });
  assert.equal(back.items[0].per100g.protein, null);
});

test('duplicate scan import is rejected, including after soft delete', async () => {
  const { db, repo } = await fresh();
  const scan = input({ source: { kind: 'camera', ref: 'scan-1' }, originalScan: { calories: 500 } });
  const m = await repo.createMeal(scan);
  await assert.rejects(repo.createMeal(scan), DuplicateScanError);
  await repo.deleteMeal(m.id, 1);
  await assert.rejects(repo.createMeal(scan), DuplicateScanError);
  await assert.rejects(repo.createMeal(input({ source: { kind: 'camera' } })), /source\.ref/);
  assert.equal(await count(db, 'meals'), 1);
});

test('original scan values cannot be changed', async () => {
  const { db, repo } = await fresh();
  const m = await repo.createMeal(input({ source: { kind: 'camera', ref: 's' }, originalScan: { calories: 1 } }));
  await assert.rejects(repo.updateMeal(m.id, 1, { originalScan: {} }), /Not editable/);
  await assert.rejects(db.runAsync("UPDATE meals SET original_scan_json = '{}' WHERE id = ?", [m.id]), /immutable/);
});

test('dayKey changes only when the time or timezone offset changes', async () => {
  const { db, repo } = await fresh();
  const m = await repo.createMeal(input({ loggedAt: Date.UTC(2026, 9, 10, 17), tzOffsetMin: 480 }));
  assert.equal(m.dayKey, '2026-10-11');
  const a = await repo.updateMeal(m.id, 1, { category: 'dinner' });
  assert.equal(a.dayKey, '2026-10-11');
  const b = await repo.updateMeal(m.id, 2, { tzOffsetMin: 0 });
  assert.equal(b.dayKey, '2026-10-10');
  const c = await repo.updateMeal(m.id, 3, { loggedAt: null });
  assert.equal(c.dayKey, null);
  assert.equal((await repo.listUndated()).length, 1);
  assert.equal((await repo.listMeals()).length, 0);
  assert.equal((await db.getFirstAsync('SELECT day_key FROM meals')).day_key, null);
});

test('offline first launch: meals are pending, then adopted atomically by the first UID', async () => {
  const { db, repo } = await fresh();
  const m = await repo.createMeal(input());
  assert.equal(m.ownerStatus, 'pending');
  assert.equal(m.ownerUid, null);
  assert.deepEqual(await repo.setIdentity('uidA'), { adopted: 1, pendingLeft: 0 });
  const back = await repo.getMeal(m.id);
  assert.equal(back.ownerUid, 'uidA');
  assert.equal(back.ownerStatus, 'owned');
  assert.deepEqual(await repo.setIdentity('uidA'), { adopted: 0, pendingLeft: 0 }); // idempotent rerun
  assert.equal((await db.getFirstAsync("SELECT value FROM sync_state WHERE key = 'bound_uid'")).value, 'uidA');
});

test('interrupted adoption leaves data pending and can be retried', async () => {
  const { db, repo } = await fresh();
  await repo.createMeal(input());
  await db.execAsync("CREATE TEMP TRIGGER fail_adopt BEFORE UPDATE OF owner_uid ON meals BEGIN SELECT RAISE(ABORT, 'crash'); END;");
  await assert.rejects(repo.setIdentity('uidA'), /crash/);
  assert.equal(await count(db, 'sync_state'), 0); // bound_uid insert rolled back too
  assert.equal((await db.getFirstAsync("SELECT count(*) AS n FROM meals WHERE owner_status = 'pending'")).n, 1);
  await db.execAsync('DROP TRIGGER fail_adopt');
  assert.equal((await repo.setIdentity('uidA')).adopted, 1);
});

test('a different UID never adopts pending data or takes over owned data', async () => {
  const { db, repo } = await fresh();
  const a = await repo.createMeal(input());
  await repo.setIdentity('uidA');
  await repo.setIdentity(null); // signed out: nothing deleted
  const p = await repo.createMeal(input());
  assert.equal(p.ownerStatus, 'pending');
  assert.deepEqual(await repo.setIdentity('uidB'), { adopted: 0, pendingLeft: 1 });
  await assert.rejects(
    db.runAsync("UPDATE meals SET owner_uid = 'uidB' WHERE id = ?", [a.id]), /owner cannot change/
  );
  assert.equal(await count(db, 'meals'), 2);
});

test('identities are isolated for reads and writes', async () => {
  const { repo } = await fresh();
  await repo.setIdentity('uidA');
  const a = await repo.createMeal(input());
  await repo.setIdentity('uidB');
  const b = await repo.createMeal(input());
  assert.deepEqual((await repo.listMeals()).map((m) => m.id), [b.id]);
  assert.equal(await repo.getMeal(a.id), null);
  await assert.rejects(repo.updateMeal(a.id, 1, { category: 'dinner' }), MealNotFoundError);
  await assert.rejects(repo.deleteMeal(a.id, 1), MealNotFoundError);
  assert.equal((await repo.listOutbox()).length, 1);
  await repo.setIdentity(null);
  assert.equal((await repo.listMeals()).length, 0); // pending view does not leak owned meals
});

test('data and identity binding survive a restart (file-backed DB)', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'platego-'));
  const path = join(dir, 'p.db');
  try {
    const first = await fresh(path);
    const m = await first.repo.createMeal(input());
    await first.repo.setIdentity('uidA');
    await first.db.closeAsync();

    const second = await fresh(path); // reruns migrations on an existing file
    await second.repo.setIdentity('uidA');
    const back = await second.repo.getMeal(m.id);
    assert.equal(back.ownerUid, 'uidA');
    assert.deepEqual(back.items[0].per100g, rice.per100g);
    assert.equal((await second.repo.listOutbox()).length, 1);
    await second.db.closeAsync();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('onChange fires after commits and unsubscribes', async () => {
  const { repo } = await fresh();
  let n = 0;
  const off = repo.onChange(() => n++);
  await repo.createMeal(input());
  await assert.rejects(repo.createMeal(input({ category: 'brunch' })), TypeError);
  assert.equal(n, 1);
  off();
  await repo.createMeal(input());
  assert.equal(n, 1);
});

test('v2 -> v3 upgrade adds preferences and keeps meals, outbox and conflicts untouched', async () => {
  const db = openNodeDb();
  for (const [i, sql] of MIGRATIONS.slice(0, 2).entries()) {
    await db.withExclusiveTransactionAsync(async (t) => {
      await t.execAsync(sql);
      await t.execAsync(`PRAGMA user_version = ${i + 1}`);
    });
  }
  const before = createMealRepository(db);
  const m = await before.createMeal(input({ items: [{ ...rice, per100g: { ...rice.per100g, fat: null } }] }));
  const snapshot = await db.getAllAsync('SELECT * FROM meals');
  await migrate(db);
  const repo = createMealRepository(db);
  assert.deepEqual(await db.getAllAsync('SELECT * FROM meals'), snapshot);
  assert.equal((await repo.listOutbox())[0].meal_id, m.id);
  assert.equal((await repo.getMeal(m.id)).totals.fat, null);
  assert.equal((await repo.getPreferences()).rev, 0);
});

test('preferences: defaults, save, optimistic rev, validation, per-identity isolation, adoption', async () => {
  const { repo } = await fresh();
  const d = await repo.getPreferences();
  assert.deepEqual([d.rev, d.goals.calories, d.diets, d.averageBasis], [0, null, [], 'logged_days']);
  const s = await repo.savePreferences(0, { goals: { calories: 1800 }, exclusions: ['Peanut'] }); // before sign-in
  assert.equal(s.rev, 1);
  await assert.rejects(repo.savePreferences(0, { goals: { calories: 1900 } }), RevisionConflictError);
  await assert.rejects(repo.savePreferences(1, { goals: { calories: -5 } }), /Invalid preferences/);
  assert.equal((await repo.getPreferences()).goals.calories, 1800);

  await repo.setIdentity('uidA'); // first sign-in adopts the device's preferences with its meals
  assert.deepEqual((await repo.getPreferences()).exclusions, ['peanut']);
  await repo.setIdentity('uidB');
  assert.equal((await repo.getPreferences()).goals.calories, null); // another account sees only its own
  await repo.savePreferences(0, { goals: { calories: 2500 } });
  await repo.setIdentity('uidA');
  assert.equal((await repo.getPreferences()).goals.calories, 1800);
});

test('preferences survive a restart (file-backed DB)', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'platego-'));
  const path = join(dir, 'p.db');
  try {
    const first = await fresh(path);
    await first.repo.setIdentity('uidA');
    await first.repo.savePreferences(0, { goals: { protein: 90 }, diets: ['halal'], averageBasis: 'calendar_days' });
    await first.db.closeAsync();
    const second = await fresh(path);
    await second.repo.setIdentity('uidA');
    const p = await second.repo.getPreferences();
    assert.deepEqual([p.rev, p.goals.protein, p.diets, p.averageBasis], [1, 90, ['halal'], 'calendar_days']);
    await second.db.closeAsync();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
