// Firebase EMULATOR tests: real Firestore emulator (rules + transactions + queries) and real SQLite
// (node:sqlite). Run with `npm run test:emulator`; refuses to run without the emulator so nothing
// can reach the production project.
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { initializeApp, deleteApp } from 'firebase/app';
import {
  initializeFirestore, connectFirestoreEmulator, doc, getDoc, getDocs, setDoc, updateDoc, deleteDoc,
  collection, writeBatch, serverTimestamp, setLogLevel,
} from 'firebase/firestore';
import { openNodeDb } from './nodeSqliteAdapter.mjs';
import { migrate } from './schema.js';
import { createMealRepository, IdentityChangedError } from './mealRepository.js';
import { createSyncEngine } from './sync.js';

const HOST = process.env.FIRESTORE_EMULATOR_HOST;
if (!HOST) throw new Error('FIRESTORE_EMULATOR_HOST not set: run via `npm run test:emulator`');
const PROJECT = 'demo-platego'; // demo-* projects never resolve to a real backend
setLogLevel('silent'); // denied writes are expected in the rules tests

let appN = 0;
const apps = [];
after(() => Promise.all(apps.map((a) => deleteApp(a)))); // open clients would keep the test process alive
// One Firestore client acting as `uid` (null = unauthenticated), like one device.
// host override = an unreachable server, i.e. what an offline device sees.
const client = (uid, host = HOST) => {
  const app = initializeApp({ projectId: PROJECT, apiKey: 'demo' }, `c${appN++}`);
  apps.push(app);
  const fs = initializeFirestore(app, {});
  const [h, p] = host.split(':');
  connectFirestoreEmulator(fs, h, Number(p), uid ? { mockUserToken: { user_id: uid } } : {});
  return fs;
};
const rules = (fs) => assert.rejects(fs, /permission|PERMISSION_DENIED/i);

// A device = its own SQLite file + repository + Firestore client, signed in as uid.
const device = async (uid, path) => {
  const db = openNodeDb(path);
  await migrate(db);
  const repo = createMealRepository(db);
  await repo.setIdentity(uid);
  const fs = client(uid);
  return { db, repo, fs, sync: (opts) => createSyncEngine({ repo, firestore: fs, uid, ...opts }) };
};
const newUid = () => `u-${randomUUID().slice(0, 8)}`;
const rice = (grams = 150) => ({ name: 'rice', grams, per100g: { calories: 130, protein: 2.7, carbs: 28, fat: 0.3 } });
const input = (over = {}) => ({
  loggedAt: Date.UTC(2026, 9, 10, 12), tzOffsetMin: 480, category: 'lunch', items: [rice()], source: { kind: 'manual' }, ...over,
});
const remoteOf = async (fs, uid, id) => (await getDoc(doc(fs, 'users', uid, 'meals', id))).data();
const outbox = (d) => d.db.getAllAsync('SELECT * FROM outbox');

const validDoc = (uid, id, over = {}) => ({
  id, ownerUid: uid, loggedAt: 1, tzOffsetMin: 0, dayKey: '1970-01-01', category: 'lunch', items: [],
  source: { kind: 'manual', ref: null }, originalScanJson: null, createdAt: 1, deletedAt: null,
  rev: 1, lastEditId: 'e1', serverUpdatedAt: serverTimestamp(), ...over,
});

// ---------------- Security rules ----------------

test('rules: owner can create, read and update with rev+1; delete is denied', async () => {
  const uid = newUid();
  const fs = client(uid);
  const ref = doc(fs, 'users', uid, 'meals', 'm1');
  await setDoc(ref, validDoc(uid, 'm1'));
  assert.equal((await getDoc(ref)).data().rev, 1);
  await setDoc(ref, validDoc(uid, 'm1', { rev: 2, lastEditId: 'e2', category: 'dinner' }));
  await rules(setDoc(ref, validDoc(uid, 'm1', { rev: 2, lastEditId: 'e3' }))); // rev must increase by exactly 1
  await rules(setDoc(ref, validDoc(uid, 'm1', { rev: 5, lastEditId: 'e3' })));
  await rules(deleteDoc(ref));
});

test('rules: other users and unauthenticated clients are denied', async () => {
  const alice = newUid();
  const bob = newUid();
  await setDoc(doc(client(alice), 'users', alice, 'meals', 'm1'), validDoc(alice, 'm1'));
  const asBob = client(bob);
  await rules(getDoc(doc(asBob, 'users', alice, 'meals', 'm1')));
  await rules(getDocs(collection(asBob, 'users', alice, 'meals')));
  await rules(setDoc(doc(asBob, 'users', alice, 'meals', 'm2'), validDoc(alice, 'm2')));
  await rules(setDoc(doc(asBob, 'users', alice, 'meals', 'm1'), validDoc(alice, 'm1', { rev: 2 })));
  await rules(setDoc(doc(asBob, 'users', bob, 'meals', 'm3'), validDoc(alice, 'm3'))); // ownerUid must match path
  const anon = client(null);
  await rules(getDoc(doc(anon, 'users', alice, 'meals', 'm1')));
  await rules(setDoc(doc(anon, 'users', alice, 'meals', 'm4'), validDoc(alice, 'm4')));
});

test('rules: shape, server timestamp and immutable origin are enforced', async () => {
  const uid = newUid();
  const fs = client(uid);
  const ref = (id) => doc(fs, 'users', uid, 'meals', id);
  await rules(setDoc(ref('a'), validDoc(uid, 'a', { rev: 2 }))); // create must be rev 1
  await rules(setDoc(ref('b'), validDoc(uid, 'b', { serverUpdatedAt: 5 })));
  await rules(setDoc(ref('c'), validDoc(uid, 'c', { extra: true })));
  await rules(setDoc(ref('d'), validDoc(uid, 'd', { category: 'brunch' })));
  await rules(setDoc(ref('e'), validDoc(uid, 'e', { source: { kind: 'sample', ref: null } })));
  await rules(setDoc(ref('f'), validDoc(uid, 'wrong-id')));
  await setDoc(ref('g'), validDoc(uid, 'g', { source: { kind: 'camera', ref: 's1' }, originalScanJson: '{"kcal":1}' }));
  await rules(setDoc(ref('g'), validDoc(uid, 'g', { rev: 2, source: { kind: 'camera', ref: 's2' }, originalScanJson: '{"kcal":1}' })));
  await rules(setDoc(ref('g'), validDoc(uid, 'g', { rev: 2, source: { kind: 'camera', ref: 's1' }, originalScanJson: '{"kcal":9}' })));
  await setDoc(ref('g'), validDoc(uid, 'g', { rev: 2, source: { kind: 'camera', ref: 's1' }, originalScanJson: '{"kcal":1}', category: 'snack' }));
  await rules(updateDoc(ref('g'), { createdAt: 2 }));
});

test('rules: legacy meal collection is read-only for signed-in users', async () => {
  const fs = client(newUid());
  await getDocs(collection(fs, 'meal'));
  await rules(setDoc(doc(fs, 'meal', 'x'), { totalCalories: 1 }));
  await rules(getDocs(collection(client(null), 'meal')));
});

// ---------------- Sync engine ----------------

test('push then pull: a meal created on device A appears on device B and edits flow back', async () => {
  const uid = newUid();
  const a = await device(uid);
  const b = await device(uid);
  const m = await a.repo.createMeal(input());
  assert.deepEqual(await a.sync().syncOnce(), { pushed: 1, pulled: 1, applied: 0, conflicts: 0 });
  assert.equal((await outbox(a)).length, 0);
  const remote = await remoteOf(a.fs, uid, m.id);
  assert.equal(remote.rev, 1);
  assert.equal(remote.ownerUid, uid);
  assert.equal((await a.repo.getMeal(m.id)).remoteRev, 1);

  await b.sync().syncOnce();
  const onB = await b.repo.getMeal(m.id);
  assert.equal(onB.ownerUid, uid);
  assert.deepEqual(onB.totals, m.totals);
  assert.deepEqual(onB.items, m.items);

  await b.repo.updateMeal(m.id, onB.revision.rev, { category: 'dinner' });
  await b.sync().syncOnce();
  assert.equal((await remoteOf(b.fs, uid, m.id)).rev, 2);
  const r = await a.sync().syncOnce();
  assert.equal(r.applied, 1);
  const onA = await a.repo.getMeal(m.id);
  assert.equal(onA.category, 'dinner');
  assert.equal(onA.revision.rev, 2); // local rev bumped, so a stale open edit form is rejected
  assert.equal(onA.remoteRev, 2);
});

test('camera meal: scan metadata with nested arrays syncs intact and stays immutable', async () => {
  const uid = newUid();
  const a = await device(uid);
  const b = await device(uid);
  const scan = { boxes: [[1, 2, 3, 4], [5, 6, 7, 8]], model: 'yolo', kcal: null };
  const m = await a.repo.createMeal(input({ source: { kind: 'camera', ref: 'pi-scan-1' }, originalScan: scan }));
  await a.sync().syncOnce();
  assert.equal(typeof (await remoteOf(a.fs, uid, m.id)).originalScanJson, 'string');
  await b.sync().syncOnce();
  const onB = await b.repo.getMeal(m.id);
  assert.deepEqual(onB.originalScan, scan);
  assert.deepEqual(onB.source, { kind: 'camera', ref: 'pi-scan-1' });
  await b.repo.updateMeal(m.id, onB.revision.rev, { category: 'dinner' });
  await b.sync().syncOnce();
  await a.sync().syncOnce();
  assert.deepEqual((await a.repo.getMeal(m.id)).originalScan, scan);
});

test('outbox coalescing: several offline edits push as one remote revision', async () => {
  const uid = newUid();
  const a = await device(uid);
  const m = await a.repo.createMeal(input());
  await a.repo.updateMeal(m.id, 1, { category: 'dinner' });
  await a.repo.updateMeal(m.id, 2, { items: [rice(300)] });
  assert.equal((await outbox(a)).length, 1);
  await a.sync().push();
  const remote = await remoteOf(a.fs, uid, m.id);
  assert.equal(remote.rev, 1);
  assert.equal(remote.category, 'dinner');
  assert.equal(remote.items[0].grams, 300);
});

test('idempotent push: a crash after the remote commit does not double-apply on retry', async () => {
  const uid = newUid();
  const a = await device(uid);
  const m = await a.repo.createMeal(input());
  const crashing = { ...a.repo, ackPush: async () => { throw new Error('crash before local ack'); } };
  await assert.rejects(createSyncEngine({ repo: crashing, firestore: a.fs, uid }).push(), /crash/);
  assert.equal((await remoteOf(a.fs, uid, m.id)).rev, 1);
  assert.equal((await outbox(a)).length, 1);
  await a.sync().push(); // retry recognises its own edit id
  assert.equal((await remoteOf(a.fs, uid, m.id)).rev, 1);
  assert.equal((await outbox(a)).length, 0);
  assert.equal((await a.repo.getMeal(m.id)).remoteRev, 1);
});

test('lost ack then a new local edit: retry builds on its own landed write, no false conflict', async () => {
  const uid = newUid();
  const a = await device(uid);
  const m = await a.repo.createMeal(input());
  const crashing = { ...a.repo, ackPush: async () => { throw new Error('crash'); } };
  await assert.rejects(createSyncEngine({ repo: crashing, firestore: a.fs, uid }).push());
  await a.repo.updateMeal(m.id, 1, { category: 'snack' });
  const r = await a.sync().syncOnce();
  assert.equal(r.conflicts, 0);
  const remote = await remoteOf(a.fs, uid, m.id);
  assert.equal(remote.rev, 2);
  assert.equal(remote.category, 'snack');
  assert.equal((await outbox(a)).length, 0);
  assert.equal((await a.repo.listConflicts()).length, 0);
});

test('lost ack is also recovered by pull, keeping a newer local edit queued', async () => {
  const uid = newUid();
  const a = await device(uid);
  const m = await a.repo.createMeal(input());
  const crashing = { ...a.repo, ackPush: async () => { throw new Error('crash'); } };
  await assert.rejects(createSyncEngine({ repo: crashing, firestore: a.fs, uid }).push());
  await a.repo.updateMeal(m.id, 1, { category: 'snack' });
  const r = await a.sync().pull();
  assert.equal(r.conflicts, 0);
  const [ob] = await outbox(a);
  assert.equal(ob.base_remote_rev, 1); // rebased onto our own landed write
  assert.equal((await a.repo.getMeal(m.id)).category, 'snack'); // local edit not overwritten
  await a.sync().push();
  assert.equal((await remoteOf(a.fs, uid, m.id)).rev, 2);
});

test('ack race: an edit made while a push is in flight survives the acknowledgement', async () => {
  const uid = newUid();
  const a = await device(uid);
  const m = await a.repo.createMeal(input());
  const racing = {
    ...a.repo,
    ackPush: async (...args) => {
      await a.repo.updateMeal(m.id, 1, { category: 'breakfast' }); // user edits before the ack lands
      return a.repo.ackPush(...args);
    },
  };
  await createSyncEngine({ repo: racing, firestore: a.fs, uid }).push();
  const [ob] = await outbox(a);
  assert.ok(ob, 'newer edit must stay queued');
  assert.equal(ob.base_remote_rev, 1);
  assert.equal((await a.repo.getMeal(m.id)).category, 'breakfast');
  await a.sync().push();
  const remote = await remoteOf(a.fs, uid, m.id);
  assert.equal(remote.rev, 2);
  assert.equal(remote.category, 'breakfast');
});

test('revision conflict on push: detected, preserved, never overwrites; keep local wins after recheck', async () => {
  const uid = newUid();
  const a = await device(uid);
  const b = await device(uid);
  const m = await a.repo.createMeal(input());
  await a.sync().syncOnce();
  await b.sync().syncOnce();
  await a.repo.updateMeal(m.id, 1, { category: 'dinner' });
  await b.repo.updateMeal(m.id, 1, { category: 'snack' });
  await b.sync().push();

  const r = await a.sync().push();
  assert.deepEqual(r, { pushed: 0, conflicts: 1 });
  assert.equal((await remoteOf(a.fs, uid, m.id)).category, 'snack'); // B's edit not overwritten
  assert.equal((await outbox(a)).length, 1); // A's edit preserved
  const [c] = await a.repo.listConflicts();
  assert.equal(c.mealId, m.id);
  assert.equal(c.remote.category, 'snack');
  assert.equal(c.remoteRev, 2);
  assert.equal((await a.repo.getMeal(m.id)).category, 'dinner');
  assert.deepEqual(await a.sync().push(), { pushed: 0, conflicts: 0 }); // conflicted rows are not retried blindly

  await a.repo.resolveConflict(c.id, 'local');
  await a.sync().push();
  const remote = await remoteOf(a.fs, uid, m.id);
  assert.equal(remote.rev, 3);
  assert.equal(remote.category, 'dinner');
  await b.sync().pull();
  assert.equal((await b.repo.getMeal(m.id)).category, 'dinner');
});

test('keep local rechecks the latest remote rev: a newer cloud edit reopens the conflict', async () => {
  const uid = newUid();
  const a = await device(uid);
  const b = await device(uid);
  const m = await a.repo.createMeal(input());
  await a.sync().syncOnce();
  await b.sync().syncOnce();
  await a.repo.updateMeal(m.id, 1, { category: 'dinner' });
  await b.repo.updateMeal(m.id, 1, { category: 'snack' });
  await b.sync().push();
  await a.sync().push();
  const [c] = await a.repo.listConflicts();

  await b.repo.updateMeal(m.id, 2, { category: 'breakfast' }); // cloud moves again after A compared
  await b.sync().push();
  await a.repo.resolveConflict(c.id, 'local');
  const r = await a.sync().push();
  assert.equal(r.conflicts, 1);
  assert.equal((await remoteOf(a.fs, uid, m.id)).category, 'breakfast');
  const [again] = await a.repo.listConflicts();
  assert.equal(again.remoteRev, 3);
  assert.equal(again.remote.category, 'breakfast');
});

test('keep cloud replaces the local edit and clears the outbox', async () => {
  const uid = newUid();
  const a = await device(uid);
  const b = await device(uid);
  const m = await a.repo.createMeal(input());
  await a.sync().syncOnce();
  await b.sync().syncOnce();
  await a.repo.updateMeal(m.id, 1, { category: 'dinner' });
  await b.repo.updateMeal(m.id, 1, { category: 'snack' });
  await b.sync().push();
  const pulled = await a.sync().pull(); // conflict detected on pull, local content untouched
  assert.equal(pulled.conflicts, 1);
  assert.equal((await a.repo.getMeal(m.id)).category, 'dinner');
  const [c] = await a.repo.listConflicts();
  await a.repo.resolveConflict(c.id, 'cloud');
  const local = await a.repo.getMeal(m.id);
  assert.equal(local.category, 'snack');
  assert.equal(local.remoteRev, 2);
  assert.equal((await outbox(a)).length, 0);
  assert.equal((await a.repo.listConflicts()).length, 0);
  assert.equal((await a.sync().syncOnce()).pushed, 0);
});

test('tombstones propagate: a soft delete on A hides the meal on B', async () => {
  const uid = newUid();
  const a = await device(uid);
  const b = await device(uid);
  const m = await a.repo.createMeal(input());
  await a.sync().syncOnce();
  await b.sync().syncOnce();
  assert.equal((await b.repo.listMeals()).length, 1);
  await a.repo.deleteMeal(m.id, 1);
  await a.sync().syncOnce();
  const remote = await remoteOf(a.fs, uid, m.id);
  assert.ok(remote.deletedAt, 'remote keeps a tombstone, not a hard delete');
  await b.sync().syncOnce();
  assert.equal((await b.repo.listMeals()).length, 0);
  assert.ok((await b.repo.getMeal(m.id)).deletedAt);
});

test('cursor pagination handles equal server timestamps (one batch) without skipping or repeating', async () => {
  const uid = newUid();
  const a = await device(uid);
  const fs = client(uid);
  const batch = writeBatch(fs); // one commit = identical serverUpdatedAt for every doc
  const ids = Array.from({ length: 7 }, (_, i) => `meal-${i}`);
  for (const id of ids) {
    batch.set(doc(fs, 'users', uid, 'meals', id), {
      ...validDoc(uid, id, { dayKey: '1970-01-01', loggedAt: 1 }),
      items: [{ id: `i-${id}`, ...rice() }],
    });
  }
  await batch.commit();
  const stamps = (await getDocs(collection(fs, 'users', uid, 'meals'))).docs.map((d) => d.get('serverUpdatedAt').toMillis());
  assert.equal(new Set(stamps).size, 1, 'precondition: all docs tie on serverUpdatedAt');
  const seen = [];
  const recording = { ...a.repo, applyRemotePage: (u, docs, c) => { seen.push(...docs.map((d) => d.id)); return a.repo.applyRemotePage(u, docs, c); } };
  const r = await createSyncEngine({ repo: recording, firestore: a.fs, uid, pageSize: 3 }).pull();
  assert.equal(r.pulled, 7);
  assert.deepEqual([...seen].sort(), ids);
  assert.equal(new Set(seen).size, 7);
  assert.equal((await a.repo.listMeals()).length, 7);
  assert.equal((await a.sync({ pageSize: 3 }).pull()).pulled, 0); // cursor durable at the end
});

test('interrupted pull: the cursor advances only with durably applied pages', async () => {
  const uid = newUid();
  const dir = mkdtempSync(join(tmpdir(), 'platego-sync-'));
  try {
    const src = await device(uid);
    for (let i = 0; i < 5; i++) await src.repo.createMeal(input({ loggedAt: Date.UTC(2026, 9, 10, i) }));
    await src.sync().push();

    const path = join(dir, 'b.db');
    const b = await device(uid, path);
    let pages = 0;
    const crashOnPage2 = {
      ...b.repo,
      applyRemotePage: async (...args) => {
        if (++pages === 2) throw new Error('app killed');
        return b.repo.applyRemotePage(...args);
      },
    };
    await assert.rejects(createSyncEngine({ repo: crashOnPage2, firestore: b.fs, uid, pageSize: 2 }).pull(), /killed/);
    assert.equal((await b.repo.listMeals()).length, 2);
    await b.db.closeAsync();

    const restarted = await device(uid, path); // app restart on the same database file
    const r = await restarted.sync({ pageSize: 2 }).pull();
    assert.equal(r.pulled, 3);
    assert.equal((await restarted.repo.listMeals()).length, 5);
    await restarted.db.closeAsync();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('restart recovery: unsynchronized edits and a half-finished push survive a restart', async () => {
  const uid = newUid();
  const dir = mkdtempSync(join(tmpdir(), 'platego-sync-'));
  try {
    const path = join(dir, 'a.db');
    const a = await device(uid, path);
    const m = await a.repo.createMeal(input());
    const crashing = { ...a.repo, ackPush: async () => { throw new Error('killed'); } };
    await assert.rejects(createSyncEngine({ repo: crashing, firestore: a.fs, uid }).push());
    await a.repo.updateMeal(m.id, 1, { category: 'snack' });
    await a.db.closeAsync();

    const again = await device(uid, path);
    assert.equal((await again.repo.listOutbox()).length, 1);
    const r = await again.sync().syncOnce();
    assert.equal(r.conflicts, 0);
    assert.equal((await remoteOf(again.fs, uid, m.id)).category, 'snack');
    assert.equal((await again.repo.listOutbox()).length, 0);
    await again.db.closeAsync();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('network failure: push is retried with backoff and nothing is lost', async () => {
  const uid = newUid();
  const a = await device(uid);
  const m = await a.repo.createMeal(input());
  let t = Date.now();
  const offline = createSyncEngine({ repo: a.repo, firestore: client(uid, '127.0.0.1:1'), uid, now: () => t });
  await assert.rejects(offline.syncOnce(), (e) => e.code === 'unavailable');
  const [ob] = await outbox(a);
  assert.equal(ob.attempts, 1);
  assert.equal(ob.next_attempt_at, t + 2000);
  assert.equal(ob.last_error, 'unavailable');
  assert.deepEqual(await a.repo.duePushes(uid, t), []); // in backoff, not hammered
  assert.equal((await a.repo.getMeal(m.id)).revision.rev, 1); // local data untouched
  t = ob.next_attempt_at;
  const online = a.sync({ now: () => t });
  assert.equal((await online.push()).pushed, 1);
  assert.equal((await outbox(a)).length, 0);
  assert.equal((await remoteOf(a.fs, uid, m.id)).rev, 1);
});

test('identity isolation: a round for one account cannot write after the identity switches', async () => {
  const alice = newUid();
  const bob = newUid();
  const d = await device(alice);
  const m = await d.repo.createMeal(input());
  const engine = d.sync();
  await d.repo.setIdentity(bob); // account switch while alice's engine still exists
  await assert.rejects(engine.push(), IdentityChangedError);
  assert.equal((await getDocs(collection(client(alice), 'users', alice, 'meals'))).size, 0);
  // bob's view and outbox never include alice's meal; bob cannot read alice's cloud data
  assert.equal(await d.repo.getMeal(m.id), null);
  assert.equal((await d.repo.listOutbox()).length, 0);
  await rules(getDocs(collection(client(bob), 'users', alice, 'meals')));
  // pulled pages are refused too
  await assert.rejects(d.repo.applyRemotePage(alice, [], { s: 0, n: 0, id: 'x' }), IdentityChangedError);
});

test('demo samples never reach the cloud', async () => {
  const uid = newUid();
  const a = await device(uid);
  await a.repo.createMeal(input({ source: { kind: 'sample' }, items: [{ ...rice(), name: 'SAMPLE rice' }] }));
  assert.equal((await outbox(a)).length, 0);
  await a.sync().syncOnce();
  assert.equal((await getDocs(collection(a.fs, 'users', uid, 'meals'))).size, 0);
});

test('invalid remote data is refused and the cursor does not move past it', async () => {
  const uid = newUid();
  const a = await device(uid);
  await assert.rejects(
    a.repo.applyRemotePage(uid, [{ id: 'x', ownerUid: 'someone-else', rev: 1 }], { s: 1, n: 0, id: 'x' }),
    /foreign owner/
  );
  assert.equal(await a.repo.getPullCursor(uid), null);
});
