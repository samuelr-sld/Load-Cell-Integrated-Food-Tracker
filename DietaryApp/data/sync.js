// Bidirectional sync between the local SQLite diary and Firestore users/{uid}/meals/{mealId}.
// Push: each outbox row goes through a Firestore transaction that checks the remote rev the edit
// was based on (optimistic concurrency) and is idempotent by edit id. Pull: pages ordered by
// (serverUpdatedAt, document id) from a durable cursor that only advances with the applied page.
import {
  collection, doc, documentId, getDocsFromServer, limit, orderBy, query, runTransaction,
  serverTimestamp, startAfter, Timestamp,
} from 'firebase/firestore';

export class SyncConflict extends Error {
  constructor(remote) {
    super(`meal ${remote.id} changed in the cloud`);
    this.remote = remote;
  }
}

// originalScan travels as a JSON string: Firestore rejects nested arrays, which scan payloads may contain.
const fromDoc = ({ serverUpdatedAt, originalScanJson, ...rest }) => ({
  ...rest, originalScan: originalScanJson == null ? null : JSON.parse(originalScanJson),
});

const toRemote = (m, uid, rev, editId) => ({
  id: m.id, ownerUid: uid, loggedAt: m.loggedAt, tzOffsetMin: m.tzOffsetMin, dayKey: m.dayKey,
  category: m.category, items: m.items, source: { kind: m.source.kind, ref: m.source.ref ?? null },
  originalScanJson: m.originalScan == null ? null : JSON.stringify(m.originalScan), createdAt: m.createdAt, deletedAt: m.deletedAt,
  rev, lastEditId: editId, serverUpdatedAt: serverTimestamp(),
});

export const createSyncEngine = ({ repo, firestore, uid, pageSize = 100, now = () => Date.now() }) => {
  const meals = collection(firestore, 'users', uid, 'meals');

  const pushOne = async (mealId) => {
    const p = await repo.beginPush(uid, mealId);
    if (!p) return null;
    const ref = doc(meals, mealId);
    let rev;
    try {
      rev = await runTransaction(firestore, async (t) => {
        const snap = await t.get(ref);
        const remote = snap.exists() ? snap.data() : null;
        if (remote?.lastEditId === p.editId) return remote.rev; // landed on an earlier attempt; ack was lost
        // Remote moved past our base and not by one of our own sent edits: never overwrite it.
        // (A missing doc is created; rules forbid deletes, so it was never pushed or was removed out of band.)
        if (remote && remote.rev !== p.baseRemoteRev && !p.sentEditIds.includes(remote.lastEditId)) {
          throw new SyncConflict(fromDoc(remote));
        }
        const next = remote ? remote.rev + 1 : 1;
        t.set(ref, toRemote(p.meal, uid, next, p.editId));
        return next;
      });
    } catch (e) {
      if (e instanceof SyncConflict) {
        await repo.recordConflict(uid, mealId, e.remote);
        return 'conflict';
      }
      await repo.failPush(uid, mealId, e?.code ?? e?.message, now());
      throw e;
    }
    await repo.ackPush(uid, mealId, p.editId, rev);
    return 'pushed';
  };

  const push = async () => {
    const s = { pushed: 0, conflicts: 0 };
    // A network error aborts the round (the failed row gets its backoff); the rest keep their schedule.
    for (const id of await repo.duePushes(uid, now())) {
      const r = await pushOne(id);
      if (r === 'pushed') s.pushed++;
      if (r === 'conflict') s.conflicts++;
    }
    return s;
  };

  // Cursor keeps Firestore's full timestamp (seconds + nanoseconds; converting to ms would drop
  // precision and could skip docs) plus the doc id to break ties between docs with equal timestamps.
  const pull = async () => {
    const s = { pulled: 0, applied: 0, conflicts: 0 };
    let cursor = await repo.getPullCursor(uid);
    for (;;) {
      const after = cursor ? [startAfter(new Timestamp(cursor.s, cursor.n), cursor.id)] : [];
      const snap = await getDocsFromServer(
        query(meals, orderBy('serverUpdatedAt'), orderBy(documentId()), ...after, limit(pageSize))
      );
      if (snap.empty) break;
      const last = snap.docs[snap.docs.length - 1];
      const ts = last.get('serverUpdatedAt');
      cursor = { s: ts.seconds, n: ts.nanoseconds, id: last.id };
      const r = await repo.applyRemotePage(uid, snap.docs.map((d) => fromDoc(d.data())), cursor);
      s.pulled += snap.size;
      s.applied += r.applied;
      s.conflicts += r.conflicts;
      if (snap.size < pageSize) break;
    }
    return s;
  };

  return {
    push,
    pull,
    async syncOnce() {
      const a = await push();
      const b = await pull();
      return { pushed: a.pushed, pulled: b.pulled, applied: b.applied, conflicts: a.conflicts + b.conflicts };
    },
  };
};

// Runs sync rounds one at a time: now, after local changes (debounced) and on an interval.
// stop() prevents further rounds; a round already running finishes, and its local writes are
// refused by the repository if the identity changed meanwhile.
export const startAutoSync = ({ repo, firestore, uid, onStatus = () => {}, intervalMs = 30000, debounceMs = 1500 }) => {
  const engine = createSyncEngine({ repo, firestore, uid });
  let stopped = false;
  let running = false;
  let again = false;
  let debounce = null;

  const run = async () => {
    if (stopped) return;
    if (running) {
      again = true;
      return;
    }
    running = true;
    onStatus({ state: 'syncing' });
    try {
      const r = await engine.syncOnce();
      if (!stopped) onStatus({ state: 'idle', lastSyncAt: Date.now(), ...r });
    } catch (e) {
      if (!stopped) onStatus({ state: 'error', error: e?.code ?? e?.message ?? String(e) });
    }
    running = false;
    if (again) {
      again = false;
      run();
    }
  };

  const kick = () => {
    clearTimeout(debounce);
    debounce = setTimeout(run, debounceMs);
  };
  const offChange = repo.onChange(kick);
  const interval = setInterval(run, intervalMs);
  run();

  return {
    syncNow: run,
    stop() {
      stopped = true;
      clearTimeout(debounce);
      clearInterval(interval);
      offChange();
    },
  };
};
