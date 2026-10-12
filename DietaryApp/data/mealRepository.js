// Local source of truth for canonical meals. Every write updates the meal and its outbox row
// in one exclusive transaction. Reads and writes are scoped to the current identity:
// a Firebase UID (owned rows) or, before any sign-in, rows still pending ownership.
import { mealTotals, validateMeal, localDayKey, softDelete, CATEGORIES } from '../domain/meal.js';
import { normalizePreferences, DEFAULT_PREFERENCES } from '../domain/preferences.js';

export class RevisionConflictError extends Error {}
export class DuplicateScanError extends Error {}
export class MealNotFoundError extends Error {}
// The signed-in identity changed while a sync round was running; its local writes are refused.
export class IdentityChangedError extends Error {}

const EDITABLE = ['loggedAt', 'tzOffsetMin', 'category', 'items'];
const BOUND_UID = 'bound_uid';
const MAX_ASSISTANT_MESSAGES = 200;
const cursorKey = (uid) => `pull_cursor:${uid}`;

const rowToMeal = (r) => ({
  id: r.id,
  ownerStatus: r.owner_status,
  ownerUid: r.owner_uid,
  loggedAt: r.logged_at,
  tzOffsetMin: r.tz_offset_min,
  dayKey: r.day_key,
  category: r.category,
  items: JSON.parse(r.items_json),
  totals: { calories: r.calories, protein: r.protein, carbs: r.carbs, fat: r.fat },
  source: { kind: r.source_kind, ref: r.source_ref },
  originalScan: r.original_scan_json == null ? null : JSON.parse(r.original_scan_json),
  revision: { rev: r.rev, updatedAt: r.updated_at },
  remoteRev: r.remote_rev,
  createdAt: r.created_at,
  deletedAt: r.deleted_at,
});

export const createMealRepository = (db) => {
  let uid = null;
  const listeners = new Set();
  const emit = () => listeners.forEach((fn) => fn());

  // expo-sqlite runs each exclusive transaction on its own connection and a second concurrent writer
  // fails with "database is locked", so transactions (user edits and background sync) run one at a time.
  let queue = Promise.resolve();
  const tx = (task) => {
    const run = queue.then(() => db.withExclusiveTransactionAsync(task));
    queue = run.catch(() => {});
    return run;
  };

  // RFC 4122 v4 from SQLite's CSPRNG (randomblob), so no crypto polyfill or extra dependency.
  const uuid = async (conn = db) => {
    const h = (await conn.getFirstAsync('SELECT lower(hex(randomblob(16))) AS h')).h;
    const v = ((parseInt(h[16], 16) & 3) | 8).toString(16);
    return `${h.slice(0, 8)}-${h.slice(8, 12)}-4${h.slice(13, 16)}-${v}${h.slice(17, 20)}-${h.slice(20)}`;
  };

  const owner = () => (uid ? ['owner_uid = ?', [uid]] : ["owner_status = 'pending'", []]);
  const assertIdentity = (syncUid) => {
    if (!syncUid || uid !== syncUid) throw new IdentityChangedError(`sync for ${syncUid} but identity is ${uid}`);
  };

  const assertValid = (meal) => {
    const errors = validateMeal(meal);
    if (meal.tzOffsetMin != null && !Number.isInteger(meal.tzOffsetMin)) errors.push('tzOffsetMin must be an integer');
    if (meal.source.kind === 'camera' && !meal.source.ref) errors.push('camera meals need source.ref');
    if (errors.length) throw new TypeError(`Invalid meal: ${errors.join('; ')}`);
  };

  const withIds = async (items, conn) => {
    if (!Array.isArray(items)) return items;
    const out = [];
    for (const it of items) out.push(it?.id ? it : { ...it, id: await uuid(conn) });
    return out;
  };

  // Coalesces into the single pending row for this meal; base_remote_rev keeps the first base.
  // Demo samples are local-only and never queued for the cloud.
  const upsertOutbox = (txn, meal, op, now, editId) =>
    meal.source.kind === 'sample' ? null : txn.runAsync(
      `INSERT INTO outbox (meal_id, op, edit_id, meal_rev, base_remote_rev, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT (meal_id) DO UPDATE SET op = excluded.op, edit_id = excluded.edit_id,
         meal_rev = excluded.meal_rev, attempts = 0, last_error = NULL, next_attempt_at = NULL,
         updated_at = excluded.updated_at`,
      [meal.id, op, editId, meal.revision.rev, meal.remoteRev ?? null, now, now]
    );

  const loadOwned = async (conn, id) => {
    const [clause, params] = owner();
    const row = await conn.getFirstAsync(`SELECT * FROM meals WHERE id = ? AND ${clause}`, [id, ...params]);
    return row ? rowToMeal(row) : null;
  };

  const clearOutbox = async (txn, mealId) => {
    await txn.runAsync('DELETE FROM outbox WHERE meal_id = ?', [mealId]);
    await txn.runAsync('DELETE FROM sent_edits WHERE meal_id = ?', [mealId]);
  };

  // Remote document -> validated local meal. Totals are re-derived from items, never trusted.
  const fromRemote = (r, syncUid, base) => {
    if (r?.ownerUid !== syncUid) throw new TypeError(`remote meal ${r?.id} has a foreign owner`);
    if (!Number.isInteger(r.rev) || r.rev < 1) throw new TypeError(`remote meal ${r.id} has an invalid rev`);
    if (r.source?.kind === 'sample') throw new TypeError(`remote meal ${r.id} is a demo sample`);
    const meal = {
      id: r.id, loggedAt: r.loggedAt ?? null, tzOffsetMin: r.tzOffsetMin ?? null, dayKey: r.dayKey ?? null,
      category: r.category, items: r.items, source: { kind: r.source?.kind, ref: r.source?.ref ?? null },
      originalScan: r.originalScan ?? null, createdAt: r.createdAt, deletedAt: r.deletedAt ?? null,
      revision: { rev: (base?.revision.rev ?? 0) + 1, updatedAt: Date.now() },
    };
    if (meal.loggedAt !== null && meal.dayKey !== localDayKey(meal.loggedAt, meal.tzOffsetMin ?? 0)) {
      throw new TypeError(`remote meal ${r.id} has an inconsistent dayKey`);
    }
    assertValid(meal);
    meal.totals = mealTotals(meal.items);
    return meal;
  };

  // Writes remote content over the local row (or inserts it). Origin fields are guarded by triggers.
  const writeRemote = async (txn, r, syncUid, cur) => {
    const m = fromRemote(r, syncUid, cur);
    const t = m.totals;
    if (cur) {
      await txn.runAsync(
        `UPDATE meals SET logged_at = ?, tz_offset_min = ?, day_key = ?, category = ?, items_json = ?,
           calories = ?, protein = ?, carbs = ?, fat = ?, rev = ?, remote_rev = ?, updated_at = ?, deleted_at = ?
         WHERE id = ? AND owner_uid = ?`,
        [m.loggedAt, m.tzOffsetMin, m.dayKey, m.category, JSON.stringify(m.items), t.calories, t.protein, t.carbs,
          t.fat, m.revision.rev, r.rev, m.revision.updatedAt, m.deletedAt, m.id, syncUid]
      );
    } else {
      await txn.runAsync(
        `INSERT INTO meals (id, owner_status, owner_uid, logged_at, tz_offset_min, day_key, category, items_json,
           calories, protein, carbs, fat, source_kind, source_ref, original_scan_json, rev, remote_rev,
           created_at, updated_at, deleted_at)
         VALUES (?, 'owned', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)`,
        [m.id, syncUid, m.loggedAt, m.tzOffsetMin, m.dayKey, m.category, JSON.stringify(m.items), t.calories,
          t.protein, t.carbs, t.fat, m.source.kind, m.source.ref,
          m.originalScan === null ? null : JSON.stringify(m.originalScan), r.rev, m.createdAt,
          m.revision.updatedAt, m.deletedAt]
      );
    }
  };

  // One open conflict per meal; a newer remote version replaces the stored one.
  const upsertConflict = async (txn, mealId, r) => {
    const local = await txn.getFirstAsync('SELECT * FROM meals WHERE id = ?', [mealId]);
    const open = await txn.getFirstAsync('SELECT id FROM meal_conflicts WHERE meal_id = ? AND resolved_at IS NULL', [mealId]);
    const vals = [local.rev, r?.rev ?? null, JSON.stringify(rowToMeal(local)), r == null ? null : JSON.stringify(r), Date.now()];
    if (open) {
      await txn.runAsync(
        'UPDATE meal_conflicts SET local_rev = ?, remote_rev = ?, local_json = ?, remote_json = ?, detected_at = ? WHERE id = ?',
        [...vals, open.id]
      );
    } else {
      await txn.runAsync(
        `INSERT INTO meal_conflicts (id, meal_id, local_rev, remote_rev, local_json, remote_json, detected_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
        [await uuid(txn), mealId, ...vals]
      );
    }
  };

  // Shared path for update and soft delete: optimistic rev check in SQL plus outbox coalescing.
  const writeExisting = async (id, expectedRev, change, op) => {
    let saved;
    await tx(async (txn) => {
      const cur = await loadOwned(txn, id);
      if (!cur) throw new MealNotFoundError(id);
      if (cur.revision.rev !== expectedRev || cur.deletedAt != null) {
        throw new RevisionConflictError(`meal ${id} is at rev ${cur.revision.rev}${cur.deletedAt != null ? ' (deleted)' : ''}`);
      }
      const now = Date.now();
      const next = { ...change(cur, now), revision: { rev: cur.revision.rev + 1, updatedAt: now } };
      next.items = await withIds(next.items, txn);
      next.totals = mealTotals(next.items);
      assertValid(next);
      const r = await txn.runAsync(
        `UPDATE meals SET logged_at = ?, tz_offset_min = ?, day_key = ?, category = ?, items_json = ?,
           calories = ?, protein = ?, carbs = ?, fat = ?, rev = ?, updated_at = ?, deleted_at = ?
         WHERE id = ? AND rev = ?`,
        [next.loggedAt, next.tzOffsetMin, next.dayKey, next.category, JSON.stringify(next.items),
          next.totals.calories, next.totals.protein, next.totals.carbs, next.totals.fat,
          next.revision.rev, now, next.deletedAt, id, expectedRev]
      );
      if (r.changes !== 1) throw new RevisionConflictError(`meal ${id} changed concurrently`);
      await upsertOutbox(txn, next, op, now, await uuid(txn));
      saved = next;
    });
    emit();
    return saved;
  };

  return {
    // Binds the identity and adopts pending meals atomically. Pending meals go only to the first
    // UID this install ever bound; any other UID leaves them pending (never merged across identities).
    async setIdentity(nextUid) {
      if (!nextUid) {
        uid = null;
        emit();
        return { adopted: 0, pendingLeft: 0 };
      }
      let result;
      await tx(async (txn) => {
        const bound = (await txn.getFirstAsync('SELECT value FROM sync_state WHERE key = ?', [BOUND_UID]))?.value;
        if (bound == null) await txn.runAsync('INSERT INTO sync_state (key, value) VALUES (?, ?)', [BOUND_UID, nextUid]);
        let adopted = 0;
        if (bound == null || bound === nextUid) {
          adopted = (await txn.runAsync(
            "UPDATE meals SET owner_status = 'owned', owner_uid = ? WHERE owner_status = 'pending'", [nextUid]
          )).changes;
          // Pre-sign-in preferences follow the meals, unless this account already has its own.
          await txn.runAsync(
            "UPDATE user_preferences SET owner_key = ? WHERE owner_key = '' AND NOT EXISTS (SELECT 1 FROM user_preferences WHERE owner_key = ?)",
            [nextUid, nextUid]
          );
        }
        const left = (await txn.getFirstAsync("SELECT count(*) AS n FROM meals WHERE owner_status = 'pending'")).n;
        result = { adopted, pendingLeft: left };
      });
      uid = nextUid;
      emit();
      return result;
    },

    async listMeals({ fromDayKey = '0000-00-00', toDayKey = '9999-99-99' } = {}) {
      const [clause, params] = owner();
      const rows = await db.getAllAsync(
        `SELECT * FROM meals WHERE ${clause} AND deleted_at IS NULL AND day_key BETWEEN ? AND ?
         ORDER BY logged_at DESC`,
        [...params, fromDayKey, toDayKey]
      );
      return rows.map(rowToMeal);
    },

    async listUndated() {
      const [clause, params] = owner();
      const rows = await db.getAllAsync(
        `SELECT * FROM meals WHERE ${clause} AND deleted_at IS NULL AND day_key IS NULL ORDER BY created_at DESC`,
        params
      );
      return rows.map(rowToMeal);
    },

    getMeal: (id) => loadOwned(db, id),

    // Input: { loggedAt|null, tzOffsetMin?, category?, items, source: { kind, ref? }, originalScan? }.
    // Totals are always derived from items; any totals on the input are ignored.
    async createMeal(input) {
      const now = Date.now();
      const loggedAt = input.loggedAt ?? null;
      const tzOffsetMin = loggedAt === null ? null : input.tzOffsetMin ?? -new Date(loggedAt).getTimezoneOffset();
      let meal;
      try {
        await tx(async (txn) => {
          const items = await withIds(input.items, txn);
          meal = {
            id: await uuid(txn),
            ownerStatus: uid ? 'owned' : 'pending',
            ownerUid: uid,
            loggedAt,
            tzOffsetMin,
            dayKey: loggedAt === null ? null : localDayKey(loggedAt, tzOffsetMin),
            category: input.category ?? 'unknown',
            items,
            totals: null,
            source: { kind: input.source?.kind, ref: input.source?.ref ?? null },
            originalScan: input.originalScan ?? null,
            revision: { rev: 1, updatedAt: now },
            remoteRev: null,
            createdAt: now,
            deletedAt: null,
          };
          assertValid(meal);
          meal.totals = mealTotals(items);
          await txn.runAsync(
            `INSERT INTO meals (id, owner_status, owner_uid, logged_at, tz_offset_min, day_key, category, items_json,
               calories, protein, carbs, fat, source_kind, source_ref, original_scan_json, rev, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)`,
            [meal.id, meal.ownerStatus, meal.ownerUid, loggedAt, tzOffsetMin, meal.dayKey, meal.category,
              JSON.stringify(items), meal.totals.calories, meal.totals.protein, meal.totals.carbs, meal.totals.fat,
              meal.source.kind, meal.source.ref,
              meal.originalScan === null ? null : JSON.stringify(meal.originalScan), now, now]
          );
          await upsertOutbox(txn, meal, 'upsert', now, await uuid(txn));
        });
      } catch (e) {
        if (/UNIQUE constraint failed: meals\.source_ref/.test(e?.message)) {
          throw new DuplicateScanError(`scan ${input.source?.ref} already imported`);
        }
        throw e;
      }
      emit();
      return meal;
    },

    // patch may only touch EDITABLE fields; dayKey is recomputed only if the time or offset changed.
    updateMeal(id, expectedRev, patch) {
      const bad = Object.keys(patch).filter((k) => !EDITABLE.includes(k));
      if (bad.length) return Promise.reject(new TypeError(`Not editable: ${bad.join(', ')}`));
      if ('category' in patch && !CATEGORIES.includes(patch.category)) {
        return Promise.reject(new TypeError('invalid category'));
      }
      return writeExisting(id, expectedRev, (cur) => {
        const next = { ...cur, ...patch };
        if (next.loggedAt !== cur.loggedAt || next.tzOffsetMin !== cur.tzOffsetMin) {
          if (next.loggedAt === null) {
            next.tzOffsetMin = null;
            next.dayKey = null;
          } else {
            next.tzOffsetMin ??= -new Date(next.loggedAt).getTimezoneOffset();
            next.dayKey = localDayKey(next.loggedAt, next.tzOffsetMin);
          }
        }
        return next;
      }, 'upsert');
    },

    deleteMeal: (id, expectedRev) => writeExisting(id, expectedRev, (cur, now) => softDelete(cur, now), 'delete'),

    // Pending changes for the current identity only.
    async listOutbox() {
      const [clause, params] = owner();
      return db.getAllAsync(
        `SELECT o.* FROM outbox o JOIN meals m ON m.id = o.meal_id WHERE m.${clause} ORDER BY o.updated_at`,
        params
      );
    },

    // ---- Sync primitives (used by data/sync.js). Each takes the UID the sync round runs for and
    // refuses to write if the identity has changed since, so one account's round never touches another's rows.

    // Meal ids whose outbox row is due, oldest first, skipping meals with an open conflict.
    async duePushes(syncUid, now) {
      const rows = await db.getAllAsync(
        `SELECT o.meal_id FROM outbox o JOIN meals m ON m.id = o.meal_id
         WHERE m.owner_uid = ? AND (o.next_attempt_at IS NULL OR o.next_attempt_at <= ?)
           AND NOT EXISTS (SELECT 1 FROM meal_conflicts c WHERE c.meal_id = o.meal_id AND c.resolved_at IS NULL)
         ORDER BY o.updated_at`,
        [syncUid, now]
      );
      return rows.map((r) => r.meal_id);
    },

    // Snapshots the meal and its pending edit, and durably records the edit id as sent *before* the
    // network call, so a retry after a lost ack recognises the write as its own.
    async beginPush(syncUid, mealId) {
      let out = null;
      await tx(async (txn) => {
        assertIdentity(syncUid);
        const ob = await txn.getFirstAsync('SELECT * FROM outbox WHERE meal_id = ?', [mealId]);
        const row = await txn.getFirstAsync('SELECT * FROM meals WHERE id = ? AND owner_uid = ?', [mealId, syncUid]);
        if (!ob || !row) return;
        await txn.runAsync('INSERT OR IGNORE INTO sent_edits (meal_id, edit_id) VALUES (?, ?)', [mealId, ob.edit_id]);
        const sent = await txn.getAllAsync('SELECT edit_id FROM sent_edits WHERE meal_id = ?', [mealId]);
        out = { meal: rowToMeal(row), editId: ob.edit_id, baseRemoteRev: ob.base_remote_rev, sentEditIds: sent.map((s) => s.edit_id) };
      });
      return out;
    },

    // Remote accepted editId at remoteRev. Removes the outbox row only if no newer local edit was
    // coalesced into it meanwhile; otherwise rebases that newer edit onto remoteRev.
    async ackPush(syncUid, mealId, editId, remoteRev) {
      await tx(async (txn) => {
        assertIdentity(syncUid);
        await txn.runAsync(
          'UPDATE meals SET remote_rev = ? WHERE id = ? AND owner_uid = ? AND (remote_rev IS NULL OR remote_rev < ?)',
          [remoteRev, mealId, syncUid, remoteRev]
        );
        const ob = await txn.getFirstAsync('SELECT edit_id FROM outbox WHERE meal_id = ?', [mealId]);
        if (ob?.edit_id === editId) {
          await clearOutbox(txn, mealId);
        } else if (ob) {
          await txn.runAsync(
            'UPDATE outbox SET base_remote_rev = ?, attempts = 0, last_error = NULL, next_attempt_at = NULL WHERE meal_id = ?',
            [remoteRev, mealId]
          );
          await txn.runAsync('DELETE FROM sent_edits WHERE meal_id = ?', [mealId]);
        }
      });
      emit();
    },

    // Exponential backoff per meal: 2s, 4s, 8s ... capped at 5 minutes; a new local edit resets it.
    async failPush(syncUid, mealId, message, now) {
      await tx(async (txn) => {
        assertIdentity(syncUid);
        await txn.runAsync(
          `UPDATE outbox SET attempts = attempts + 1, last_error = ?,
             next_attempt_at = ? + min(300000, 1000 * (1 << min(attempts + 1, 9))) WHERE meal_id = ?`,
          [String(message).slice(0, 500), now, mealId]
        );
      });
    },

    async recordConflict(syncUid, mealId, remote) {
      await tx(async (txn) => {
        assertIdentity(syncUid);
        await upsertConflict(txn, mealId, remote);
      });
      emit();
    },

    async getPullCursor(syncUid) {
      const v = (await db.getFirstAsync('SELECT value FROM sync_state WHERE key = ?', [cursorKey(syncUid)]))?.value;
      return v == null ? null : JSON.parse(v);
    },

    // Applies one pulled page and advances the cursor in the same transaction: either both are
    // durable or neither is, so a crash re-pulls the page. Never overwrites a pending local edit.
    async applyRemotePage(syncUid, docs, cursor) {
      const stats = { applied: 0, skipped: 0, acked: 0, conflicts: 0 };
      await tx(async (txn) => {
        assertIdentity(syncUid);
        for (const r of docs) {
          const row = await txn.getFirstAsync('SELECT * FROM meals WHERE id = ?', [r.id]);
          if (row && row.owner_uid !== syncUid) throw new Error(`meal ${r.id} belongs to another local identity`);
          const cur = row ? rowToMeal(row) : null;
          const ob = await txn.getFirstAsync('SELECT edit_id FROM outbox WHERE meal_id = ?', [r.id]);
          if (cur && r.rev <= (cur.remoteRev ?? 0)) {
            stats.skipped++;
          } else if (!ob) {
            await writeRemote(txn, r, syncUid, cur);
            stats.applied++;
          } else if (r.lastEditId === ob.edit_id) {
            // Our own push landed but its ack was lost: local content already equals remote.
            await txn.runAsync('UPDATE meals SET remote_rev = ? WHERE id = ?', [r.rev, r.id]);
            await clearOutbox(txn, r.id);
            stats.acked++;
          } else if (await txn.getFirstAsync('SELECT 1 FROM sent_edits WHERE meal_id = ? AND edit_id = ?', [r.id, r.lastEditId])) {
            // An earlier edit of ours landed; the newer local edit is rebased onto it.
            await txn.runAsync('UPDATE meals SET remote_rev = ? WHERE id = ?', [r.rev, r.id]);
            await txn.runAsync('UPDATE outbox SET base_remote_rev = ? WHERE meal_id = ?', [r.rev, r.id]);
            await txn.runAsync('DELETE FROM sent_edits WHERE meal_id = ?', [r.id]);
            stats.acked++;
          } else {
            await upsertConflict(txn, r.id, r);
            stats.conflicts++;
          }
        }
        await txn.runAsync(
          'INSERT INTO sync_state (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value',
          [cursorKey(syncUid), JSON.stringify(cursor)]
        );
      });
      if (stats.applied || stats.acked || stats.conflicts) emit();
      return stats;
    },

    // ---- Goals and preferences (current identity only; stored on this device, not synced).

    // Defaults with rev 0 when nothing is stored yet.
    async getPreferences() {
      const row = await db.getFirstAsync('SELECT * FROM user_preferences WHERE owner_key = ?', [uid ?? '']);
      return row
        ? { ...DEFAULT_PREFERENCES, ...JSON.parse(row.prefs_json), rev: row.rev, updatedAt: row.updated_at }
        : { ...DEFAULT_PREFERENCES, rev: 0, updatedAt: null };
    },

    // Replaces the whole document. expectedRev is the rev from getPreferences (0 = none stored yet).
    async savePreferences(expectedRev, prefs) {
      const { errors, value } = normalizePreferences(prefs);
      if (errors.length) throw new TypeError(`Invalid preferences: ${errors.join('; ')}`);
      let saved;
      await tx(async (txn) => {
        const key = uid ?? '';
        const cur = await txn.getFirstAsync('SELECT rev FROM user_preferences WHERE owner_key = ?', [key]);
        if ((cur?.rev ?? 0) !== expectedRev) throw new RevisionConflictError(`preferences are at rev ${cur?.rev ?? 0}`);
        const now = Date.now();
        await txn.runAsync(
          `INSERT INTO user_preferences (owner_key, prefs_json, rev, updated_at) VALUES (?, ?, ?, ?)
           ON CONFLICT (owner_key) DO UPDATE SET prefs_json = excluded.prefs_json, rev = excluded.rev, updated_at = excluded.updated_at`,
          [key, JSON.stringify(value), expectedRev + 1, now]
        );
        saved = { ...value, rev: expectedRev + 1, updatedAt: now };
      });
      emit();
      return saved;
    },

    // ---- Assistant conversation (current identity only; on this device, never synced). Messages do not
    // emit change events: the diary has nothing to reload.

    // The latest conversation, oldest message first. A row that fails to parse is skipped, not fatal.
    async getConversation() {
      const key = uid ?? '';
      const rows = await db.getAllAsync(
        `SELECT * FROM assistant_messages WHERE owner_key = ? AND conversation_id =
           (SELECT conversation_id FROM assistant_messages WHERE owner_key = ? ORDER BY seq DESC LIMIT 1)
         ORDER BY seq`,
        [key, key]
      );
      return rows.flatMap((r) => {
        try {
          return [{ ...JSON.parse(r.content_json), seq: r.seq, conversationId: r.conversation_id, role: r.role, createdAt: r.created_at }];
        } catch {
          return [];
        }
      });
    },

    // Keeps the newest MAX_ASSISTANT_MESSAGES per identity.
    async appendAssistantMessage(conversationId, role, content) {
      if (typeof conversationId !== 'string' || !conversationId || conversationId.length > 64) throw new TypeError('conversationId required');
      if (role !== 'user' && role !== 'assistant') throw new TypeError('role must be user or assistant');
      const json = JSON.stringify(content);
      if (!content || typeof content !== 'object' || json.length > 200000) throw new TypeError('content must be an object under 200 KB');
      const key = uid ?? '';
      let seq;
      await tx(async (txn) => {
        seq = (await txn.runAsync(
          'INSERT INTO assistant_messages (owner_key, conversation_id, role, content_json, created_at) VALUES (?, ?, ?, ?, ?)',
          [key, conversationId, role, json, Date.now()]
        )).lastInsertRowId;
        await txn.runAsync(
          'DELETE FROM assistant_messages WHERE owner_key = ? AND seq <= (SELECT seq FROM assistant_messages WHERE owner_key = ? ORDER BY seq DESC LIMIT 1 OFFSET ?)',
          [key, key, MAX_ASSISTANT_MESSAGES]
        );
      });
      return seq;
    },

    async clearConversations() {
      const key = uid ?? '';
      await tx((txn) => txn.runAsync('DELETE FROM assistant_messages WHERE owner_key = ?', [key]));
    },

    // ---- Conflicts (current identity only).

    async listConflicts() {
      const [clause, params] = owner();
      const rows = await db.getAllAsync(
        `SELECT c.* FROM meal_conflicts c JOIN meals m ON m.id = c.meal_id
         WHERE c.resolved_at IS NULL AND m.${clause} ORDER BY c.detected_at`,
        params
      );
      return rows.map((c) => ({
        id: c.id, mealId: c.meal_id, remoteRev: c.remote_rev, detectedAt: c.detected_at,
        remote: c.remote_json == null ? null : JSON.parse(c.remote_json),
      }));
    },

    // Keep cloud: the stored remote version replaces the local meal and the pending edit is dropped.
    // Keep local: the pending edit is rebased onto the remote rev the user compared against; the next
    // push re-checks that rev inside a Firestore transaction and reopens the conflict if it moved again.
    async resolveConflict(conflictId, keep) {
      if (keep !== 'cloud' && keep !== 'local') throw new TypeError('keep must be cloud or local');
      await tx(async (txn) => {
        const c = await txn.getFirstAsync('SELECT * FROM meal_conflicts WHERE id = ? AND resolved_at IS NULL', [conflictId]);
        const cur = c && await loadOwned(txn, c.meal_id);
        if (!cur || !uid) throw new MealNotFoundError(conflictId);
        if (keep === 'cloud') {
          if (c.remote_json == null) throw new TypeError('no cloud version to keep');
          await writeRemote(txn, JSON.parse(c.remote_json), uid, cur);
          await clearOutbox(txn, c.meal_id);
        } else {
          await txn.runAsync(
            'UPDATE outbox SET base_remote_rev = ?, attempts = 0, last_error = NULL, next_attempt_at = NULL WHERE meal_id = ?',
            [c.remote_rev, c.meal_id]
          );
          await txn.runAsync('DELETE FROM sent_edits WHERE meal_id = ?', [c.meal_id]);
        }
        await txn.runAsync('UPDATE meal_conflicts SET resolved_at = ? WHERE id = ?', [Date.now(), conflictId]);
      });
      emit();
    },

    onChange(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
};
