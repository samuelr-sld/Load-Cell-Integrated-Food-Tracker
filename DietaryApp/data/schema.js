// Versioned SQLite schema. Each migration runs in one exclusive transaction together with
// its PRAGMA user_version bump, so a crash leaves the previous version intact and a rerun is a no-op.
// Takes any connection exposing the expo-sqlite async API (execAsync, getFirstAsync, withExclusiveTransactionAsync).

export const MIGRATIONS = [
  // v1
  `
  CREATE TABLE IF NOT EXISTS meals (
    id TEXT PRIMARY KEY NOT NULL,
    owner_status TEXT NOT NULL CHECK (owner_status IN ('pending', 'owned')),
    owner_uid TEXT,
    logged_at INTEGER,
    tz_offset_min INTEGER,
    day_key TEXT,
    category TEXT NOT NULL,
    items_json TEXT NOT NULL,
    calories REAL, protein REAL, carbs REAL, fat REAL,
    source_kind TEXT NOT NULL,
    source_ref TEXT,
    original_scan_json TEXT,
    rev INTEGER NOT NULL CHECK (rev >= 1),
    remote_rev INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    deleted_at INTEGER,
    CHECK ((owner_status = 'pending' AND owner_uid IS NULL) OR (owner_status = 'owned' AND owner_uid IS NOT NULL)),
    CHECK ((logged_at IS NULL) = (day_key IS NULL))
  );
  CREATE INDEX IF NOT EXISTS meals_owner_day ON meals (owner_uid, day_key);
  CREATE INDEX IF NOT EXISTS meals_pending ON meals (owner_status) WHERE owner_status = 'pending';
  CREATE UNIQUE INDEX IF NOT EXISTS meals_scan_ref ON meals (source_ref)
    WHERE source_kind = 'camera' AND source_ref IS NOT NULL;

  -- One coalesced pending change per meal; removed only when a future sync acknowledges it.
  CREATE TABLE IF NOT EXISTS outbox (
    meal_id TEXT PRIMARY KEY NOT NULL,
    op TEXT NOT NULL CHECK (op IN ('upsert', 'delete')),
    edit_id TEXT NOT NULL,
    meal_rev INTEGER NOT NULL,
    base_remote_rev INTEGER,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    next_attempt_at INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
  );
  CREATE INDEX IF NOT EXISTS outbox_next_attempt ON outbox (next_attempt_at);

  CREATE TABLE IF NOT EXISTS meal_conflicts (
    id TEXT PRIMARY KEY NOT NULL,
    meal_id TEXT NOT NULL,
    local_rev INTEGER,
    remote_rev INTEGER,
    local_json TEXT,
    remote_json TEXT,
    detected_at INTEGER NOT NULL,
    resolved_at INTEGER
  );
  CREATE INDEX IF NOT EXISTS meal_conflicts_meal ON meal_conflicts (meal_id);

  CREATE TABLE IF NOT EXISTS sync_state (
    key TEXT PRIMARY KEY NOT NULL,
    value TEXT
  );

  -- Guards enforced in the database so no code path (or future sync bug) can bypass them.
  CREATE TRIGGER IF NOT EXISTS meals_owner_immutable
  BEFORE UPDATE OF owner_uid, owner_status ON meals
  WHEN OLD.owner_status = 'owned' AND (NEW.owner_status != 'owned' OR NEW.owner_uid IS NOT OLD.owner_uid)
  BEGIN SELECT RAISE(ABORT, 'meal owner cannot change once assigned'); END;

  CREATE TRIGGER IF NOT EXISTS meals_origin_immutable
  BEFORE UPDATE OF original_scan_json, source_kind, source_ref, created_at ON meals
  WHEN NEW.original_scan_json IS NOT OLD.original_scan_json OR NEW.source_kind IS NOT OLD.source_kind
    OR NEW.source_ref IS NOT OLD.source_ref OR NEW.created_at IS NOT OLD.created_at
  BEGIN SELECT RAISE(ABORT, 'original scan values are immutable'); END;

  CREATE TRIGGER IF NOT EXISTS meals_no_unsynced_delete
  BEFORE DELETE ON meals
  WHEN EXISTS (SELECT 1 FROM outbox WHERE meal_id = OLD.id)
  BEGIN SELECT RAISE(ABORT, 'meal has unsynchronized changes'); END;
  `,
  // v2: cloud sync. Edits already sent to Firestore but not yet acknowledged (lets a retry recognise
  // its own landed write after a lost ack), one open conflict per meal, and on-device demo samples
  // re-tagged as source_kind 'sample' so they can never enter the outbox or reach the cloud.
  `
  CREATE TABLE IF NOT EXISTS sent_edits (
    meal_id TEXT NOT NULL,
    edit_id TEXT NOT NULL,
    PRIMARY KEY (meal_id, edit_id)
  );
  CREATE UNIQUE INDEX IF NOT EXISTS meal_conflicts_open ON meal_conflicts (meal_id) WHERE resolved_at IS NULL;

  DROP TRIGGER IF EXISTS meals_origin_immutable;
  UPDATE meals SET source_kind = 'sample'
    WHERE source_kind = 'manual' AND items_json LIKE '[{"name":"SAMPLE %' AND items_json NOT LIKE '%},{%';
  DELETE FROM outbox WHERE meal_id IN (SELECT id FROM meals WHERE source_kind = 'sample');
  CREATE TRIGGER meals_origin_immutable
  BEFORE UPDATE OF original_scan_json, source_kind, source_ref, created_at ON meals
  WHEN NEW.original_scan_json IS NOT OLD.original_scan_json OR NEW.source_kind IS NOT OLD.source_kind
    OR NEW.source_ref IS NOT OLD.source_ref OR NEW.created_at IS NOT OLD.created_at
  BEGIN SELECT RAISE(ABORT, 'original scan values are immutable'); END;
  `,
  // v3: goals and dietary preferences, one validated JSON document per identity. Local only (not
  // synced yet); rev/updated_at give a future sync the same optimistic-concurrency base as meals.
  `
  CREATE TABLE IF NOT EXISTS user_preferences (
    owner_key TEXT PRIMARY KEY NOT NULL, -- Firebase uid, or '' for this device before its first sign-in
    prefs_json TEXT NOT NULL,
    rev INTEGER NOT NULL CHECK (rev >= 1),
    updated_at INTEGER NOT NULL
  );
  `,
  // v4: assistant conversation, per identity, on this device only (never synced). seq orders messages.
  `
  CREATE TABLE IF NOT EXISTS assistant_messages (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_key TEXT NOT NULL, -- Firebase uid, or '' before the first sign-in
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content_json TEXT NOT NULL,
    created_at INTEGER NOT NULL
  );
  CREATE INDEX IF NOT EXISTS assistant_messages_owner ON assistant_messages (owner_key, seq);
  `,
];

export const migrate = async (db) => {
  const current = (await db.getFirstAsync('PRAGMA user_version')).user_version;
  for (let v = current + 1; v <= MIGRATIONS.length; v++) {
    await db.withExclusiveTransactionAsync(async (txn) => {
      await txn.execAsync(MIGRATIONS[v - 1]);
      await txn.execAsync(`PRAGMA user_version = ${v}`);
    });
  }
  return MIGRATIONS.length;
};
