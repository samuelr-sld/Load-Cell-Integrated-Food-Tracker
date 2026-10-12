// Test-only: exposes the subset of the expo-sqlite async API the repository uses, backed by
// Node's built-in real SQLite (node:sqlite). The SQL, constraints and triggers run for real;
// the expo-sqlite native binding itself is not exercised here.
import { DatabaseSync } from 'node:sqlite';

export const openNodeDb = (path = ':memory:') => {
  const raw = new DatabaseSync(path);
  let queue = Promise.resolve();
  const conn = {
    raw,
    execAsync: async (sql) => raw.exec(sql),
    runAsync: async (sql, params = []) => {
      const r = raw.prepare(sql).run(...params);
      return { changes: Number(r.changes), lastInsertRowId: Number(r.lastInsertRowid) };
    },
    getFirstAsync: async (sql, params = []) => raw.prepare(sql).get(...params) ?? null,
    getAllAsync: async (sql, params = []) => raw.prepare(sql).all(...params),
    // Serialized like expo's exclusive transaction; rolls back on any thrown error.
    withExclusiveTransactionAsync: (task) => {
      const run = queue.then(async () => {
        raw.exec('BEGIN EXCLUSIVE');
        try {
          await task(conn);
          raw.exec('COMMIT');
        } catch (e) {
          raw.exec('ROLLBACK');
          throw e;
        }
      });
      queue = run.catch(() => {});
      return run;
    },
    closeAsync: async () => raw.close(),
  };
  return conn;
};
