import * as SQLite from 'expo-sqlite';
import { migrate } from './schema';
import { createMealRepository } from './mealRepository';

let opening = null;

// One shared repository per JS runtime (survives Fast Refresh of App.js). A failed open is not cached,
// so a later call can retry.
export const openMealRepository = () => {
  opening ??= (async () => {
    const db = await SQLite.openDatabaseAsync('platego.db');
    // WAL lets withExclusiveTransactionAsync's separate connection coexist with normal reads.
    await db.execAsync('PRAGMA journal_mode = WAL;');
    await migrate(db);
    return createMealRepository(db);
  })().catch((e) => {
    opening = null;
    throw e;
  });
  return opening;
};
