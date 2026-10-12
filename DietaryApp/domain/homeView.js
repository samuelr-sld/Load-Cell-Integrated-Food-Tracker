import { provenance } from './analytics.js';

// Pure view-models for the Home screen. Every number comes from analytics.summarize(); nothing is recomputed here,
// and unknown stays null (rendered as a dash), never 0.

const row = (sum, n) => {
  const { total, missingMeals } = sum.nutrients[n];
  const goal = sum.goals?.[n]?.goal ?? null;
  return {
    key: n,
    total,
    goal,
    fraction: goal && total != null ? Math.min(total / goal, 1) : 0,
    over: goal != null && total != null && total > goal,
    lowerBound: total != null && missingMeals > 0, // some meals lack this nutrient, so the real figure may be higher
  };
};

// `sum` = summarize(meals, { from: today, to: today, goals }).
export const todayOverview = (sum) => ({
  mealCount: sum.mealCount,
  calories: row(sum, 'calories'),
  macros: ['protein', 'carbs', 'fat'].map((n) => row(sum, n)),
});

// One short status for the header pill. Raw sync errors stay in the console; the user only sees what to expect.
export const syncPill = ({ uid, conflicts, unsynced, status }) => {
  if (!uid) return { tone: 'neutral', label: 'Saved on this phone' };
  if (conflicts) return { tone: 'warn', label: `${conflicts} ${conflicts === 1 ? 'meal needs' : 'meals need'} your choice` };
  if (status?.state === 'error') return { tone: 'warn', label: 'Sync paused · will retry' };
  if (status?.state === 'syncing') return { tone: 'info', label: 'Syncing…' };
  if (unsynced) return { tone: 'info', label: `${unsynced} waiting to sync` };
  if (status?.state === 'idle') return { tone: 'ok', label: 'Up to date' };
  return null;
};

// 'unpaired' = no laptop address/token saved; 'offline' = last poll failed; 'online' = last poll succeeded.
export const scaleIndicator = (link) => ({
  unpaired: { tone: 'neutral', title: 'Scale not paired', body: 'Open the assistant settings to pair this phone with your laptop.' },
  offline: { tone: 'warn', title: 'Can’t reach your scale', body: 'Check that your laptop is on and on the same Wi-Fi. You can still log meals by hand.' },
  online: { tone: 'ok', title: 'Scale connected', body: 'Place your plate on the scale. The meal appears here to review.' },
}[link] ?? { tone: 'neutral', title: 'Looking for your scale…', body: 'Meals you log by hand are saved right away.' });

// Camera values are scan estimates until the user corrects them; typed-in values are the user's own.
export const mealTag = (m) => {
  const p = provenance(m);
  if (p.source === 'sample') return { label: 'Sample', tone: 'warn' };
  if (p.source === 'camera') return p.edited ? { label: 'Corrected', tone: 'ok' } : { label: 'Estimated', tone: 'info' };
  return null;
};
