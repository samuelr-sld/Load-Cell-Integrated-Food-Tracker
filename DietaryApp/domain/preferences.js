// User goals and dietary preferences (pure). Goals are the user's own daily targets; nothing here
// diagnoses, prescribes, or judges a diet. Diet labels are stored as stated, never checked against meals.
import { NUTRIENTS } from './meal.js';
import { AVERAGE_BASES } from './analytics.js';

export const DIETS = ['vegetarian', 'vegan', 'pescatarian', 'halal', 'kosher', 'dairy_free', 'gluten_free'];
const GOAL_MAX = { calories: 10000, protein: 1000, carbs: 2000, fat: 1000 }; // sanity bounds per day
const MAX_EXCLUSIONS = 30;

export const DEFAULT_PREFERENCES = {
  goals: { calories: null, protein: null, carbs: null, fat: null },
  diets: [],
  exclusions: [],
  averageBasis: 'logged_days',
};

// Returns { errors, value }; value is the normalized document to store (exclusions trimmed,
// lowercased and de-duplicated). Unknown keys are rejected so the stored shape stays stable.
export const normalizePreferences = (p) => {
  const errors = [];
  if (!p || typeof p !== 'object') return { errors: ['preferences must be an object'], value: null };
  const extra = Object.keys(p).filter((k) => !(k in DEFAULT_PREFERENCES));
  if (extra.length) errors.push(`unknown preference: ${extra.join(', ')}`);

  const goals = {};
  for (const n of NUTRIENTS) {
    const v = p.goals?.[n] ?? null;
    if (v !== null && (typeof v !== 'number' || !Number.isFinite(v) || v <= 0 || v > GOAL_MAX[n])) {
      errors.push(`${n} goal must be blank or a number above 0 and at most ${GOAL_MAX[n]}`);
    }
    goals[n] = v;
  }
  if (p.goals && Object.keys(p.goals).some((k) => !NUTRIENTS.includes(k))) errors.push('goals may only set calories, protein, carbs, fat');

  const diets = p.diets ?? [];
  if (!Array.isArray(diets) || diets.some((d) => !DIETS.includes(d))) errors.push(`diets must be chosen from ${DIETS.join(', ')}`);

  const raw = p.exclusions ?? [];
  const exclusions = Array.isArray(raw)
    ? [...new Set(raw.map((x) => (typeof x === 'string' ? x.trim().toLowerCase().replace(/\s+/g, ' ') : x)))].filter((x) => x !== '')
    : null;
  if (!exclusions || exclusions.some((x) => typeof x !== 'string' || x.length > 40)) errors.push('each excluded food must be text of at most 40 characters');
  else if (exclusions.length > MAX_EXCLUSIONS) errors.push(`at most ${MAX_EXCLUSIONS} excluded foods`);

  const averageBasis = p.averageBasis ?? DEFAULT_PREFERENCES.averageBasis;
  if (!AVERAGE_BASES.includes(averageBasis)) errors.push(`averageBasis must be ${AVERAGE_BASES.join(' or ')}`);

  return { errors, value: errors.length ? null : { goals, diets: [...new Set(diets)], exclusions, averageBasis } };
};
