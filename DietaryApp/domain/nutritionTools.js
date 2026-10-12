// Read-only nutrition tools for a future local LLM (also usable without one). Each tool takes
// schema-validated arguments, reads through the identity-scoped repository (so a caller can only
// ever see the signed-in user's meals: no tool accepts a user id), computes with the deterministic
// engine, and returns bounded structured data. No tool writes, deletes, or runs caller-supplied SQL.
import { NUTRIENTS, CATEGORIES, localDayKey } from './meal.js';
import {
  summarize, comparePeriods, detectPatterns, chartData, periodRange, previousRange, isDayKey, daysBetween,
  PATTERN_TYPES, UNSUPPORTED_PATTERNS, CHART_TYPES, GROUP_BY,
} from './analytics.js';

const MAX_SPAN_DAYS = 366;
const MAX_DAY_POINTS = 62;
const MAX_LIST = 50; // longest array any tool returns; longer ones are cut and reported in `truncated`

const DATE = { type: 'string', pattern: '^\\d{4}-\\d{2}-\\d{2}$', description: 'Local calendar date YYYY-MM-DD' };
const PERIOD = {
  period: { type: 'string', enum: ['day', 'week', 'month', 'custom'], description: 'Weeks run Monday to Sunday' },
  date: { ...DATE, description: 'Any day inside the day/week/month (default: today). Not used with custom.' },
  from: { ...DATE, description: 'custom only: first day' },
  to: { ...DATE, description: 'custom only: last day' },
};
const obj = (properties, required = []) => ({ type: 'object', properties, required, additionalProperties: false });
const PERIOD_SPEC = obj(PERIOD, ['period']);

export const TOOL_DEFINITIONS = [
  {
    name: 'get_nutrition_summary',
    description: 'Totals, per-day averages, macro distribution, meal-type breakdown, goal progress, data completeness and the ids of the meals used, for one period.',
    parameters: obj(PERIOD, ['period']),
  },
  {
    name: 'get_food_patterns',
    description: 'Patterns supported by stored records only (food names, meal categories, daily totals vs goals, exclusion name matches), each with evidence.',
    parameters: obj({
      ...PERIOD,
      types: { type: 'array', minItems: 1, maxItems: 8, items: { type: 'string', enum: [...PATTERN_TYPES, ...Object.keys(UNSUPPORTED_PATTERNS)] } },
    }, ['period']),
  },
  {
    name: 'compare_periods',
    description: 'Compares per-day averages of two periods. previous defaults to the period immediately before current.',
    parameters: obj({ current: PERIOD_SPEC, previous: PERIOD_SPEC }, ['current']),
  },
  {
    name: 'get_meal_history',
    description: 'Meals logged between two dates, newest first: category, foods, totals, missing nutrients and source.',
    parameters: obj({
      from: DATE, to: DATE,
      category: { type: 'string', enum: CATEGORIES },
      limit: { type: 'integer', minimum: 1, maximum: MAX_LIST },
    }, ['from', 'to']),
  },
  {
    name: 'render_nutrition_chart',
    description: 'Chart dataset computed by the nutrition engine. bar/line need metric; pie is the macro energy split and takes no metric or groupBy.',
    parameters: obj({
      ...PERIOD,
      type: { type: 'string', enum: CHART_TYPES },
      metric: { type: 'string', enum: NUTRIENTS },
      groupBy: { type: 'string', enum: GROUP_BY },
    }, ['period', 'type']),
  },
  {
    name: 'get_user_preferences',
    description: "The user's optional daily goals, stated dietary preferences, excluded foods and averaging basis.",
    parameters: obj({}),
  },
];

class ToolError extends Error {
  constructor(code, message) { super(message); this.code = code; }
}
const invalid = (message) => new ToolError('invalid_arguments', message);

// Validates the JSON-Schema subset used above. Objects never accept unlisted keys.
const check = (s, v, path, errs) => {
  if (s.type === 'object') {
    if (!v || typeof v !== 'object' || Array.isArray(v)) return errs.push(`${path} must be an object`);
    for (const k of Object.keys(v)) if (!(k in s.properties)) errs.push(`${path}.${k} is not allowed`);
    for (const k of s.required) if (v[k] === undefined) errs.push(`${path}.${k} is required`);
    for (const [k, sub] of Object.entries(s.properties)) if (v[k] !== undefined) check(sub, v[k], `${path}.${k}`, errs);
  } else if (s.type === 'array') {
    if (!Array.isArray(v)) return errs.push(`${path} must be an array`);
    if (v.length < s.minItems || v.length > s.maxItems) errs.push(`${path} must have ${s.minItems}-${s.maxItems} items`);
    v.forEach((x, i) => check(s.items, x, `${path}[${i}]`, errs));
  } else if (s.type === 'string') {
    if (typeof v !== 'string') return errs.push(`${path} must be a string`);
    if (s.pattern && !new RegExp(s.pattern).test(v)) errs.push(`${path} has an invalid format`);
    if (s.enum && !s.enum.includes(v)) errs.push(`${path} must be one of: ${s.enum.join(', ')}`);
  } else if (s.type === 'integer') {
    if (!Number.isInteger(v) || v < s.minimum || v > s.maximum) errs.push(`${path} must be an integer from ${s.minimum} to ${s.maximum}`);
  }
};

const checkRange = (from, to, today) => {
  for (const k of [from, to]) if (!isDayKey(k)) throw invalid(`${k} is not a calendar date`);
  if (from > to) throw invalid('from must not be after to');
  if (from > today) throw new ToolError('unsupported_range', 'the period starts in the future');
  if (daysBetween(from, to) + 1 > MAX_SPAN_DAYS) throw new ToolError('unsupported_range', `periods are limited to ${MAX_SPAN_DAYS} days`);
  return { from, to };
};

const resolvePeriod = (a, today) => {
  if (a.period === 'custom') {
    if (a.date !== undefined || !a.from || !a.to) throw invalid('custom periods need from and to, and no date');
    return checkRange(a.from, a.to, today);
  }
  if (a.from !== undefined || a.to !== undefined) throw invalid(`${a.period} periods take an optional date, not from/to`);
  const date = a.date ?? today;
  if (!isDayKey(date)) throw invalid(`${date} is not a calendar date`);
  const r = periodRange(a.period, date);
  return checkRange(r.from, r.to, today);
};

// Cuts arrays longer than MAX_LIST and lists what was cut, so results stay bounded but never silently partial.
const bound = (v, path = '', cut = []) => {
  if (Array.isArray(v)) {
    if (v.length > MAX_LIST) cut.push({ path, total: v.length, returned: MAX_LIST });
    return v.slice(0, MAX_LIST).map((x, i) => bound(x, `${path}[${i}]`, cut));
  }
  if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, bound(x, path ? `${path}.${k}` : k, cut)]));
  return v;
};

// repo: the identity-scoped meal repository. today(): local day key (injectable for tests).
export const createNutritionTools = ({ repo, today = () => localDayKey(Date.now()) }) => {
  // Dated meals in [from, to] plus undated ones (only counted: they belong to no period).
  const load = async (from, to) => [...await repo.listMeals({ fromDayKey: from, toDayKey: to }), ...await repo.listUndated()];
  const prefs = async () => {
    const { goals, diets, exclusions, averageBasis } = await repo.getPreferences();
    return { goals, diets, exclusions, averageBasis };
  };

  const handlers = {
    async get_nutrition_summary(a, t) {
      const r = resolvePeriod(a, t);
      const p = await prefs();
      return summarize(await load(r.from, r.to), { ...r, today: t, averageBasis: p.averageBasis, goals: p.goals });
    },

    async get_food_patterns(a, t) {
      const types = a.types ?? PATTERN_TYPES;
      const unsupported = types.find((x) => UNSUPPORTED_PATTERNS[x]);
      if (unsupported) throw new ToolError('unsupported_claim', UNSUPPORTED_PATTERNS[unsupported]);
      const { types: _, ...spec } = a;
      const r = resolvePeriod(spec, t);
      return { ...r, patterns: detectPatterns(await load(r.from, r.to), { ...r, types, preferences: await prefs() }) };
    },

    async compare_periods(a, t) {
      const cur = resolvePeriod(a.current, t);
      const prev = a.previous ? resolvePeriod(a.previous, t) : previousRange(a.current.period, cur);
      const p = await prefs();
      const meals = await load(cur.from < prev.from ? cur.from : prev.from, cur.to > prev.to ? cur.to : prev.to);
      const opts = { today: t, averageBasis: p.averageBasis };
      return comparePeriods(summarize(meals, { ...cur, ...opts }), summarize(meals, { ...prev, ...opts }));
    },

    async get_meal_history(a, t) {
      const r = checkRange(a.from, a.to, t);
      const all = (await repo.listMeals({ fromDayKey: r.from, toDayKey: r.to })).filter((m) => !a.category || m.category === a.category);
      const limit = a.limit ?? 20;
      return {
        ...r, totalMeals: all.length, returned: Math.min(limit, all.length),
        // No owner id, scan reference, original scan payload or exact timestamp: the day and category suffice.
        meals: all.slice(0, limit).map((m) => ({
          id: m.id, dayKey: m.dayKey, category: m.category,
          foods: m.items.map((i) => ({ name: i.name, grams: i.grams })),
          totals: m.totals, missing: NUTRIENTS.filter((n) => m.totals[n] == null),
          source: m.source.kind, edited: m.revision.rev > 1 || (m.remoteRev ?? 1) > 1,
        })),
      };
    },

    async render_nutrition_chart(a, t) {
      const { type, metric, groupBy, ...spec } = a;
      if (type === 'pie' && (metric || groupBy)) throw invalid('pie charts show the macro split; omit metric and groupBy');
      if (type !== 'pie' && !metric) throw invalid(`${type} charts need a metric`);
      const r = resolvePeriod(spec, t);
      if (type !== 'pie' && (groupBy ?? 'day') === 'day' && daysBetween(r.from, r.to) + 1 > MAX_DAY_POINTS) {
        throw new ToolError('unsupported_range', `daily charts are limited to ${MAX_DAY_POINTS} days; group by week or month`);
      }
      const p = await prefs();
      return chartData(await load(r.from, r.to), { type, metric, groupBy, ...r, today: t, averageBasis: p.averageBasis, goals: p.goals });
    },

    get_user_preferences: () => prefs(),
  };

  // Never throws: returns { ok: true, data, truncated } or { ok: false, error: { code, message } }.
  const call = async (name, args = {}) => {
    const def = TOOL_DEFINITIONS.find((d) => d.name === name);
    if (!def) return { ok: false, error: { code: 'unknown_tool', message: `unknown tool ${String(name).slice(0, 60)}` } };
    const errs = [];
    check(def.parameters, args, 'args', errs);
    if (errs.length) return { ok: false, error: { code: 'invalid_arguments', message: errs.join('; ') } };
    try {
      const truncated = [];
      const data = bound(await handlers[name](args, today()), '', truncated);
      return { ok: true, data, truncated };
    } catch (e) {
      if (e instanceof ToolError) return { ok: false, error: { code: e.code, message: e.message } };
      console.error(`nutrition tool ${name} failed:`, e?.message);
      return { ok: false, error: { code: 'internal_error', message: 'The nutrition data could not be read.' } };
    }
  };

  return { definitions: TOOL_DEFINITIONS, call };
};
