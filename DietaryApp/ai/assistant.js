// Nutrition assistant orchestration. The model only proposes; the app decides:
//   question -> model replies {"tool","arguments"} -> nutritionTools.call validates and runs it on the local
//   SQLite engine -> a bounded digest goes back to the model -> model replies {"answer"} -> every number in
//   the answer is checked against the tool results -> shown with deterministic evidence notes and charts.
// Nothing here executes model output: the model can only name one of the six read-only tools.
import { NUTRIENTS, addDays } from '../domain/meal.js';
import { periodRange, previousRange, daysBetween, UNSUPPORTED_PATTERNS } from '../domain/analytics.js';
import { TOOL_DEFINITIONS } from '../domain/nutritionTools.js';

export const MAX_ROUNDS = 4;      // model calls per question; then a deterministic answer
export const HISTORY_MESSAGES = 6; // earlier chat messages sent as context
const NUDGE = '[PlateGo] ';        // marks app-written corrections so they are never mistaken for the question
const LABEL = { calories: 'calories', protein: 'protein', carbs: 'carbohydrates', fat: 'fat' };
const UNIT = { calories: 'kcal', protein: 'g', carbs: 'g', fat: 'g' };

// ---- Dates: the model copies these instead of doing date arithmetic.

const weekday = (k) => new Date(`${k}T00:00:00Z`).toLocaleDateString('en-US', { timeZone: 'UTC', weekday: 'long' });

export const rangeLabel = (from, to) => {
  if (from === to) return `${weekday(from)} ${from}`;
  const w = periodRange('week', from);
  if (w.from === from && w.to === to) return `calendar week Mon ${from} to Sun ${to}`;
  const m = periodRange('month', from);
  if (m.from === from && m.to === to) return `calendar month ${from.slice(0, 7)}`;
  return `${from} to ${to} (${daysBetween(from, to) + 1} days)`;
};

// Unsupported pattern types stay valid in the contract (they return an explanation) but are not advertised:
// small models copy whole enums.
const sig = (p) => Object.entries(p.properties).map(([k, s]) => `${k}${p.required.includes(k) ? '' : '?'}: ${
  s.enum ? s.enum.join('|') : s.type === 'object' ? `{${sig(s)}}` : s.type === 'array' ? `[${s.items.enum.filter((x) => !UNSUPPORTED_PATTERNS[x]).join('|')}]` : s.pattern ? 'YYYY-MM-DD' : s.type}`).join(', ');

// The exact period arguments for common phrases; also returned with argument errors so the model can copy one.
export const periodExamples = (today) => {
  const week = periodRange('week', today);
  const last = previousRange('week', week);
  return {
    today: { period: 'day' },
    yesterday: { period: 'day', date: addDays(today, -1) },
    [`this calendar week (${week.from} to ${week.to})`]: { period: 'week' },
    [`last calendar week (${last.from} to ${last.to})`]: { period: 'week', date: last.from },
    [`last 7 days, rolling (${addDays(today, -6)} to ${today})`]: { period: 'custom', from: addDays(today, -6), to: today },
    'this month': { period: 'month' },
  };
};

export const systemPrompt = (today) => `You are PlateGo's nutrition assistant. You explain the user's own logged meals.
Today is ${weekday(today)} ${today}. Calendar weeks run Monday to Sunday.
"period" must be exactly day, week, month or custom. Copy these arguments:
${Object.entries(periodExamples(today)).map(([k, v]) => `- ${k}: ${JSON.stringify(v)}`).join('\n')}
Leave out optional arguments you do not need (never send null).

Reply with ONE JSON object and nothing else:
{"tool": "<name>", "arguments": {...}} to read the user's data, or
{"answer": "<plain text for the user>"} when you have what you need.

Tools (read-only):
${TOOL_DEFINITIONS.map((d) => `- ${d.name}(${sig(d.parameters)}): ${d.description}`).join('\n')}

Which tool:
- totals, averages, goal progress for a period: get_nutrition_summary
- which day was highest/lowest, daily values, trends, any chart: render_nutrition_chart, e.g. {"type":"bar","metric":"carbs","period":"week"}; macro split: {"type":"pie","period":"week"} (no metric)
- this period vs another: compare_periods
- foods eaten most often, habits: get_food_patterns with {"period":"month"} unless the user names a period
- goals, diets, excluded foods: get_user_preferences
- meal ideas: get_user_preferences, then get_food_patterns
- which meals were logged: get_meal_history

Rules:
- Read data with a tool before answering any question about the user's meals or goals.
- Every number in an answer must be copied from a tool result. Never estimate, add up, or invent meals, foods or values.
- null means unknown, not zero. If meals have unknown values, say the total is at least that amount.
- Goals are daily targets. Do not compute differences, sums or percentages yourself; use the values given.
- Mention sample demo meals or scan estimates when the result reports them.
- Say "calendar week" or "last 7 days" to match the period you used.
- Preparation method (fried, grilled …), ingredients, food groups and micronutrients are not recorded. Say so; never guess them.
- Suggestions are general healthy-eating ideas that respect the user's diets and excluded foods, not medical advice. No amounts in suggestions.
- To show a chart, call render_nutrition_chart; the app draws it. Do not describe chart code.
- If a tool returns an error, fix the arguments or say what you could not read. Do not use other results instead.
- Answers: at most 120 words, plain text, no tables.`;

// ---- Bounded tool results for the model (ids and evidence lists become counts).

const nonZero = (o) => Object.fromEntries(Object.entries(o).filter(([, v]) => v));

export const digest = (name, r) => {
  if (!r.ok) return { error: r.error };
  const d = r.data;
  const extra = r.truncated?.length ? { truncated: r.truncated } : {};
  switch (name) {
    case 'get_nutrition_summary': return {
      period: rangeLabel(d.from, d.to), from: d.from, to: d.to, daysSoFar: d.elapsedDays, loggedDays: d.loggedDays, mealCount: d.mealCount,
      totals: Object.fromEntries(NUTRIENTS.map((n) => [n, d.nutrients[n].total])),
      mealsWithUnknown: nonZero(Object.fromEntries(NUTRIENTS.map((n) => [n, d.nutrients[n].missingMeals]))),
      ...(d.from !== d.to && { averagePerDay: d.averages.perDay, averageBasis: d.averages.basis }),
      macroEnergyPct: d.macros.pct, goals: d.goals,
      ...nonZero({
        daysWithNothingLogged: d.completeness.unloggedDays.length, undatedMealsExcluded: d.completeness.undatedMealsExcluded,
        sampleMeals: d.evidence.sampleMealIds.length, scanEstimatedMeals: d.evidence.estimatedMealIds.length,
        userCorrectedScans: d.evidence.userEditedScanMealIds.length,
      }),
      ...extra,
    };
    case 'compare_periods': return {
      current: { period: rangeLabel(d.current.from, d.current.to), mealCount: d.current.mealCount },
      previous: { period: rangeLabel(d.previous.from, d.previous.to), mealCount: d.previous.mealCount },
      perDayAverages: d.nutrients, averageBasis: d.basis, ...extra,
    };
    case 'get_food_patterns': return {
      period: rangeLabel(d.from, d.to),
      patterns: d.patterns.map(({ evidence, ...p }) => ({ ...p, ...(evidence.matchedNames && { matchedNames: evidence.matchedNames }) })),
      ...extra,
    };
    case 'get_meal_history': return {
      period: rangeLabel(d.from, d.to), totalMeals: d.totalMeals, shown: Math.min(d.meals.length, 15),
      meals: d.meals.slice(0, 15).map(({ id, ...m }) => m), ...extra,
    };
    case 'render_nutrition_chart': {
      if (d.type === 'pie') return { chart: 'pie', period: rangeLabel(d.from, d.to), slices: d.slices, basedOnMeals: d.basedOnMeals, excludedMeals: d.excludedMeals };
      const known = d.points.filter((p) => p.value != null);
      const pick = (better) => known.reduce((a, p) => (!a || better(p.value, a.value) ? p : a), null);
      const point = (p) => p && { date: p.key, value: p.value, ...(p.lowerBound && { atLeast: true }) };
      return {
        chart: d.type, metric: d.metric, unit: d.unit, valueKind: d.valueKind, period: rangeLabel(d.from, d.to),
        points: d.points.map(point), highest: point(pick((a, b) => a > b)), lowest: point(pick((a, b) => a < b)),
        bucketsWithNothingLogged: d.points.length - known.length, goal: d.goal,
      };
    }
    default: return { ...d, ...extra };
  }
};

// ---- Grounding: every number the model states must appear in what the app gave it.

// ISO dates are checked as whole tokens, so their parts (10, 28, 2026 …) never make a stray number pass.
// Written dates ("Oct 12, 2026", "October 5–11", "12 October 2026") are checked as dates: with a year the exact
// date must be known; without one, some known date must fall on that month and day. List numbering is ignored.
const DATE = /\d{4}-\d{2}-\d{2}/g;
const MONTH = '(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\\.?';
const DAY = '(\\d{1,2})(?:st|nd|rd|th)?';
const WRITTEN_DATE = new RegExp(`\\b(?:${DAY} ${MONTH}|${MONTH} ${DAY}(?:\\s*(?:-|–|to)\\s*${DAY})?)(?:,? (\\d{4}))?\\b`, 'gi');
const MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'];
const pad2 = (n) => String(n).padStart(2, '0');
const NUM = /\d[\d,]*(?:\.\d+)?/g;
const numbersIn = (text) => (text.replace(DATE, ' ').match(NUM) ?? []).map((t) => Number(t.replace(/,/g, ''))).filter(Number.isFinite);
const text = (x) => (typeof x === 'string' ? x : JSON.stringify(x));

// `dateSources` may supply dates only (e.g. the system prompt's calendar); `sources` supply numbers and dates.
export const ungroundedNumbers = (answer, sources, dateSources = []) => {
  const all = sources.map(text).join(' ');
  const dates = new Set([all, ...dateSources].join(' ').match(DATE) ?? []);
  const allowed = numbersIn(all);
  const badWritten = [];
  const a = String(answer)
    .replace(/(\d)[‐-―−](?=\d)/g, '$1-') // granite writes 2026‑10‑12 with non-breaking hyphens
    .replace(/^\s*\d+[.)]\s/gm, '').replace(WRITTEN_DATE, (m, d1, m1, m2, d2, d3, year) => {
    const mmdd = `-${pad2(MONTHS.indexOf((m1 ?? m2).slice(0, 3).toLowerCase()) + 1)}-`;
    const known = (d) => (year ? dates.has(`${year}${mmdd}${pad2(d)}`) : [...dates].some((x) => x.endsWith(`${mmdd}${pad2(d)}`)));
    if (![d1 ?? d2, d3].filter(Boolean).every(known)) badWritten.push(m);
    return ' ';
  });
  return [
    ...badWritten,
    ...(a.match(DATE) ?? []).filter((d) => !dates.has(d)),
    ...numbersIn(a).filter((n) => !allowed.some((v) => Math.abs(n - v) <= Math.max(0.51, v * 0.005))),
  ];
};

// ---- Deterministic planner: common questions -> tool calls. Used by the mock provider and the fallback.

export const planFallback = (question, today) => {
  const q = ` ${String(question).toLowerCase()} `;
  const has = (...w) => w.some((x) => q.includes(x));
  const metric = has('protein') ? 'protein' : has('carb') ? 'carbs' : has(' fat') ? 'fat' : has('calor', 'kcal', 'energy') ? 'calories' : null;
  const week = (date) => ({ period: 'week', ...(date && { date }) });
  const spec = has('yesterday') ? { period: 'day', date: addDays(today, -1) }
    : has('today', 'so far') ? { period: 'day' }
    : has('last 7', 'past 7', 'seven days', 'rolling') ? { period: 'custom', from: addDays(today, -6), to: today }
    : has('last week', 'previous week') && !has('compare', ' vs', 'versus') ? week(addDays(today, -7))
    : has('month') ? { period: 'month' }
    : week();
  const multi = spec.period === 'day' ? week() : spec; // charts and patterns need more than one day

  if (has('suggest', 'prepare', 'cook', 'recipe', 'meal idea', 'what should i eat')) {
    return [{ name: 'get_user_preferences', args: {} }, { name: 'get_food_patterns', args: { period: 'month', types: ['frequent_foods', 'exclusion_matches'] } }];
  }
  if (has('fried', 'grilled', 'steamed', 'baked', 'boiled', 'roasted', 'preparation', 'processed')) {
    return [{ name: 'get_food_patterns', args: { period: 'month', types: ['preparation_method'] } }]; // answered with "not recorded"
  }
  if (has('goal', 'preference', 'diet', 'configured', 'target', 'exclu')) return [{ name: 'get_user_preferences', args: {} }];
  if (has('compare', ' vs', 'versus', 'than last', 'change')) {
    return [{ name: 'compare_periods', args: { current: { period: spec.period === 'custom' ? 'week' : spec.period } } }];
  }
  if (has('frequent', 'most often', 'usually', 'pattern', 'common', 'favorite', 'favourite')) {
    return [{ name: 'get_food_patterns', args: { ...(spec.period === 'week' && !spec.date ? { period: 'month' } : multi), types: ['frequent_foods', 'exclusion_matches'] } }];
  }
  if (has('pie', 'split', 'distribution', 'breakdown', 'macro')) return [{ name: 'render_nutrition_chart', args: { ...multi, type: 'pie' } }];
  if (has('which day', 'most', 'highest', 'least', 'lowest', 'chart', 'graph', 'plot', 'trend', 'show')) {
    return [
      { name: 'render_nutrition_chart', args: { ...multi, type: 'bar', metric: metric ?? 'calories' } },
      ...(has('show') ? [{ name: 'get_nutrition_summary', args: multi }] : []),
    ];
  }
  if (has('history', 'what did i eat', 'what have i eaten', 'list', 'my meals')) {
    const r = spec.period === 'custom' ? spec : periodRange(spec.period, spec.date ?? today);
    return [{ name: 'get_meal_history', args: { from: r.from, to: r.to > today ? today : r.to } }];
  }
  if (has('analy', 'review')) return [{ name: 'get_nutrition_summary', args: spec }, { name: 'get_food_patterns', args: { ...multi, types: ['frequent_foods', 'goal_adherence', 'exclusion_matches'] } }];
  if (metric || has('summar', 'how much', 'how many', 'intake', 'nutrition', 'total', 'average', 'eat', 'ate')) return [{ name: 'get_nutrition_summary', args: spec }];
  return null;
};

// ---- Deterministic answers, built only from digests.

const HELP = 'I can answer questions about your logged meals, for example: "How much protein did I eat today?", "Compare this week and last week", "Which day had the most carbs?", "What foods do I eat most often?" or "Suggest a meal".';
const IDEAS = [ // [idea, diets it does not fit]
  ['a vegetable and chickpea curry with brown rice', []],
  ['a tofu and vegetable stir-fry with rice', []],
  ['a bean and vegetable burrito bowl', []],
  ['fish with steamed vegetables and quinoa', ['vegetarian', 'vegan']],
  ['a lentil soup with whole-grain bread', ['gluten_free']],
  ['a chicken and salad wrap', ['vegetarian', 'vegan', 'pescatarian', 'gluten_free']],
  ['an egg and vegetable omelette', ['vegan']],
  ['Greek yogurt with fruit', ['vegan', 'dairy_free']],
];
const words = (s) => ` ${String(s).toLowerCase().replace(/[^a-z0-9]+/g, ' ')} `;
const SUGGEST = /suggest|prepare|cook|recipe|idea|should i eat/;

// Foods a suggestion must not name for a stated diet. shortcut: a word list, not ingredient knowledge;
// it only guards suggestions and errs toward PlateGo's own filtered ideas.
const MEAT = ['chicken', 'beef', 'pork', 'lamb', 'turkey', 'bacon', 'ham', 'sausage', 'meat'];
const SEAFOOD = ['fish', 'salmon', 'tuna', 'shrimp', 'prawn', 'shellfish'];
const DAIRY = ['milk', 'cheese', 'yogurt', 'yoghurt', 'butter', 'cream'];
const DIET_CONFLICTS = {
  vegetarian: [...MEAT, ...SEAFOOD], vegan: [...MEAT, ...SEAFOOD, ...DAIRY, 'egg', 'eggs', 'honey'], pescatarian: MEAT,
  halal: ['pork', 'bacon', 'ham', 'wine'], kosher: ['pork', 'bacon', 'ham', 'shrimp', 'prawn', 'shellfish'],
  dairy_free: DAIRY, gluten_free: ['bread', 'pasta', 'wheat', 'couscous', 'noodles'],
};
export const preferenceConflicts = (answer, prefs) => {
  const w = words(answer);
  const terms = [...(prefs?.exclusions ?? []), ...(prefs?.diets ?? []).flatMap((d) => DIET_CONFLICTS[d] ?? [])];
  return [...new Set(terms.filter((t) => words(t).trim() && w.includes(words(t))))];
};
const value = (v, n, unknownMeals) => (v == null ? 'unknown' : `${unknownMeals ? 'at least ' : ''}${v} ${UNIT[n]}`);
const list = (xs) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`);
const cap = (s) => s[0].toUpperCase() + s.slice(1);

const describeOne = (name, d, metric) => {
  if (d.error?.code === 'unsupported_claim') return `PlateGo can't tell from your log: ${d.error.message}`;
  if (d.error) return `I couldn't read that data (${d.error.message}).`;
  switch (name) {
    case 'get_nutrition_summary': {
      if (!d.mealCount) return `No meals are logged for ${d.period}.`;
      const shown = metric ? [metric] : NUTRIENTS;
      const totals = shown.map((n) => `${LABEL[n]} ${value(d.totals[n], n, d.mealsWithUnknown[n])}`);
      const avg = d.averagePerDay && shown.map((n) => `${LABEL[n]} ${value(d.averagePerDay[n], n, d.mealsWithUnknown[n])}`);
      const goals = Object.entries(d.goals ?? {}).filter(([n, g]) => shown.includes(n) && g.pctOfGoal != null)
        .map(([n, g]) => `${LABEL[n]} ${g.lowerBound ? 'at least ' : ''}${g.pctOfGoal}% of your ${g.goal} ${UNIT[n]} goal`);
      return [
        `${cap(d.period)}: ${d.mealCount} meal(s) on ${d.loggedDays} day(s). Total ${list(totals)}.`,
        avg && `Average per ${d.averageBasis === 'calendar_days' ? 'calendar' : 'logged'} day: ${list(avg)}.`,
        goals.length && `${cap(list(goals))}.`,
      ].filter(Boolean).join(' ');
    }
    case 'compare_periods': {
      const lines = NUTRIENTS.filter((n) => !metric || n === metric).map((n) => {
        const x = d.perDayAverages[n];
        return x.comparable
          ? `${LABEL[n]} ${x.current} vs ${x.previous} ${UNIT[n]} (${x.change > 0 ? '+' : ''}${x.change}${x.changePct == null ? '' : `, ${x.changePct > 0 ? '+' : ''}${x.changePct}%`})`
          : `${LABEL[n]} not comparable (${x.current == null || x.previous == null ? 'no data in one period' : 'some values unknown'})`;
      });
      return `Average per ${d.averageBasis === 'calendar_days' ? 'calendar' : 'logged'} day, ${d.current.period} vs ${d.previous.period}: ${lines.join('; ')}.`;
    }
    case 'get_food_patterns': {
      const freq = d.patterns.filter((p) => p.type === 'frequent_foods').slice(0, 5);
      const excl = d.patterns.filter((p) => p.type === 'exclusion_matches');
      const goal = d.patterns.filter((p) => p.type === 'goal_adherence');
      return [
        freq.length ? `Most often logged (${d.period}): ${list(freq.map((p) => `${p.food} (${p.mealCount} meals)`))}.` : `No food was logged more than once in ${d.period}.`,
        ...goal.map((p) => `${cap(LABEL[p.nutrient])} was above your goal on ${p.daysAbove} day(s) and at or below it on ${p.daysAtOrBelow}.`),
        ...excl.map((p) => `"${p.term}" appears in ${p.mealCount} logged meal name(s).`),
      ].join(' ');
    }
    case 'get_meal_history': {
      if (!d.totalMeals) return `No meals are logged for ${d.period}.`;
      const recent = d.meals.slice(0, 5).map((m) => `${m.dayKey} ${m.category}: ${m.foods.map((f) => f.name).join(', ')} (${value(m.totals.calories, 'calories')})`);
      return `${d.totalMeals} meal(s) logged for ${d.period}. Most recent: ${recent.join('; ')}.`;
    }
    case 'render_nutrition_chart': {
      if (d.chart === 'pie') return d.slices ? `Energy split for ${d.period}: ${list(d.slices.map((s) => `${LABEL[s.key]} ${s.pct}%`))} (from ${d.basedOnMeals} meal(s) with known macros).` : `Not enough meals with known carbs, protein and fat in ${d.period} for a split.`;
      if (!d.highest) return `Nothing is logged for ${d.period}.`;
      const each = d.valueKind === 'total' ? 'day' : 'period';
      const pt = (p) => `${p.atLeast ? 'at least ' : ''}${p.value} ${d.unit} on ${p.date}`;
      return `${cap(LABEL[d.metric])} per ${each}, ${d.period}: highest ${pt(d.highest)}, lowest ${pt(d.lowest)}.${d.bucketsWithNothingLogged ? ` ${d.bucketsWithNothingLogged} ${each}(s) with nothing logged.` : ''}`;
    }
    case 'get_user_preferences': {
      const goals = NUTRIENTS.filter((n) => d.goals[n] != null).map((n) => `${LABEL[n]} ${d.goals[n]} ${UNIT[n]}`);
      return [
        goals.length ? `Your daily goals: ${list(goals)}.` : 'You have not set any daily goals.',
        d.diets.length ? `Diets: ${list(d.diets.map((x) => x.replace('_', '-')))}.` : '',
        d.exclusions.length ? `Excluded foods: ${list(d.exclusions)}.` : '',
      ].filter(Boolean).join(' ');
    }
    default: return '';
  }
};

const suggestions = (prefs) => {
  const ideas = IDEAS.filter(([idea, bad]) => !bad.some((x) => prefs?.diets?.includes(x)) && !prefs?.exclusions?.some((t) => words(idea).includes(words(t))))
    .slice(0, 4).map(([idea]) => idea);
  return `Some ideas that fit your stated preferences: ${list(ideas)}. These are general suggestions, not medical advice.`;
};

export const describe = (question, calls) => {
  if (!calls.length) return HELP;
  const q = String(question).toLowerCase();
  const metric = NUTRIENTS.find((n) => q.includes(n === 'carbs' ? 'carb' : n === 'calories' ? 'calor' : n));
  const text = calls.map((c) => describeOne(c.name, c.digest, metric)).filter(Boolean);
  const prefs = calls.find((c) => c.name === 'get_user_preferences' && !c.digest.error)?.digest;
  if (SUGGEST.test(q)) text.push(suggestions(prefs));
  return text.join('\n\n');
};

// Deterministic notes shown under an answer: period semantics, unknowns, sample and scan provenance.
export const evidenceNotes = (calls) => {
  const notes = new Set();
  for (const { name, digest: d } of calls) {
    if (d.error) continue;
    if (d.period) notes.add(`${cap(d.period)} · from your on-device diary`);
    if (d.current) notes.add(`${cap(d.current.period)} vs ${d.previous.period}`);
    for (const [n, k] of Object.entries(d.mealsWithUnknown ?? {})) notes.add(`${cap(LABEL[n])} unknown in ${k} meal(s): totals are lower bounds`);
    if (d.sampleMeals) notes.add(`Includes ${d.sampleMeals} SAMPLE demo meal(s)`);
    if (d.scanEstimatedMeals) notes.add(`${d.scanEstimatedMeals} meal(s) are camera scan estimates`);
    if (d.userCorrectedScans) notes.add(`${d.userCorrectedScans} scanned meal(s) corrected by you`);
    if (d.daysWithNothingLogged) notes.add(`${d.daysWithNothingLogged} day(s) with nothing logged (not counted as zero)`);
    if (d.undatedMealsExcluded) notes.add(`${d.undatedMealsExcluded} undated meal(s) not included`);
    if (name === 'compare_periods' && NUTRIENTS.some((n) => !d.perDayAverages[n].comparable)) notes.add('Some nutrients are not comparable: no data or unknown values');
    if (d.points?.some((p) => p.atLeast)) notes.add('Faded bars: some values unknown');
  }
  return [...notes];
};

const FOLLOW_UPS = {
  get_nutrition_summary: ['Show my calories as a chart', 'Compare this week and last week'],
  compare_periods: ['Which day had the most calories?', 'What foods do I eat most often?'],
  render_nutrition_chart: ['Summarize my nutrition this week', 'Show my macro split'],
  get_food_patterns: ['Suggest a meal', 'Summarize my nutrition this week'],
  get_user_preferences: ['Suggest a meal', 'How much protein did I eat today?'],
  get_meal_history: ['Summarize my nutrition this week'],
};

// ---- Mock provider brain: plan by keyword, then answer from the tool results in the conversation.

const isQuestion = (m) => m.role === 'user' && !m.content.startsWith(NUDGE);

// A follow-up like "and last week?" has no intent of its own: plan it together with the previous question.
const FOLLOW_UP = /^\s*(and|what about|how about|same|also)\b/i;
const planInContext = (question, previous, today) => planFallback(question, today)
  ?? (previous && FOLLOW_UP.test(question) ? planFallback(`${previous} ${question}`, today) : null);

export const mockReply = (messages, today) => {
  const i = messages.findLastIndex(isQuestion);
  const question = messages[i]?.content ?? '';
  const results = messages.slice(i + 1).filter((m) => m.role === 'tool').map((m) => JSON.parse(m.content));
  const plan = planInContext(question, messages.slice(0, i).findLast(isQuestion)?.content, today) ?? [];
  if (results.length < plan.length) return { tool: plan[results.length].name, arguments: plan[results.length].args };
  return { answer: describe(question, plan.map((p, k) => ({ name: p.name, digest: results[k] }))) };
};

// Small models send null for optional arguments; null means "not given", and the tool validator checks the rest.
const dropNulls = (v) => (v && typeof v === 'object' && !Array.isArray(v)
  ? Object.fromEntries(Object.entries(v).filter(([, x]) => x !== null).map(([k, x]) => [k, dropNulls(x)]))
  : v);

// A pie chart is the macro split, so metric "macros" (or "") on a pie says nothing extra. Only that exact case is
// rewritten; "macros" on a bar chart is ambiguous and left for the validator to reject.
const PIE_METRIC_ALIASES = new Set(['', 'macro', 'macros', 'macronutrient', 'macronutrients', 'macro_split', 'macros_split']);
const normalizeArgs = (tool, args) => {
  if (tool !== 'render_nutrition_chart' || args?.type !== 'pie' || !PIE_METRIC_ALIASES.has(String(args.metric).toLowerCase())) return args;
  const { metric, ...rest } = args;
  return rest;
};

// Accepts {"tool","arguments"}, {"name","arguments"} (some models' native style) or {"answer"}.
export const parseReply = (content) => {
  const s = String(content ?? '');
  let v;
  try { v = JSON.parse(s); } catch {
    try { v = JSON.parse(s.slice(s.indexOf('{'), s.lastIndexOf('}') + 1)); } catch { return null; }
  }
  if (!v || typeof v !== 'object') return null;
  const tool = v.tool ?? v.name;
  if (typeof tool === 'string') {
    let args = v.arguments ?? v.args ?? {};
    if (typeof args === 'string') try { args = JSON.parse(args); } catch { /* left invalid: the tool validator rejects it */ }
    return { tool, args: normalizeArgs(tool, dropNulls(args)) };
  }
  return typeof v.answer === 'string' && v.answer.trim() ? { answer: v.answer.trim() } : null;
};

// ---- The loop.

// provider: see providers.js. tools: createNutritionTools(...). history: [{ role, content }] of earlier turns.
// fallbackProvider: deterministic provider used when `provider` fails or misbehaves. Rethrows only cancellation.
// Returns { text, facts?, charts, notes, source, model, notice, errorCode, followUps, trace }.
export const ask = async ({ provider, fallbackProvider, tools, question, history = [], today, signal, requestId = 'q' }) => {
  const sys = systemPrompt(today);
  const messages = [{ role: 'system', content: sys }, ...history.slice(-HISTORY_MESSAGES), { role: 'user', content: question }];
  const calls = [];
  const trace = { rounds: 0, invalidReplies: 0, toolErrors: 0, nudges: 0, duplicates: 0, ms: 0 };
  const started = Date.now();
  let model = null;

  const traceNow = () => ({ ...trace, tools: calls.map((c) => `${c.name}${c.result.ok ? '' : `!${c.result.error.code}`}`), ms: Date.now() - started });
  const finish = (answer, source, extra = {}) => ({
    text: answer, source, model,
    charts: calls.filter((c) => c.name === 'render_nutrition_chart' && c.result.ok).map((c) => c.result.data),
    notes: evidenceNotes(calls),
    followUps: FOLLOW_UPS[calls.find((c) => c.result.ok)?.name] ?? ['Analyze my week', 'Suggest a meal'],
    trace: traceNow(),
    ...extra,
  });

  const fallback = async (notice, errorCode) => {
    if (!fallbackProvider || fallbackProvider === provider) return finish(describe(question, calls), 'fallback', { notice, errorCode });
    const r = await ask({ provider: fallbackProvider, tools, question, history, today, signal, requestId });
    return { ...r, source: 'fallback', notice, errorCode, trace: { ...traceNow(), fallback: r.trace } };
  };

  const plan = planInContext(question, history.findLast((m) => m.role === 'user')?.content, today);
  for (let round = 0; round < MAX_ROUNDS; round++) {
    // The last round must end in an answer: a tool requested now would be run but never explained.
    if (round === MAX_ROUNDS - 1 && calls.some((c) => c.result.ok)) {
      messages.push({ role: 'user', content: `${NUDGE}Last step: no more tools. Answer now with {"answer": ...} using the results above.` });
    }
    let res;
    try {
      res = await provider.chat({ messages, json: true, signal, requestId: `${requestId}-${round}` });
    } catch (e) {
      if (e?.code === 'cancelled' || signal?.aborted) throw e;
      return fallback(e?.message ?? 'The AI could not answer.', e?.code ?? 'unknown');
    }
    if (signal?.aborted) throw Object.assign(new Error('Request cancelled.'), { code: 'cancelled' });
    trace.rounds++;
    model = res.model ?? model;
    const reply = parseReply(res.content);

    const nudge = (text, counted = true) => {
      if (counted) trace.nudges++;
      messages.push({ role: 'assistant', content: String(res.content).slice(0, 2000) }, { role: 'user', content: `${NUDGE}${text}` });
    };

    if (reply?.tool) {
      const result = await tools.call(reply.tool, reply.args);
      if (!result.ok) trace.toolErrors++;
      const d = digest(reply.tool, result);
      // Duplicates are judged by result, not arguments: {"period":"week"} and the same week with an explicit date are one call.
      if (result.ok && calls.some((c) => c.result.ok && c.name === reply.tool && JSON.stringify(c.digest) === JSON.stringify(d))) {
        trace.duplicates++;
        nudge('You already have that result above. Answer now with {"answer": ...}.');
        continue;
      }
      calls.push({ name: reply.tool, args: reply.args, result, digest: d });
      const help = result.error?.code === 'invalid_arguments' ? { validPeriodArguments: periodExamples(today) } : {};
      messages.push({ role: 'assistant', content: JSON.stringify({ tool: reply.tool, arguments: reply.args }) }, { role: 'tool', content: JSON.stringify({ ...d, ...help }) });
      continue;
    }
    if (reply?.answer) {
      // A diary question answered without reading the diary (never accepted), or with none of the tools the
      // deterministic planner expects (hinted once; the model still makes the call and the validator checks it).
      const read = (c) => c.result.ok || c.result.error.code === 'unsupported_claim'; // "not recorded" is an answer too
      const anyData = calls.some(read);
      if (plan && !plan.some((p) => calls.some((c) => read(c) && c.name === p.name)) && (!anyData || trace.nudges < 1)) {
        nudge(`Read the data first, for example {"tool": "${plan[0].name}", "arguments": ${JSON.stringify(plan[0].args)}}.`);
        continue;
      }
      const prefs = calls.find((c) => c.name === 'get_user_preferences' && c.result.ok)?.digest;
      const clash = provider.id !== 'offline' && SUGGEST.test(question.toLowerCase()) ? preferenceConflicts(reply.answer, prefs) : [];
      if (clash.length) {
        trace.rejectedAnswer = reply.answer;
        return fallback(`The AI suggested foods that conflict with your preferences (${clash.slice(0, 3).join(', ')}), so these ideas come from PlateGo instead.`, 'preference_conflict');
      }
      const bad = ungroundedNumbers(reply.answer, [question, ...history.map((m) => m.content), ...calls.map((c) => c.digest)], [sys]);
      // `facts`: the same results in deterministic words, shown beside the model's wording.
      if (!bad.length) return provider.id === 'offline' ? finish(reply.answer, 'fallback') : finish(reply.answer, 'model', { facts: describe(question, calls) });
      trace.rejectedAnswer = reply.answer;
      return fallback(`The AI answer contained numbers that are not in your diary (${bad.slice(0, 3).join(', ')}), so this answer was built from your data instead.`, 'ungrounded');
    }
    trace.invalidReplies++;
    nudge('Reply with exactly one JSON object: {"tool": ..., "arguments": ...} or {"answer": ...}.', false);
  }
  return fallback('The AI did not finish within the allowed steps, so this answer was built from your data.', 'max_rounds');
};
