// Live benchmark of the laptop model through the real gateway, orchestrator and nutrition tools.
//   node ai/benchmark.mjs http://127.0.0.1:8100 <pairing token> [--cold]
// Seeds an in-memory diary (node:sqlite), asks the milestone questions and prints latency, tool choice,
// JSON validity and grounding per question. --cold unloads the model first to measure loading.
import { openNodeDb } from '../data/nodeSqliteAdapter.mjs';
import { migrate } from '../data/schema.js';
import { createMealRepository } from '../data/mealRepository.js';
import { createNutritionTools } from '../domain/nutritionTools.js';
import { localDayKey, addDays } from '../domain/meal.js';
import { ask } from './assistant.js';
import { createGatewayProvider, createMockProvider } from './providers.js';

const [baseUrl, token] = process.argv.slice(2);
const TODAY = localDayKey(Date.now());
const food = (name, grams, calories, protein, carbs, fat) => ({ name, grams, per100g: { calories, protein, carbs, fat } });

const db = openNodeDb();
await migrate(db);
const repo = createMealRepository(db);
await repo.setIdentity('bench');
const mk = (daysAgo, category, items, source = { kind: 'manual' }) =>
  repo.createMeal({ loggedAt: Date.parse(`${addDays(TODAY, -daysAgo)}T12:00:00Z`), tzOffsetMin: 0, category, items, source });
for (const d of [0, 1, 2, 3, 5, 8, 9, 11]) {
  await mk(d, 'breakfast', [food('oatmeal', 250, 70, 2.5, 12, 1.5), food('banana', 120, 89, 1.1, 23, 0.3)]);
  await mk(d, 'lunch', [food(d % 2 ? 'chicken rice' : 'tofu stir-fry', 350, 160, 9, 20, 4)]);
}
await mk(1, 'dinner', [food('adobo', 250, 200, 15, 3, 14)], { kind: 'camera', ref: 'scan-1' });
await mk(2, 'dinner', [food('mystery stew', 300, 120, null, null, null)]);
await repo.savePreferences(0, { goals: { calories: 2000, protein: 90, carbs: null, fat: null }, diets: ['pescatarian'], exclusions: ['peanut'], averageBasis: 'logged_days' });
const realTools = createNutritionTools({ repo, today: () => TODAY });
const tools = { ...realTools, call: async (name, args) => {
  const r = await realTools.call(name, args);
  console.log(r.ok ? `  call ${name}(${JSON.stringify(args)})` : `  rejected ${name}(${JSON.stringify(args)}): ${r.error.message}`);
  return r;
} };

const gateway = createGatewayProvider({ baseUrl, token, chatTimeoutMs: 180000 });
const timings = [];
const provider = { ...gateway, chat: async (a) => { const r = await gateway.chat(a); timings.push(r.timings); return r; } };
const mock = createMockProvider({ today: () => TODAY });

const QUESTIONS = [
  ['How much protein did I consume today?', ['get_nutrition_summary']],
  ['Summarize my nutrition this week.', ['get_nutrition_summary']],
  ['Compare my calorie intake this week and last week.', ['compare_periods']],
  ['Which day had the most carbohydrates?', ['render_nutrition_chart', 'get_meal_history']],
  ['What foods do I eat most frequently?', ['get_food_patterns']],
  ['What meals could I prepare based on my preferences?', ['get_user_preferences', 'get_food_patterns']],
  ['What nutritional goals have I configured?', ['get_user_preferences']],
  ['Show my macronutrient split for this week as a pie chart.', ['render_nutrition_chart']],
  ['How many calories did I eat in the last 7 days?', ['get_nutrition_summary']],
  // The app's starter prompts.
  ['Analyze my week', ['get_nutrition_summary', 'get_food_patterns']],
  ['Show my protein intake', ['render_nutrition_chart', 'get_nutrition_summary']],
  ['Compare this week and last week', ['compare_periods']],
  ['Suggest a meal', ['get_user_preferences', 'get_food_patterns']],
  ['Do I eat a lot of fried food?', ['get_food_patterns']],
];

console.log(`health: ${JSON.stringify(await gateway.health())}`);
if (process.argv.includes('--cold')) {
  const h = await gateway.health();
  await fetch('http://127.0.0.1:11434/api/generate', { method: 'POST', body: JSON.stringify({ model: h.model, keep_alive: 0 }) });
  console.log('model unloaded for a cold start');
}

const rows = [];
const runOne = async (question, expected, history = []) => {
  timings.length = 0;
  const r = await ask({ provider, fallbackProvider: mock, tools, question, history, today: TODAY, requestId: 'bench' });
  const first = r.trace.tools[0]?.split('!')[0] ?? '-';
  rows.push({
    question: question.slice(0, 44), ms: r.trace.ms, loadMs: timings.reduce((s, t) => s + (t?.loadMs ?? 0), 0), rounds: r.trace.rounds,
    tools: r.trace.tools.join(' > ') || '-', charts: r.charts.length, firstOk: expected.includes(first), invalidJson: r.trace.invalidReplies,
    toolErrors: r.trace.toolErrors, nudges: r.trace.nudges,
    // Unnecessary: repeats of a result already held, plus successful calls to tools the question does not need.
    unneeded: r.trace.duplicates + r.trace.tools.filter((t) => !t.includes('!') && !expected.includes(t)).length,
    outcome: r.source === 'model' ? 'grounded' : `fallback:${r.errorCode}`,
  });
  console.log(`\nQ: ${question}\nA [${r.source}${r.notice ? ` – ${r.notice}` : ''}]: ${r.text}`);
  if (r.trace.rejectedAnswer) console.log(`  rejected model answer: ${r.trace.rejectedAnswer}`);
  return r;
};

for (const [q, expected] of QUESTIONS) await runOne(q, expected);
// Multi-turn: a follow-up that only makes sense with the previous turn.
const first = await runOne('Summarize my nutrition this week.', ['get_nutrition_summary']);
await runOne('And what about last week?', ['get_nutrition_summary', 'compare_periods'], [
  { role: 'user', content: 'Summarize my nutrition this week.' }, { role: 'assistant', content: first.text }]);
// Repeated identical requests (model warm).
for (let i = 0; i < 3; i++) await runOne('How much protein did I consume today?', ['get_nutrition_summary']);

console.table(rows);
const n = rows.length;
const sum = (k) => rows.reduce((s, r) => s + r[k], 0);
console.log(`first tool correct: ${rows.filter((r) => r.firstOk).length}/${n} · grounded model answers: ${rows.filter((r) => r.outcome === 'grounded').length}/${n} · valid JSON replies: ${sum('rounds') - sum('invalidJson')}/${sum('rounds')} · unnecessary calls: ${sum('unneeded')} · rejected tool args: ${sum('toolErrors')} · median ms: ${rows.map((r) => r.ms).sort((a, b) => a - b)[Math.floor(n / 2)]}`);
