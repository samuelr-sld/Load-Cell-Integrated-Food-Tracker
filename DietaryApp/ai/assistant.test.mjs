// Assistant orchestration, providers and conversation storage against the real repository (node:sqlite)
// and the real nutrition tools. Models are scripted, so every behaviour here is deterministic.
import test from 'node:test';
import assert from 'node:assert/strict';
import { openNodeDb } from '../data/nodeSqliteAdapter.mjs';
import { migrate } from '../data/schema.js';
import { createMealRepository } from '../data/mealRepository.js';
import { createNutritionTools } from '../domain/nutritionTools.js';
import { ask, planFallback, ungroundedNumbers, parseReply, MAX_ROUNDS, systemPrompt } from './assistant.js';
import { createGatewayProvider, createMockProvider, normalizeBaseUrl, AiError } from './providers.js';

const TODAY = '2026-10-11'; // a Sunday
const food = (name, grams, calories, protein, carbs, fat) => ({ name, grams, per100g: { calories, protein, carbs, fat } });
const at = (day) => Date.parse(`${day}T12:00:00Z`);

const setup = async () => {
  const db = openNodeDb();
  await migrate(db);
  const repo = createMealRepository(db);
  await repo.setIdentity('uidB');
  await repo.createMeal({ loggedAt: at('2026-10-10'), tzOffsetMin: 0, items: [food('secret B food', 100, 999, 1, 1, 1)], source: { kind: 'manual' } });
  await repo.setIdentity('uidA');
  const mk = (day, category, items, source = { kind: 'manual' }) => repo.createMeal({ loggedAt: at(day), tzOffsetMin: 0, category, items, source });
  await mk('2026-10-06', 'lunch', [food('Rice', 200, 130, 2.7, 28, 0.3)]);
  await mk('2026-10-08', 'dinner', [food('rice', 150, 130, 2.7, 28, 0.3), food('Fried chicken', 100, 250, 20, 10, 15)]);
  await mk('2026-10-09', 'breakfast', [food('mystery bowl', 300, 120, null, null, null)]);
  await mk('2026-10-10', 'dinner', [food('adobo', 250, 200, 15, 3, 14)], { kind: 'camera', ref: 'scan-1' });
  await mk('2026-10-10', 'snack', [food('SAMPLE oatmeal', 250, 70, 2.5, 12, 1.5)], { kind: 'sample' });
  await mk('2026-09-30', 'lunch', [food('rice', 100, 130, 2.7, 28, 0.3)]);
  await repo.savePreferences(0, { goals: { calories: null, protein: 100, carbs: null, fat: null }, diets: ['vegetarian'], exclusions: ['chicken'], averageBasis: 'logged_days' });
  const tools = createNutritionTools({ repo, today: () => TODAY });
  return { db, repo, tools, mock: createMockProvider({ today: () => TODAY }) };
};

// A fake model that replies with the given contents in order and records what it was sent.
const scripted = (...replies) => {
  const sent = [];
  return {
    id: 'laptop', sent,
    chat: async ({ messages }) => {
      sent.push(messages.map((m) => ({ ...m })));
      const r = replies[Math.min(sent.length - 1, replies.length - 1)];
      if (r instanceof Error) throw r;
      return { content: typeof r === 'string' ? r : JSON.stringify(r), model: 'fake' };
    },
  };
};

const run = (ctx, question, provider = ctx.mock, extra = {}) =>
  ask({ provider, fallbackProvider: ctx.mock, tools: ctx.tools, question, today: TODAY, ...extra });

// ---- Built-in (mock) answers for the milestone's example questions.

test('built-in provider answers the example questions from the engine, with evidence notes', async () => {
  const ctx = await setup();
  const yesterday = await run(ctx, 'How much protein did I consume yesterday?');
  assert.deepEqual(yesterday.trace.tools, ['get_nutrition_summary']);
  assert.match(yesterday.text, /protein 43\.8 g/); // adobo 37.5 + sample 6.25, from the engine
  assert.equal(yesterday.source, 'fallback');
  assert.ok(yesterday.notes.includes('Includes 1 SAMPLE demo meal(s)'));
  assert.ok(yesterday.notes.includes('1 meal(s) are camera scan estimates'));

  const today = await run(ctx, 'How much protein did I consume today?');
  assert.match(today.text, /No meals are logged for Sunday 2026-10-11/);

  const week = await run(ctx, 'Summarize my nutrition this week.');
  assert.match(week.text, /calendar week Mon 2026-10-05 to Sun 2026-10-11: 5 meal\(s\) on 4 day\(s\)/i);
  assert.match(week.text, /calories 1740 kcal/);
  assert.match(week.text, /protein at least 73\.3 g/); // the mystery bowl's protein is unknown, not zero
  assert.ok(week.notes.some((n) => /Protein unknown in 1 meal/.test(n)));

  const cmp = await run(ctx, 'Compare my calorie intake this week and last week.');
  assert.deepEqual(cmp.trace.tools, ['compare_periods']);
  assert.match(cmp.text, /calories 435 vs 130 kcal \(\+305, \+235%\)/);

  const carbs = await run(ctx, 'Which day had the most carbohydrates?');
  assert.match(carbs.text, /highest 56 g on 2026-10-06/);
  assert.equal(carbs.charts.length, 1);
  assert.equal(carbs.charts[0].points.find((p) => p.key === '2026-10-09').value, null); // unknown day stays unknown

  const freq = await run(ctx, 'What foods do I eat most frequently?');
  assert.match(freq.text, /rice \(2 meals\)/);
  assert.match(freq.text, /"chicken" appears in 1 logged meal name/);

  const goals = await run(ctx, 'What nutritional goals have I configured?');
  assert.match(goals.text, /protein 100 g/);
  assert.match(goals.text, /Diets: vegetarian/);

  const ideas = await run(ctx, 'What meals could I prepare based on my preferences?');
  assert.doesNotMatch(ideas.text, /fish|chicken and salad/); // vegetarian, chicken excluded
  assert.match(ideas.text, /not medical advice/);

  const fried = await run(ctx, 'Do I eat a lot of fried food?');
  assert.match(fried.text, /can't tell from your log: Preparation method .* is not recorded/);
  assert.doesNotMatch(systemPrompt(TODAY), /preparation_method/); // not advertised to the model, still valid in the contract

  const followUp = await run(ctx, 'And what about last week?', ctx.mock, { history: [{ role: 'user', content: 'Summarize my nutrition this week.' }, { role: 'assistant', content: week.text }] });
  assert.match(followUp.text, /calendar week Mon 2026-09-28 to Sun 2026-10-04: 1 meal/i);

  for (const r of [yesterday, week, cmp, carbs, freq, goals, ideas, fried, followUp]) assert.equal(r.notice, undefined, r.text);
});

test('calendar weeks and rolling 7-day periods are planned and labelled differently', () => {
  const thursday = '2026-10-08';
  assert.deepEqual(planFallback('summarize this week', thursday)[0].args, { period: 'week' });
  assert.deepEqual(planFallback('summarize the last 7 days', thursday)[0].args, { period: 'custom', from: '2026-10-02', to: thursday });
  assert.deepEqual(planFallback('summarize last week', thursday)[0].args, { period: 'week', date: '2026-10-01' });
  const p = systemPrompt(thursday);
  assert.match(p, /this calendar week \(2026-10-05 to 2026-10-11\)/);
  assert.match(p, /last 7 days, rolling \(2026-10-02 to 2026-10-08\)/);
  assert.equal(planFallback('tell me a joke', thursday), null);
});

// ---- Orchestration with a (scripted) model.

const SUMMARY_WEEK = { tool: 'get_nutrition_summary', arguments: { period: 'week' } };

test('a valid tool call runs locally and a grounded answer is returned as the model wrote it', async () => {
  const ctx = await setup();
  const model = scripted(SUMMARY_WEEK, { answer: 'This calendar week you logged 5 meals, at least 1740 kcal.' });
  const r = await run(ctx, 'Summarize my week', model);
  assert.equal(r.source, 'model');
  assert.equal(r.text, 'This calendar week you logged 5 meals, at least 1740 kcal.');
  assert.match(r.facts, /calories 1740 kcal/); // deterministic figures shown beside the model's wording
  const toModel = JSON.stringify(model.sent[1]);
  assert.match(model.sent[1].at(-1).content, /"mealCount":5/);
  assert.doesNotMatch(toModel, /uidA|uidB|secret B food|scan-1|"id"/); // no owner, other user, scan ref or meal id reaches the model
});

test('unknown tools and invalid arguments are rejected and reported back to the model', async () => {
  const ctx = await setup();
  const model = scripted(
    { tool: 'run_sql', arguments: { sql: 'DELETE FROM meals' } },
    { tool: 'get_nutrition_summary', arguments: { period: 'week', userId: 'uidB' } },
    { tool: 'render_nutrition_chart', arguments: { period: 'week', type: 'pie', metric: 'fat' } },
    { answer: 'I could not read your data.' },
  );
  const r = await run(ctx, 'hello', model);
  assert.deepEqual(r.trace.tools, ['run_sql!unknown_tool', 'get_nutrition_summary!invalid_arguments', 'render_nutrition_chart!invalid_arguments']);
  assert.equal(r.charts.length, 0);
  assert.equal((await ctx.repo.listMeals()).length, 6); // nothing was deleted
});

test('metric "macros" on a pie (the macro split) is dropped; anywhere else it stays and is rejected', async () => {
  const pie = (metric, type = 'pie') => parseReply(JSON.stringify({ tool: 'render_nutrition_chart', arguments: { type, period: 'week', metric } })).args;
  assert.deepEqual(pie('macros'), { type: 'pie', period: 'week' });
  assert.deepEqual(pie('Macronutrients'), { type: 'pie', period: 'week' });
  assert.deepEqual(pie(''), { type: 'pie', period: 'week' });
  assert.deepEqual(pie('fat'), { type: 'pie', period: 'week', metric: 'fat' });
  assert.deepEqual(pie('macros', 'bar'), { type: 'bar', period: 'week', metric: 'macros' });
  const ctx = await setup();
  const r = await run(ctx, 'Show my macro split as a pie chart', scripted(
    { tool: 'render_nutrition_chart', arguments: { type: 'pie', period: 'week', metric: 'macros' } }, { answer: 'Here is your macro split.' }));
  assert.deepEqual(r.trace.tools, ['render_nutrition_chart']);
  assert.equal(r.charts[0].type, 'pie');
});

test('numbers the tools did not produce are rejected and replaced by a built-in answer', async () => {
  const ctx = await setup();
  const r = await run(ctx, 'Summarize my week', scripted(SUMMARY_WEEK, { answer: 'You ate 2450 kcal and 12 meals this week.' }));
  assert.equal(r.source, 'fallback');
  assert.equal(r.errorCode, 'ungrounded');
  assert.match(r.notice, /2450/);
  assert.match(r.text, /calories 1740 kcal/);
  assert.deepEqual(ungroundedNumbers('1. Eat more beans\n2. 1,740 kcal', [{ total: 1740 }]), []);
  assert.deepEqual(ungroundedNumbers('about 1800 kcal', [{ total: 1740 }]), [1800]);
  // Seen live: date parts in the prompt must not let the model's own arithmetic through.
  assert.deepEqual(ungroundedNumbers('You are 28 g short of your goal.', [{ goal: 90 }], ['week 2026-09-28']), [28]);
  assert.deepEqual(ungroundedNumbers('Your best day was 2026-10-06 (Oct 6).', [{ date: '2026-10-06' }]), []);
  assert.deepEqual(ungroundedNumbers('You ate well on 2026-10-03.', [{ date: '2026-10-06' }]), ['2026-10-03']);
});

test('written dates are checked against known dates; years and date parts never ground a quantity', () => {
  const src = [{ from: '2026-10-05', to: '2026-10-11', total: 841.8 }];
  const ok = (answer) => assert.deepEqual(ungroundedNumbers(answer, src), [], answer);
  // Valid dates in the forms seen from granite4.2 and qwen.
  ok('On Oct 5, 2026 you logged 841.8 kcal.');
  ok('Monday, October 11 2026 was your best day.');
  ok('Between Oct 5–11, 2026 you ate 841.8 kcal.');
  ok('From 5 October 2026 onwards.');
  ok('Your best day was Oct 11th.');
  ok('Week of Monday 2026‑10‑05: 841.8 kcal.'); // non-breaking hyphens (granite)
  assert.deepEqual(ungroundedNumbers('Monday 2026‑10‑12.', src), ['2026-10-12']);
  // Fabricated dates.
  assert.deepEqual(ungroundedNumbers('On Oct 12, 2026 you ate 841.8 kcal.', src), ['Oct 12, 2026']);
  assert.deepEqual(ungroundedNumbers('Oct 5, 2025 was a good day.', src), ['Oct 5, 2025']);
  assert.deepEqual(ungroundedNumbers('From Oct 5 to 19 you ate well.', src), ['Oct 5 to 19']);
  assert.deepEqual(ungroundedNumbers('On Feb 30 you ate well.', src), ['Feb 30']);
  // Fabricated values next to a valid date.
  assert.deepEqual(ungroundedNumbers('On Oct 5, 2026 you ate 900 kcal.', src), [900]);
  // Numbers in the wrong context: the year and day only exist inside dates, so they cannot be quantities.
  assert.deepEqual(ungroundedNumbers('You ate 2026 kcal this week.', src), [2026]);
  assert.deepEqual(ungroundedNumbers('You logged 11 meals.', src), [11]);
  assert.deepEqual(ungroundedNumbers('You are 28 g short; see Oct 5, 2026.', src, ['2026-09-28']), [28]);
  // Words that start like a month are not dates, so their number is still checked.
  assert.deepEqual(ungroundedNumbers('You marked 3 meals.', src), [3]);
});

test('suggestions that clash with stated diets or exclusions are replaced', async () => {
  const ctx = await setup(); // vegetarian, excludes chicken
  const model = scripted({ tool: 'get_user_preferences', arguments: {} }, { answer: 'Try a grilled salmon bowl or chicken rice.' });
  const r = await run(ctx, 'Suggest a meal', model);
  assert.equal(r.errorCode, 'preference_conflict');
  assert.match(r.notice, /chicken.*salmon|salmon.*chicken/);
  assert.match(r.text, /chickpea curry/);
  const ok = await run(ctx, 'Suggest a meal', scripted({ tool: 'get_user_preferences', arguments: {} }, { answer: 'Try a lentil and vegetable soup.' }));
  assert.equal(ok.source, 'model');
});

test('a diary question answered without reading data is sent back for a tool call', async () => {
  const ctx = await setup();
  const model = scripted({ answer: 'You eat very well!' }, SUMMARY_WEEK, { answer: 'You logged 5 meals this calendar week.' });
  const r = await run(ctx, 'Analyze my week', model);
  assert.equal(r.source, 'model');
  assert.equal(r.trace.nudges, 1);
});

test('an answer built on the wrong tool gets one hint naming the expected tool', async () => {
  const ctx = await setup();
  const model = scripted(SUMMARY_WEEK, { answer: 'Your top day had 145.5 g.' },
    { tool: 'render_nutrition_chart', arguments: { period: 'week', type: 'bar', metric: 'carbs' } }, { answer: 'Your highest day was 2026-10-06 with 56 g.' });
  const r = await run(ctx, 'Which day had the most carbs?', model);
  assert.match(model.sent[2].at(-1).content, /render_nutrition_chart/);
  assert.equal(r.trace.nudges, 1);
  assert.equal(r.text, 'Your highest day was 2026-10-06 with 56 g.');
});

test('tool-call rounds are bounded, and unparseable replies end in a built-in answer', async () => {
  const ctx = await setup();
  const loop = scripted(SUMMARY_WEEK);
  const r = await run(ctx, 'Summarize my week', loop);
  assert.equal(loop.sent.length, MAX_ROUNDS);
  assert.deepEqual(r.trace.tools, ['get_nutrition_summary']); // a repeated identical call is not run again
  assert.equal(r.errorCode, 'max_rounds');
  assert.match(r.text, /calories 1740 kcal/);

  assert.match(loop.sent[MAX_ROUNDS - 1].at(-1).content, /Last step: no more tools/); // the final round asks for an answer

  // Same data with different arguments counts as a repeat (this week = {"period":"week"} = its Monday as date).
  const same = scripted(SUMMARY_WEEK, { tool: 'get_nutrition_summary', arguments: { period: 'week', date: '2026-10-05' } },
    { answer: 'You logged 5 meals this calendar week.' });
  const r2 = await run(ctx, 'Summarize my week', same);
  assert.deepEqual(r2.trace.tools, ['get_nutrition_summary']);
  assert.equal(r2.trace.duplicates, 1);
  assert.match(same.sent[2].at(-1).content, /already have that result/);
  assert.equal(r2.source, 'model');

  const junk = await run(ctx, 'Summarize my week', scripted('Sure! Here is your summary...'));
  assert.equal(junk.trace.invalidReplies, MAX_ROUNDS);
  assert.equal(junk.source, 'fallback');
  assert.deepEqual(parseReply('```json\n{"name":"get_user_preferences","arguments":{}}\n```'), { tool: 'get_user_preferences', args: {} });
});

test('provider failures fall back to built-in answers with the reason; cancellation is not swallowed', async () => {
  const ctx = await setup();
  const down = await run(ctx, 'Summarize my week', scripted(new AiError('unreachable', 'gateway down', true)));
  assert.equal(down.source, 'fallback');
  assert.equal(down.errorCode, 'unreachable');
  assert.equal(down.notice, 'gateway down');
  assert.match(down.text, /calories 1740 kcal/);

  const unknown = await run(ctx, 'tell me a joke', scripted(new AiError('timeout', 'slow')));
  assert.equal(unknown.errorCode, 'timeout');
  assert.match(unknown.text, /I can answer questions about your logged meals/);

  await assert.rejects(run(ctx, 'Summarize my week', scripted(new AiError('cancelled', 'Request cancelled.'))), { code: 'cancelled' });
  const ac = new AbortController();
  const slow = { id: 'laptop', chat: async () => { ac.abort(); return { content: JSON.stringify(SUMMARY_WEEK) }; } };
  await assert.rejects(run(ctx, 'Summarize my week', slow, { signal: ac.signal }), { code: 'cancelled' });
});

test('chart data always comes from the engine, never from the model', async () => {
  const ctx = await setup();
  const model = scripted(
    { tool: 'render_nutrition_chart', arguments: { period: 'week', type: 'bar', metric: 'carbs', points: [999] } },
    { tool: 'render_nutrition_chart', arguments: { period: 'week', type: 'bar', metric: 'carbs' } },
    { answer: 'Your highest carbohydrate day was 2026-10-06 with 56 g.' },
  );
  const r = await run(ctx, 'Which day had the most carbs?', model);
  const direct = await ctx.tools.call('render_nutrition_chart', { period: 'week', type: 'bar', metric: 'carbs' });
  assert.deepEqual(r.charts, [direct.data]);
  assert.equal(r.source, 'model');
  const pie = await run(ctx, 'show my macro split');
  assert.equal(pie.charts[0].type, 'pie');
  assert.equal(pie.charts[0].slices.reduce((s, x) => s + x.pct, 0), 100);
});

// ---- Gateway provider (HTTP behaviour with a stubbed fetch).

const reply = (status, body) => ({ ok: status < 300, status, json: async () => body });
const gateway = (fetchImpl, extra = {}) => createGatewayProvider({ baseUrl: 'http://192.168.1.20:8100/', token: 'tok', fetchImpl, retryDelayMs: 1, ...extra });

test('gateway provider sends the token, request id and JSON mode to the configured address', async () => {
  let seen;
  const p = gateway(async (url, init) => { seen = { url, init }; return reply(200, { content: '{"answer":"hi"}', model: 'm' }); });
  const r = await p.chat({ messages: [{ role: 'user', content: 'hi' }], json: true, requestId: 'abc' });
  assert.equal(r.content, '{"answer":"hi"}');
  assert.equal(seen.url, 'http://192.168.1.20:8100/v1/chat');
  assert.equal(seen.init.headers.Authorization, 'Bearer tok');
  assert.deepEqual(JSON.parse(seen.init.body), { requestId: 'abc', messages: [{ role: 'user', content: 'hi' }], format: 'json' });
  assert.equal(normalizeBaseUrl('http://user:pw@host:1'), null);
  assert.equal(normalizeBaseUrl('http://host:8100/v1/chat'), null);
  await assert.rejects(createGatewayProvider({ baseUrl: '', token: 'x' }).chat({ messages: [] }), { code: 'not_configured' });
});

test('gateway errors map to user-facing codes; only connection failures and busy are retried', async () => {
  const cases = [
    [reply(401, { error: 'unauthorized' }), 'unauthorized', 1],
    [reply(503, { error: 'model_missing' }), 'model_missing', 1],
    [reply(503, { error: 'ollama_unavailable' }), 'ollama_unavailable', 1],
    [reply(504, { error: 'timeout' }), 'timeout', 1],
    [reply(429, { error: 'busy' }), 'busy', 2],
    [reply(500, null), 'gateway_error', 1],
    [new TypeError('Network request failed'), 'unreachable', 2],
  ];
  for (const [res, code, attempts] of cases) {
    let n = 0;
    const p = gateway(async () => { n++; if (res instanceof Error) throw res; return res; });
    await assert.rejects(p.chat({ messages: [] }), { code }, code);
    assert.equal(n, attempts, code);
  }
  let n = 0;
  const flaky = gateway(async () => (++n === 1 ? Promise.reject(new TypeError('fail')) : reply(200, { content: 'x' })));
  assert.equal((await flaky.chat({ messages: [] })).content, 'x');
});

const hang = (url, { signal }) => new Promise((_, reject) => signal.addEventListener('abort', () => reject(new Error('aborted'))));

test('requests time out, and the user can cancel one in flight', async () => {
  await assert.rejects(gateway(hang, { chatTimeoutMs: 30 }).chat({ messages: [] }), { code: 'timeout' });
  const ac = new AbortController();
  const pending = gateway(hang).chat({ messages: [], signal: ac.signal });
  setTimeout(() => ac.abort(), 10);
  await assert.rejects(pending, { code: 'cancelled' });
});

test('health reports connection, Ollama and model state without throwing', async () => {
  const h = (body) => gateway(async () => reply(200, body)).health();
  assert.deepEqual(await h({ ollama: 'ok', modelInstalled: true, model: 'm' }), { ok: true, message: 'Connected · m', model: 'm' });
  assert.equal((await h({ ollama: 'unavailable', model: 'm' })).code, 'ollama_unavailable');
  assert.equal((await h({ ollama: 'ok', modelInstalled: false, model: 'm' })).code, 'model_missing');
  assert.equal((await gateway(async () => reply(401, {})).health()).code, 'unauthorized');
  assert.equal((await gateway(hang, { healthTimeoutMs: 20 }).health()).code, 'timeout');
});

// ---- Conversation persistence.

test('conversations are ordered, isolated per identity, bounded, clearable and survive a bad row', async () => {
  const { db, repo } = await setup();
  await repo.appendAssistantMessage('c1', 'user', { text: 'A asks' });
  await repo.appendAssistantMessage('c1', 'assistant', { text: 'A answer', charts: [] });
  await repo.setIdentity('uidB');
  assert.deepEqual(await repo.getConversation(), []);
  await repo.appendAssistantMessage('c9', 'user', { text: 'B asks' });
  await repo.setIdentity('uidA');
  assert.deepEqual((await repo.getConversation()).map((m) => [m.role, m.text]), [['user', 'A asks'], ['assistant', 'A answer']]);

  await repo.appendAssistantMessage('c2', 'user', { text: 'new chat' }); // latest conversation wins
  assert.deepEqual((await repo.getConversation()).map((m) => m.text), ['new chat']);

  await db.runAsync("UPDATE assistant_messages SET content_json = '{oops' WHERE conversation_id = 'c2'");
  await repo.appendAssistantMessage('c2', 'assistant', { text: 'still works' });
  assert.deepEqual((await repo.getConversation()).map((m) => m.text), ['still works']);

  await assert.rejects(repo.appendAssistantMessage('c2', 'system', { text: 'x' }), TypeError);
  await assert.rejects(repo.appendAssistantMessage('', 'user', { text: 'x' }), TypeError);

  for (let i = 0; i < 205; i++) await repo.appendAssistantMessage('c3', 'user', { text: `m${i}` });
  const kept = await repo.getConversation();
  assert.equal(kept.length, 200);
  assert.equal(kept.at(-1).text, 'm204');

  await repo.clearConversations();
  assert.deepEqual(await repo.getConversation(), []);
  await repo.setIdentity('uidB');
  assert.deepEqual((await repo.getConversation()).map((m) => m.text), ['B asks']); // another identity's history is untouched
});
