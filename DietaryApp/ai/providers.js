// AI providers. The assistant only sees this interface, so the laptop gateway can later be swapped for an
// on-device model without touching the orchestrator or the UI:
//   { id, label, health({ signal }) -> { ok, code?, message, model? },
//     chat({ messages, json, requestId, signal }) -> { content, model, timings? } }   (chat throws AiError)
import { mockReply } from './assistant.js';

export class AiError extends Error {
  constructor(code, message, retryable = false) {
    super(message);
    this.code = code;
    this.retryable = retryable;
  }
}

// What the user reads when laptop AI cannot answer. The diary and dashboards never depend on any of this.
export const AI_MESSAGES = {
  not_configured: 'Laptop AI is not set up. Enter the gateway address and pairing token in Settings.',
  unreachable: 'The laptop AI gateway cannot be reached. Check that the laptop is on the same Wi-Fi and the gateway is running.',
  unauthorized: 'The laptop rejected the pairing token. Copy the token shown when the gateway starts.',
  busy: 'The laptop AI is busy with another request. Try again in a moment.',
  timeout: 'The laptop AI took too long to answer.',
  ollama_unavailable: 'The gateway is running, but Ollama is not. Start the Ollama app on the laptop.',
  model_missing: 'The AI model is not installed on the laptop (see the gateway window for the pull command).',
  cancelled: 'Request cancelled.',
  bad_response: 'The laptop AI sent an unexpected response.',
  gateway_error: 'The laptop AI gateway reported an error.',
};
const fail = (code, retryable = false) => new AiError(code, AI_MESSAGES[code], retryable);

let seq = 0;
export const newId = () => `${Date.now().toString(36)}-${(seq++).toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

// http(s)://host[:port] only (no path, query or credentials); returns null when invalid.
export const normalizeBaseUrl = (raw) => {
  const u = String(raw ?? '').trim().replace(/\/+$/, '');
  return /^https?:\/\/[A-Za-z0-9.-]+(:\d{1,5})?$/.test(u) ? u : null;
};

// An AbortSignal that fires when the caller aborts or after `ms`.
const deadline = (outer, ms) => {
  const c = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; c.abort(); }, ms);
  const onAbort = () => c.abort();
  if (outer?.aborted) c.abort();
  else outer?.addEventListener('abort', onAbort);
  return { signal: c.signal, timedOut: () => timedOut, done: () => { clearTimeout(timer); outer?.removeEventListener('abort', onAbort); } };
};

const pause = (ms, signal) => new Promise((resolve) => {
  const t = setTimeout(resolve, ms);
  signal?.addEventListener('abort', () => { clearTimeout(t); resolve(); });
});

const STATUS_CODES = { 401: 'unauthorized', 403: 'unauthorized', 429: 'busy', 504: 'timeout' };

export const createGatewayProvider = ({ baseUrl, token, chatTimeoutMs = 120000, healthTimeoutMs = 5000, retries = 1, retryDelayMs = 700, fetchImpl = fetch }) => {
  const url = normalizeBaseUrl(baseUrl);

  const request = async (path, { body, signal, timeoutMs, requestId }) => {
    if (!url || !token) throw fail('not_configured');
    for (let attempt = 0; ; attempt++) {
      const d = deadline(signal, timeoutMs);
      let res;
      let data = null;
      try {
        res = await fetchImpl(`${url}${path}`, {
          method: body ? 'POST' : 'GET',
          headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', 'X-Request-Id': requestId },
          body: body && JSON.stringify(body),
          signal: d.signal,
        });
        data = await res.json().catch(() => null);
      } catch {
        if (signal?.aborted) throw fail('cancelled');
        if (d.timedOut()) throw fail('timeout', true);
        res = null;
      } finally {
        d.done();
      }
      // Only retry when nothing was generated: the connection failed, or the gateway was busy.
      const code = !res ? 'unreachable' : res.ok ? null : data?.error === 'model_missing' || data?.error === 'ollama_unavailable' ? data.error : STATUS_CODES[res.status] ?? 'gateway_error';
      if (!code) return data ?? {};
      if ((code === 'unreachable' || code === 'busy') && attempt < retries) {
        await pause(retryDelayMs, signal);
        if (signal?.aborted) throw fail('cancelled');
        continue;
      }
      throw fail(code, code === 'unreachable' || code === 'busy' || code === 'timeout');
    }
  };

  return {
    id: 'laptop',
    label: 'Laptop AI (Ollama)',
    async health({ signal } = {}) {
      try {
        const h = await request('/v1/health', { signal, timeoutMs: healthTimeoutMs, requestId: newId() });
        if (h.ollama !== 'ok') return { ok: false, code: 'ollama_unavailable', message: AI_MESSAGES.ollama_unavailable, model: h.model };
        if (!h.modelInstalled) return { ok: false, code: 'model_missing', message: AI_MESSAGES.model_missing, model: h.model };
        return { ok: true, message: `Connected · ${h.model}`, model: h.model };
      } catch (e) {
        if (e.code === 'cancelled') throw e;
        return { ok: false, code: e.code, message: e.message };
      }
    },
    async chat({ messages, json = false, requestId = newId(), signal }) {
      const data = await request('/v1/chat', { body: { requestId, messages, ...(json && { format: 'json' }) }, signal, timeoutMs: chatTimeoutMs, requestId });
      if (typeof data.content !== 'string') throw fail('bad_response');
      return { content: data.content, model: data.model, timings: data.timings };
    },
  };
};

// Deterministic stand-in with the same interface: plans tool calls by keyword and answers from the tool
// results alone. Used by tests, for offline UI work, and as the fallback when laptop AI is unavailable.
export const createMockProvider = ({ today }) => ({
  id: 'offline',
  label: 'Built-in answers (no AI)',
  health: async () => ({ ok: true, message: 'Answers come straight from your diary; no language model.' }),
  async chat({ messages, signal }) {
    if (signal?.aborted) throw fail('cancelled');
    return { content: JSON.stringify(mockReply(messages, today())), model: 'built-in' };
  },
});
