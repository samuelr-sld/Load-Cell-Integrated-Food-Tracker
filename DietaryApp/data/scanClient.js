// Phone -> laptop scan API (testbench, port 8000). Same Wi-Fi address and pairing token as the AI gateway.
import { normalizeBaseUrl } from '../ai/providers.js';

// The gateway address (http://192.168.1.20:8100) with the testbench port.
export const scanBaseUrl = (gatewayUrl, port = 8000) => {
  const u = normalizeBaseUrl(gatewayUrl);
  return u && `${u.replace(/:\d+$/, '')}:${port}`;
};

export const createScanClient = ({ baseUrl, token, fetchImpl = fetch, timeoutMs = 4000 }) => {
  const call = async (path, init = {}) => {
    const c = new AbortController();
    const timer = setTimeout(() => c.abort(), timeoutMs);
    try {
      const res = await fetchImpl(baseUrl + path, {
        ...init, signal: c.signal, headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      });
      if (!res.ok) throw new Error(`scan API ${res.status}`);
      return await res.json();
    } finally {
      clearTimeout(timer);
    }
  };
  return {
    latest: async () => (await call('/api/scans/latest')).scan,
    acknowledge: (scanId, outcome) => call(`/api/scans/${encodeURIComponent(scanId)}/acknowledge`, { method: 'POST', body: JSON.stringify({ outcome }) }),
  };
};

// Polls for the newest unacknowledged scan and hands each scan_id to onScan once (processing scans wait for the next tick).
// canShow() false (a form is already open) leaves the scan unseen so it is offered later.
export const startScanPolling = ({ getClient, onScan, canShow = () => true, intervalMs = 2500 }) => {
  const seen = new Set();
  let stopped = false;
  let timer;
  const tick = async () => {
    try {
      const client = await getClient();
      const scan = client ? await client.latest() : null;
      if (scan && scan.status !== 'processing' && !seen.has(scan.scan_id) && canShow()) {
        seen.add(scan.scan_id);
        onScan(scan, client);
      }
    } catch { /* laptop unreachable or off: try again next tick */ }
    if (!stopped) timer = setTimeout(tick, intervalMs);
  };
  tick();
  return () => { stopped = true; clearTimeout(timer); };
};
