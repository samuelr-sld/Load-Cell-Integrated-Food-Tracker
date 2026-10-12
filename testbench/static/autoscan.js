"use strict";
/* Load-cell scans on the dashboard. Polls the existing scan API (read-only: it never starts inference) and renders a finished
   scan's stored result with the same finishUpload/renderCard path a manual upload uses. Loaded after app.js (shares its helpers). */

const AUTOSCAN_POLL_MS = 1500;
const autoscan = { current: null, shown: new Set(), seq: 0, applied: 0, first: true };

function autoscanFoods(scan) {
  return h("ul", { class: "caveats" }, scan.foods.map((f) => h("li", {},
    `Food ${f.region_id}: `, f.candidates.map((c) => {
      const kcal = c.per_100g && c.per_100g.energy_kcal != null ? ` · ${c.per_100g.energy_kcal} kcal/100 g` : "";
      return `${c.label} ${fmtPct(c.confidence)} (${c.source}${kcal})`;
    }).join("; "))));
}

function autoscanBanner(scan) {
  const box = $("#autoscan");
  const grams = `${scan.measured_total_grams} g on the scale` + (scan.tare_g ? ` (${scan.net_grams} g net of the ${scan.tare_g} g plate)` : "");
  box.hidden = false;
  box.className = `panel ${scan.status}`;
  if (scan.status === "processing") {
    box.replaceChildren(h("h3", {}, h("span", { class: "spinner" }), "Analyzing meal…"), h("p", {}, `Automatic scan in progress · ${grams}`));
  } else if (scan.status === "error") {
    box.replaceChildren(h("h3", {}, "Automatic scan failed"), h("p", {}, `${scan.error?.message || "unknown error"} · ${grams}`),
      h("p", { class: "hint" }, "Nothing is retried automatically. Use Camera capture or upload a photo above to analyze this meal manually."));
  } else {
    box.replaceChildren(...[h("h3", {}, "Automatic scan"), h("p", {}, `${grams} · scan ${scan.scan_id} · result ${scan.result_id}`),
      scan.needs_manual_food ? h("p", { class: "hint" }, "No food detected. The result card below still shows the captured image.") : autoscanFoods(scan),
      scan.suggested_grams ? h("p", {}, `Single food: ${scan.suggested_grams} g is proposed as its portion (confirm it on the phone).`) : null].filter(Boolean));
  }
}

/* Applies one poll response. Exported on window for the browser/unit tests. seq orders overlapping requests: a slow older
   response can never replace a newer one, and scans older than the one on screen are dropped. */
async function autoscanApply(scan, seq) {
  if (seq < autoscan.applied) return;
  autoscan.applied = seq;
  if (!scan) return;
  if (autoscan.current && scan.timestamp < autoscan.current.timestamp) return;
  const wasShown = autoscan.shown.has(scan.scan_id);
  const firstLook = autoscan.first;
  autoscan.first = false;
  // Page (re)load: a finished scan is already among the stored results loaded at startup, so it is not shown again.
  if (firstLook && scan.status !== "processing") { autoscan.current = scan; autoscan.shown.add(scan.scan_id); return; }
  const changed = !autoscan.current || autoscan.current.scan_id !== scan.scan_id || autoscan.current.status !== scan.status;
  autoscan.current = scan;
  if (changed) autoscanBanner(scan);
  if (scan.status === "ready" && scan.result_id && !wasShown) {
    autoscan.shown.add(scan.scan_id);
    try {
      const res = await api(`/api/results/${scan.result_id}`);
      await finishUpload(res, scan.camera_id || null, { set() {}, text() { return ""; } });
    } catch (e) {
      if (e.status === 404) {                      // the result is gone from server memory (e.g. server restart): say so once, do not loop
        $("#autoscan").append(h("p", { class: "hint" }, "The detailed result is no longer in server memory. Capture again to see it here."));
      } else autoscan.shown.delete(scan.scan_id);  // transient failure: retry on the next poll
      console.warn("autoscan: result not available", e.message);
    }
  } else if (scan.status === "error") autoscan.shown.add(scan.scan_id);
}

async function autoscanPoll() {
  const seq = ++autoscan.seq;
  try {
    const { scan } = await api("/api/scans/latest?include_acknowledged=true");
    await autoscanApply(scan, seq);
  } catch { /* server restarting or network hiccup: try again next tick */ }
}

if (typeof document !== "undefined" && !window.__AUTOSCAN_TEST__) setInterval(autoscanPoll, AUTOSCAN_POLL_MS);
window.autoscanApply = autoscanApply;
window.autoscanState = autoscan;
