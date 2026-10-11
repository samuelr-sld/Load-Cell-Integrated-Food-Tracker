"use strict";
/* Debugging UI for the inference testbench. All DOM is built with createElement/textContent (no innerHTML),
   so filenames, labels and error strings can never inject markup. */

const $ = (sel, root = document) => root.querySelector(sel);

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

const fmtMs = (v) => (v === null || v === undefined ? "n/a" : `${v < 10 ? v.toFixed(2) : v.toFixed(1)} ms`);
const fmtPct = (v) => (v === null || v === undefined ? "n/a" : `${(v * 100).toFixed(1)}%`);

async function api(path, opts) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON error body */ }
  if (!res.ok) throw Object.assign(new Error(body?.error?.message || `HTTP ${res.status}`), { status: res.status, body });
  return body;
}

const state = { limits: null };

/* ------------------------------------------------------------------ health */
async function refreshHealth() {
  const el = $("#health");
  try {
    const hl = await api("/api/health");
    const pills = [];
    if (!hl.models_loaded) {
      pills.push(h("span", { class: "pill bad" }, `Models NOT loaded: ${hl.load_error || "unknown error"}`));
    } else {
      const d = hl.device;
      pills.push(h("span", { class: "pill good" }, "Models loaded"));
      pills.push(h("span", { class: "pill" }, `Device: ${d.resolved}${d.gpu_name ? ` (${d.gpu_name})` : ""}`));
      if (d.fallback_reason) pills.push(h("span", { class: "pill bad" }, `Fallback: ${d.fallback_reason}`));
      pills.push(h("span", { class: "pill" }, `YOLO: ${hl.yolo.weights} (${hl.yolo.num_classes} classes, conf ${hl.yolo.conf})`));
      pills.push(h("span", { class: "pill" }, `Classifier: ${hl.classifier.architecture}, ${hl.classifier.num_classes} classes, ${hl.classifier.weights}`));
      const list = $("#classes");
      if (!list.children.length && hl.class_names) hl.class_names.forEach((n) => list.append(h("option", { value: n })));
    }
    const st = hl.store;
    pills.push(h("span", { class: "pill" }, `Memory store: ${st.count}/${st.max_results} results, ${(st.bytes / 2 ** 20).toFixed(1)}/${(st.max_bytes / 2 ** 20).toFixed(0)} MB`));
    el.replaceChildren(...pills);
    state.limits = hl.limits;
    $("#limits-hint").textContent = `JPEG / PNG, up to ${hl.limits.max_upload_mb} MB each (larger images are downscaled to ${hl.limits.max_image_side}px)`;
  } catch (e) {
    el.replaceChildren(h("span", { class: "pill bad" }, `Server unreachable: ${e.message}`));
  }
}

/* ------------------------------------------------------------------ upload queue */
const queue = [];
let running = false;

function queueRow(file) {
  const bar = h("progress", { max: "1", value: "0", "aria-label": `Upload progress for ${file.name}` });
  const st = h("span", { class: "st", role: "status" }, "Queued");
  const el = h("li", {}, h("span", { class: "name", title: file.name }, file.name), bar, st);
  $("#queue").prepend(el);
  return {
    set(kind, text, frac) {
      st.className = `st ${kind}`;
      st.textContent = text;
      if (frac !== undefined) bar.value = frac;
      if (kind === "processing") bar.removeAttribute("value");   // indeterminate while the server works
      if (kind === "done") bar.value = 1;
    },
  };
}

function clientProblem(file) {
  if (!/\.(jpe?g|png)$/i.test(file.name)) return "Rejected: only .jpg, .jpeg and .png files are accepted";
  if (state.limits && file.size > state.limits.max_upload_mb * 2 ** 20) return `Rejected: larger than ${state.limits.max_upload_mb} MB`;
  if (file.size === 0) return "Rejected: file is empty";
  return null;
}

function enqueue(files) {
  for (const file of files) {
    const row = queueRow(file);
    const problem = clientProblem(file);
    if (problem) { row.set("error", problem, 0); continue; }
    queue.push({ file, row });
  }
  pump();
}

async function pump() {
  if (running) return;
  running = true;
  while (queue.length) await uploadOne(queue.shift());   // strictly sequential, like the server's GPU lock
  running = false;
  refreshHealth();
}

function uploadOne({ file, row }) {
  return new Promise((resolve) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/inference");
    xhr.responseType = "json";
    row.set("", "Uploading 0%", 0);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) row.set("", `Uploading ${Math.round((e.loaded / e.total) * 100)}%`, e.loaded / e.total);
    };
    xhr.upload.onload = () => row.set("processing", "Processing: YOLO segmentation, cropping, classification…");
    xhr.onerror = () => { row.set("error", "Network error: is the server still running?", 0); resolve(); };
    xhr.onload = () => {
      const body = xhr.response;
      const item = body?.items?.[0];
      if (item?.status === "completed") {
        const t = item.result.timings_ms;
        row.set("done", `Done: ${t.num_segments} segment(s), ${fmtMs(t.total_ms)}`);
        addCard(item.result, true);
        refreshSummary();
      } else {
        row.set("error", item?.error?.message || body?.error?.message || `HTTP ${xhr.status}`, 0);
      }
      resolve();
    };
    const fd = new FormData();
    fd.append("files", file, file.name);
    xhr.send(fd);
  });
}

/* ------------------------------------------------------------------ results */
function imgFigure(url, caption, cls = "") {
  return h("figure", { class: cls },
    h("img", { class: "zoomable", src: url, alt: caption, loading: "lazy", onclick: () => openLightbox(url, caption) }),
    h("figcaption", {}, caption));
}

function openLightbox(url, caption) {
  const img = $("#lightbox-img");
  img.src = url; img.alt = caption;
  $("#lightbox").showModal();
}

function addCard(res, prepend) {
  $("#empty")?.remove();
  const card = renderCard(res);
  if (prepend) $("#results").prepend(card); else $("#results").append(card);
}

function renderCard(res) {
  const classified = res.segments.filter((s) => s.status === "classified").length;
  const evalSpan = h("span", { class: "meta" });
  const refreshEval = () => { evalSpan.textContent = `evaluated ${countEvaluated(res)}/${classified}`; };
  refreshEval();

  const card = h("article", { class: "card", "data-result-id": res.result_id },
    h("header", {},
      h("h3", {}, res.filename),
      h("span", { class: "meta" }, `${new Date(res.created_at).toLocaleString()} · result ${res.result_id}`),
      h("span", { class: "meta" }, `${res.segments.length} segments (${classified} classified)`),
      evalSpan,
      h("span", { class: "spacer" }),
      h("span", { class: "meta" }, `device ${res.device} · ${fmtMs(res.timings_ms.total_ms)}`)),
    h("div", { class: "body" },
      ...res.notes.map((n) => h("div", { class: "note" }, n)),
      h("div", { class: "two" },
        h("section", {}, h("h4", {}, "A. Original image"),
          imgFigure(res.image_urls.original, `Original ${res.image.width}×${res.image.height}px${res.image.downscaled ? ` (downscaled from ${res.image.original_width}×${res.image.original_height})` : ""}`)),
        h("section", {}, h("h4", {}, "B. YOLO segmentation"),
          imgFigure(res.image_urls.overlay, "Overlay: #instance, YOLO class, confidence"),
          segmentTable(res))),
      h("section", {}, h("h4", {}, "C + D. Masked crops and MobileNet predictions"),
        res.segments.length ? h("div", { class: "segs" }, res.segments.map((s) => segmentBlock(res, s, refreshEval)))
          : h("p", { class: "empty" }, "YOLO detected no regions, so the classifier was not called.")),
      h("section", {}, h("h4", {}, "E. Inference performance"), perfTable(res))));
  return card;
}

function countEvaluated(res) {
  return res.segments.filter((s) => s.feedback && ["correct", "incorrect"].includes(s.feedback.verdict)).length;
}

function swatch(rgb) { return h("span", { class: "swatch", style: `background: rgb(${rgb.join(",")})` }); }

function segmentTable(res) {
  if (!res.segments.length) return h("p", { class: "empty" }, "No segments.");
  return h("table", {},
    h("thead", {}, h("tr", {}, ["#", "YOLO class", "YOLO conf", "box (px)", "mask px", "status"].map((t) => h("th", {}, t)))),
    h("tbody", {}, res.segments.map((s) => h("tr", {},
      h("td", {}, swatch(s.color_rgb), `#${s.id}`), h("td", {}, s.yolo_class_name), h("td", {}, s.yolo_confidence.toFixed(3)),
      h("td", { class: "mono" }, s.box_xyxy.join(", ")), h("td", {}, String(s.mask_area_px)),
      h("td", {}, s.status === "classified" ? "classified" : `skipped: ${s.skip_reason}`)))));
}

function segmentBlock(res, s, onFeedbackChange) {
  const head = h("h3", {}, swatch(s.color_rgb), `Segment #${s.id}`,
    h("span", { class: "sub" }, `YOLO: ${s.yolo_class_name} (${s.yolo_confidence.toFixed(3)}) · crop ${s.crop_size_wh[0]}×${s.crop_size_wh[1]}px · mask fills ${(s.mask_fill_ratio * 100).toFixed(0)}% of box`));
  if (s.status !== "classified") {
    return h("div", { class: "seg skipped" }, h("div", {}, head, h("p", { class: "empty" }, `Not classified: ${s.skip_reason}`)));
  }
  const u = s.image_urls;
  const left = h("div", {}, head,
    h("div", { class: "crops" },
      imgFigure(u.masked, `Masked crop ${s.crop_size_wh[0]}×${s.crop_size_wh[1]} (classifier input source)`),
      imgFigure(u.bbox, "Bounding-box crop (comparison only)"),
      imgFigure(u.model_input, "Model input 224×224 (letterboxed)")));
  const top = s.predictions[0];
  const right = h("div", {},
    h("div", { class: "top1" }, top.label, h("small", {}, `top-1 · softmax ${top.softmax_score.toFixed(3)}`)),
    h("table", {},
      h("thead", {}, h("tr", {}, [h("th", {}, "#"), h("th", {}, "Food-101 class"), h("th", {}, "softmax score (uncalibrated)"), h("th", {}, "")])),
      h("tbody", {}, s.predictions.map((p) => h("tr", { class: p.rank === 1 ? "rank1" : "" },
        h("td", {}, String(p.rank)), h("td", {}, p.label), h("td", { class: "mono" }, p.softmax_score.toFixed(4)),
        h("td", {}, h("div", { class: "bar" }, h("i", { style: `width:${Math.max(1, p.softmax_score * 100).toFixed(1)}%` }))))))),
    feedbackControl(res, s, onFeedbackChange));
  return h("div", { class: "seg" }, left, right);
}

function feedbackControl(res, seg, onChange) {
  const url = `/api/results/${res.result_id}/segments/${seg.id}/feedback`;
  const msg = h("span", { class: "fb-msg", role: "status" });
  const input = h("input", { type: "text", list: "classes", placeholder: "actual food (optional)", "aria-label": `Actual food label for segment ${seg.id}`, autocomplete: "off" });
  const buttons = {};
  const mk = (v, text, title) => (buttons[v] = h("button", { class: "btn", type: "button", "data-v": v, title, onclick: () => send(v) }, text));
  const reset = h("button", { class: "link", type: "button", onclick: clear }, "reset");

  function paint() {
    const fb = seg.feedback;
    for (const [v, b] of Object.entries(buttons)) { b.classList.toggle("active", fb?.verdict === v); b.setAttribute("aria-pressed", String(fb?.verdict === v)); }
    reset.hidden = !fb;
    if (fb?.actual_label) input.value = fb.actual_label_raw;
    msg.classList.remove("err");
    msg.textContent = fb?.actual_label
      ? (fb.in_taxonomy ? `Actual "${fb.actual_label}" is a classifier class: counts toward top-5.` : `Actual "${fb.actual_label}" is outside the classifier's 101 classes: excluded from top-5.`)
      : (fb?.verdict === "incorrect" ? "No actual label given: excluded from top-5." : "");
  }
  async function send(verdict) {
    try {
      const r = await api(url, { method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ verdict, actual_label: verdict === "incorrect" && input.value.trim() ? input.value.trim() : null }) });
      seg.feedback = r.feedback;
      paint(); onChange(); refreshSummary();
    } catch (e) { msg.textContent = e.message; msg.classList.add("err"); }
  }
  async function clear() {
    try { await api(url, { method: "DELETE" }); seg.feedback = null; input.value = ""; paint(); onChange(); refreshSummary(); }
    catch (e) { msg.textContent = e.message; msg.classList.add("err"); }
  }
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") send("incorrect"); });   // typing a label implies "incorrect"
  const box = h("div", { class: "fb" },
    mk("correct", "✓ Top-1 correct", "The top-1 prediction is the right food"),
    mk("incorrect", "✗ Incorrect", "Top-1 is wrong; optionally type the real food, then press Enter or this button"),
    mk("ambiguous", "? Ambiguous / not evaluable", "Exclude this segment from accuracy"),
    input, reset, msg);
  paint();
  return box;
}

function perfTable(res) {
  const t = res.timings_ms, bd = t.yolo_ultralytics_breakdown_ms || {};
  const rows = [
    ["Device", res.device],
    ["Segments detected / classified", `${t.num_segments} / ${t.num_classified}`],
    ["YOLO inference (predict + masks to CPU)", `${fmtMs(t.yolo_ms)}${bd.inference !== undefined ? `  [Ultralytics: pre ${fmtMs(bd.preprocess)}, net ${fmtMs(bd.inference)}, post ${fmtMs(bd.postprocess)}]` : ""}`],
    ["Cropping + classifier preprocessing", fmtMs(t.crop_preprocess_ms)],
    ["MobileNet classification (all segments)", fmtMs(t.classify_ms)],
    ["MobileNet per classified segment", fmtMs(t.classify_ms_per_segment)],
    ["Pipeline inference (YOLO + crop + MobileNet)", fmtMs(t.inference_ms)],
    ["Total wall clock (adds decode, overlay render, JPEG encode)", fmtMs(t.total_ms)],
  ];
  return h("table", {}, h("tbody", {}, rows.map(([k, v]) => h("tr", {}, h("th", {}, k), h("td", {}, v)))));
}

/* ------------------------------------------------------------------ summary */
async function refreshSummary() {
  const el = $("#summary");
  try {
    const s = await api("/api/evaluation/summary");
    const c = s.counts;
    const stat = (value, label) => h("div", { class: "stat" }, h("b", {}, String(value)), h("span", {}, label));
    el.replaceChildren(
      h("div", { class: "banner" }, s.label),
      h("div", { class: "stats" },
        stat(c.images_processed, "images processed"),
        stat(c.segments_classified, `segments classified (${c.segments_skipped} skipped)`),
        stat(c.segments_manually_evaluated, `manually evaluated (${c.ambiguous_or_not_evaluable} ambiguous, ${c.not_yet_evaluated} pending)`),
        stat(fmtPct(s.top1_accuracy), `top-1 accuracy, n=${s.top1_denominator}`),
        stat(fmtPct(s.top5_accuracy), `top-5 accuracy, n=${s.top5_denominator} (${s.top5_excluded_segments} excluded: no in-taxonomy label)`),
        stat(fmtMs(s.timing.avg_inference_ms), "avg pipeline inference / image"),
        stat(fmtMs(s.timing.avg_total_pipeline_ms), "avg total wall clock / image"),
        stat(fmtMs(s.timing.avg_classification_ms_per_segment), "avg MobileNet time / segment")),
      h("ul", { class: "caveats" }, s.caveats.map((t) => h("li", {}, t))));
  } catch (e) { el.textContent = `Summary unavailable: ${e.message}`; }
}

/* ------------------------------------------------------------------ wiring */
async function loadExisting() {
  try {
    const { results } = await api("/api/results");
    for (const r of results.slice().reverse()) {                // list is newest-first; prepend oldest-first to keep that order
      if (r.status !== "completed") continue;
      addCard(await api(`/api/results/${r.result_id}`), true);
    }
  } catch { /* server may not be up yet */ }
}

function wireUpload() {
  const zone = $("#dropzone"), input = $("#file-input");
  input.addEventListener("change", () => { enqueue([...input.files]); input.value = ""; });
  zone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  for (const ev of ["dragenter", "dragover"]) zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("over"); });
  for (const ev of ["dragleave", "drop"]) zone.addEventListener(ev, () => zone.classList.remove("over"));
  zone.addEventListener("drop", (e) => { e.preventDefault(); enqueue([...e.dataTransfer.files]); });
  // dropping a file outside the zone must not make the browser navigate away to the image
  for (const ev of ["dragover", "drop"]) window.addEventListener(ev, (e) => e.preventDefault());
}

function wireClear() {
  const btn = $("#clear-btn");
  let timer = null;
  btn.addEventListener("click", async () => {
    if (!btn.classList.contains("confirm")) {
      btn.classList.add("confirm"); btn.textContent = "Click again to confirm";
      timer = setTimeout(() => { btn.classList.remove("confirm"); btn.textContent = "Clear all results"; }, 4000);
      return;
    }
    clearTimeout(timer);
    btn.classList.remove("confirm"); btn.textContent = "Clear all results";
    try {
      await api("/api/results", { method: "DELETE" });
      $("#results").replaceChildren(h("p", { class: "empty", id: "empty" }, "Results cleared. Upload an image to begin."));
      $("#queue").replaceChildren();
      refreshSummary(); refreshHealth();
    } catch (e) { alert(`Could not clear results: ${e.message}`); }
  });
}

wireUpload();
wireClear();
refreshHealth().then(() => { loadExisting(); refreshSummary(); });
