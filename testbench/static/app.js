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

const state = { limits: null, defaultMode: "compare", autoLocalize: true, meta: {} };   // meta[result_id] = { cameraId, note }
const MODE_LABEL = { yolo_only: "YOLO only", compare: "YOLO + MobileNet (comparison)" };

function selectedMode() { return $('input[name="mode"]:checked')?.value || state.defaultMode; }
function applyDefaultMode(mode) {
  state.defaultMode = mode;
  if (!$('input[name="mode"]:checked')) { const r = $(`input[name="mode"][value="${mode}"]`); if (r) r.checked = true; }
}

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
      const ylist = $("#yolo-classes");
      if (!ylist.children.length && hl.yolo_class_names) hl.yolo_class_names.forEach((n) => ylist.append(h("option", { value: n })));
      state.yoloClasses = hl.yolo_class_names || [];
      applyDefaultMode(hl.default_mode || "compare");
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
    text: () => st.textContent,
  };
}

function clientProblem(file) {
  if (!/\.(jpe?g|png)$/i.test(file.name)) return "Rejected: only .jpg, .jpeg and .png files are accepted";
  if (state.limits && file.size > state.limits.max_upload_mb * 2 ** 20) return `Rejected: larger than ${state.limits.max_upload_mb} MB`;
  if (file.size === 0) return "Rejected: file is empty";
  return null;
}

function enqueue(files, opts = {}) {   // opts.cameraId: the files are frames from that camera (enables its saved calibration)
  const mode = selectedMode();                                 // fixed when the file is added, not when it is uploaded
  for (const file of files) {
    const row = queueRow(file);
    const problem = clientProblem(file);
    if (problem) { row.set("error", problem, 0); continue; }
    queue.push({ file, row, mode, cameraId: opts.cameraId || null });
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

function uploadOne({ file, row, mode, cameraId }) {
  return new Promise((resolve) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/inference");
    xhr.responseType = "json";
    row.set("", "Uploading 0%", 0);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) row.set("", `Uploading ${Math.round((e.loaded / e.total) * 100)}%`, e.loaded / e.total);
    };
    xhr.upload.onload = () => row.set("processing", mode === "yolo_only" ? "Processing: YOLO segmentation and cropping…" : "Processing: YOLO segmentation, cropping, MobileNet classification…");
    xhr.onerror = () => { row.set("error", "Network error: is the server still running?", 0); resolve(); };
    xhr.onload = () => {
      const body = xhr.response;
      const item = body?.items?.[0];
      if (item?.status === "completed") {
        const t = item.result.timings_ms;
        row.set("done", `Done: ${t.num_segments} segment(s), ${fmtMs(t.total_ms)}`);
        finishUpload(item.result, cameraId, row);
        refreshSummary();
      } else {
        row.set("error", item?.error?.message || body?.error?.message || `HTTP ${xhr.status}`, 0);
      }
      resolve();
    };
    const fd = new FormData();
    fd.append("mode", mode);                                   // text field before the file so it is parsed first
    fd.append("files", file, file.name);
    xhr.send(fd);
  });
}

/* Plate localization for a fresh result (saved calibration -> OpenCV -> ask for manual calibration), then show the card.
   Failure here never blocks the card: the manual three-point calibration in section F stays available. */
async function finishUpload(res, cameraId, row) {
  let shown = res;
  state.meta[res.result_id] = { cameraId, note: null };
  if (state.autoLocalize) {
    try {
      const r = await api(`/api/results/${res.result_id}/plate/auto`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cameraId ? { camera_id: cameraId } : {}) });
      if (r.applied) { shown = await api(`/api/results/${res.result_id}`); row.set("done", `${row.text()} · plate: ${r.source}`); }
      else { state.meta[res.result_id].note = r.message; row.set("done", `${row.text()} · plate: needs manual calibration`); }
    } catch (e) { state.meta[res.result_id].note = `Automatic plate localization failed: ${e.message}`; }
  }
  addCard(shown, true);
}

/* ------------------------------------------------------------------ results */
// width/height attributes let the browser reserve the image's space before it loads (otherwise the page jumps under the cursor)
function imgFigure(url, caption, cls = "", width, height) {
  return h("figure", { class: cls },
    h("img", { class: "zoomable", src: url, alt: caption, width, height, loading: "lazy", onclick: () => openLightbox(url, caption) }),
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
  const yoloOnly = res.mode === "yolo_only";
  const usable = res.segments.filter((s) => s.status !== "skipped").length;
  const cmp = yoloOnly ? null : comparisonTable(res);
  const evalSpan = h("span", { class: "meta" });
  const refreshEval = () => {
    const y = `YOLO judged ${countEvaluated(res, "yolo_feedback")}/${usable}`;
    evalSpan.textContent = yoloOnly ? y : `${y} · MobileNet judged ${countEvaluated(res, "feedback")}/${usable}`;
    cmp?.refresh();
  };
  refreshEval();
  const plate = plateSection(res, () => refreshCard(res.result_id));

  const card = h("article", { class: "card", "data-result-id": res.result_id },
    h("header", {},
      h("h3", {}, res.filename),
      h("span", { class: `badge ${res.mode}` }, MODE_LABEL[res.mode] || res.mode),
      h("span", { class: "meta" }, `${new Date(res.created_at).toLocaleString()} · result ${res.result_id}`),
      h("span", { class: "meta" }, `${res.segments.length} segments (${usable} ${yoloOnly ? "identified" : "classified"})`),
      evalSpan,
      h("span", { class: "spacer" }),
      h("span", { class: "meta" }, `device ${res.device} · ${fmtMs(res.timings_ms.total_ms)}`)),
    h("div", { class: "body" },
      ...res.notes.map((n) => h("div", { class: "note" }, n)),
      h("div", { class: "two" },
        h("section", {}, h("h4", {}, "A. Original image"),
          imgFigure(res.image_urls.original, `Original ${res.image.width}×${res.image.height}px${res.image.downscaled ? ` (downscaled from ${res.image.original_width}×${res.image.original_height})` : ""}`, "", res.image.width, res.image.height)),
        h("section", {}, h("h4", {}, "B. YOLO segmentation"),
          imgFigure(res.image_urls.overlay, "Overlay: #instance, YOLO class, confidence", "", res.image.width, res.image.height),
          segmentTable(res))),
      cmp ? h("section", {}, h("h4", {}, "C. Model comparison: YOLO vs MobileNet on the same segments"), cmp.element) : null,
      h("section", {}, h("h4", {}, yoloOnly ? "C. Masked crops and YOLO food identity" : "D. Per-segment detail and evaluation"),
        res.segments.length ? h("div", { class: "segs" }, res.segments.map((s) => segmentBlock(res, s, refreshEval)))
          : h("p", { class: "empty" }, "YOLO detected no regions.")),
      res.regions?.length ? h("section", {}, h("h4", {}, "E. Confirm food identity and USDA nutrition"), regionsPanel(res)) : null,
      h("section", { class: "plate-section" }, h("h4", {}, "F. Plate analysis, estimated weights and nutrition"), plate.element),
      h("section", {}, h("h4", {}, "G. Inference performance"), perfTable(res))));
  // identity / portion edits in section E change what the plate analysis can resolve
  card.addEventListener("regions-changed", () => plate.load());
  plate.load();
  return card;
}

function countEvaluated(res, field) {
  return res.segments.filter((s) => s[field] && ["correct", "incorrect"].includes(s[field].verdict)).length;
}

function swatch(rgb) { return h("span", { class: "swatch", style: `background: rgb(${rgb.join(",")})` }); }

function segmentTable(res) {
  if (!res.segments.length) return h("p", { class: "empty" }, "No segments.");
  return h("table", {},
    h("thead", {}, h("tr", {}, ["#", "YOLO class", "class id", "YOLO conf", "box (px)", "mask px", "status"].map((t) => h("th", {}, t)))),
    h("tbody", {}, res.segments.map((s) => h("tr", {},
      h("td", {}, swatch(s.color_rgb), `#${s.segment_id}`), h("td", {}, s.yolo_class_name), h("td", {}, String(s.yolo_class_id)),
      h("td", {}, s.yolo_confidence.toFixed(3)),
      h("td", { class: "mono" }, s.box_xyxy.join(", ")), h("td", {}, String(s.mask_area_px)),
      h("td", {}, s.status === "skipped" ? `skipped: ${s.skip_reason}` : s.status)))));
}

function verdictMark(fb) {
  if (!fb) return h("span", { class: "vd none", title: "not judged yet" }, "–");
  const glyph = { correct: "✓", incorrect: "✗", ambiguous: "?" }[fb.verdict];
  return h("span", { class: `vd ${fb.verdict}`, title: fb.verdict + (fb.actual_label ? ` (actual: ${fb.actual_label})` : "") }, glyph);
}

/* Side-by-side view of both models for the same segments. Verdict cells follow the feedback buttons below. */
function comparisonTable(res) {
  const rows = res.segments.filter((s) => s.status !== "skipped");
  const body = h("tbody");
  const head = ["#", "masked crop", "YOLO (ingredient, FoodSeg103)", "MobileNet top-1 (dish, Food-101)", "MobileNet next two", "YOLO judged", "MobileNet judged"];
  const refresh = () => body.replaceChildren(...rows.map((s) => {
    const p = s.mobilenet?.predictions || [];
    return h("tr", {},
      h("td", {}, swatch(s.color_rgb), `#${s.segment_id}`),
      h("td", {}, h("img", { class: "thumb zoomable", src: s.image_urls.masked, alt: `Masked crop of segment ${s.segment_id}`, width: 56, height: 56, loading: "lazy",
        onclick: () => openLightbox(s.image_urls.masked, `Segment #${s.segment_id} masked crop`) })),
      h("td", {}, h("b", {}, s.yolo_class_name), ` · conf ${s.yolo_confidence.toFixed(3)}`),
      h("td", {}, p[0] ? [h("b", {}, p[0].label), ` · softmax ${p[0].softmax_score.toFixed(3)}`] : "–"),
      h("td", {}, p.slice(1, 3).map((x) => `${x.label} (${x.softmax_score.toFixed(2)})`).join(", ") || "–"),
      h("td", {}, verdictMark(s.yolo_feedback)), h("td", {}, verdictMark(s.feedback)));
  }));
  const element = h("div", {},
    h("p", { class: "info" }, "The models answer different questions: YOLO names an ingredient region (103 FoodSeg103 classes), MobileNet names a whole dish (101 Food-101 classes). " +
      "Their names are not expected to match, so judge each one against the crop using the buttons in the per-segment detail below."),
    rows.length ? h("table", { class: "cmp" }, h("thead", {}, h("tr", {}, head.map((t) => h("th", {}, t)))), body) : h("p", { class: "empty" }, "No segments were identified."));
  return { element, refresh };
}

function segmentBlock(res, s, onFeedbackChange) {
  const compare = res.mode === "compare";
  const head = h("h3", {}, swatch(s.color_rgb), `Segment #${s.segment_id}`,
    h("span", { class: "sub" }, `crop ${s.crop_size_wh[0]}×${s.crop_size_wh[1]}px · mask ${s.mask_area_px} px, fills ${(s.mask_fill_ratio * 100).toFixed(0)}% of box`));
  if (s.status === "skipped") {
    return h("div", { class: "seg skipped" }, h("div", {}, head,
      h("p", { class: "empty" }, `Skipped: ${s.skip_reason}. YOLO saw "${s.yolo_class_name}" (${s.yolo_confidence.toFixed(3)}) but the region is too small to use.`)));
  }
  const u = s.image_urls, [cw, ch] = s.crop_size_wh;
  const crops = [
    imgFigure(u.masked, `Masked crop ${cw}×${ch}${compare ? " (classifier input source)" : ""}`, "", cw, ch),
    imgFigure(u.bbox, "Bounding-box crop (comparison only)", "", cw, ch),
    imgFigure(u.mask, "Segmentation mask (PNG, white = food)", "", cw, ch),
  ];
  if (compare) crops.push(imgFigure(u.model_input, "Model input 224×224 (letterboxed)", "", 224, 224));
  const left = h("div", {}, head, h("div", { class: "crops" }, crops));

  const yolo = h("div", { class: "model" },
    h("h5", {}, "YOLO: primary food identity"),
    h("div", { class: "top1" }, s.yolo_class_name, h("small", {}, `class id ${s.yolo_class_id} · confidence ${s.yolo_confidence.toFixed(3)} (uncalibrated)`)),
    feedbackControl(res, s, onFeedbackChange, "yolo"));
  const panels = [yolo];
  if (compare) {
    const top = s.mobilenet.top1;
    panels.push(h("div", { class: "model" },
      h("h5", {}, "MobileNet: experimental comparison"),
      h("div", { class: "top1" }, top.label, h("small", {}, `top-1 · softmax ${top.softmax_score.toFixed(3)}`)),
      h("table", {},
        h("thead", {}, h("tr", {}, [h("th", {}, "#"), h("th", {}, "Food-101 class"), h("th", {}, "softmax score (uncalibrated)"), h("th", {}, "")])),
        h("tbody", {}, s.mobilenet.predictions.map((p) => h("tr", { class: p.rank === 1 ? "rank1" : "" },
          h("td", {}, String(p.rank)), h("td", {}, p.label), h("td", { class: "mono" }, p.softmax_score.toFixed(4)),
          h("td", {}, h("div", { class: "bar" }, h("i", { style: `width:${Math.max(1, p.softmax_score * 100).toFixed(1)}%` }))))))),
      feedbackControl(res, s, onFeedbackChange, "mobilenet")));
  }
  return h("div", { class: "seg" }, left, h("div", { class: "models" }, panels));
}

/* Verdict controls for one model's prediction on one segment. model: "yolo" judges YOLO's class, "mobilenet" its top-1. */
function feedbackControl(res, seg, onChange, model = "mobilenet") {
  const yolo = model === "yolo";
  const field = yolo ? "yolo_feedback" : "feedback";
  const who = yolo ? "YOLO" : "MobileNet";
  const url = `/api/results/${res.result_id}/segments/${seg.segment_id}/feedback`;
  const msg = h("span", { class: "fb-msg", role: "status" });
  const input = h("input", { type: "text", list: yolo ? "yolo-classes" : "classes", placeholder: yolo ? "actual ingredient (optional)" : "actual food (optional)",
    "aria-label": `Actual label for segment ${seg.segment_id} (${who})`, autocomplete: "off" });
  const buttons = {};
  const mk = (v, text, title) => (buttons[v] = h("button", { class: "btn", type: "button", "data-v": v, title, onclick: () => send(v) }, text));
  const reset = h("button", { class: "link", type: "button", onclick: clear }, "reset");

  function paint() {
    const fb = seg[field];
    for (const [v, b] of Object.entries(buttons)) { b.classList.toggle("active", fb?.verdict === v); b.setAttribute("aria-pressed", String(fb?.verdict === v)); }
    reset.hidden = !fb;
    if (fb?.actual_label) input.value = fb.actual_label_raw;
    msg.classList.remove("err");
    if (fb?.actual_label) {
      msg.textContent = yolo
        ? (fb.in_taxonomy ? `Actual "${fb.actual_label}" is one of YOLO's classes.` : `Actual "${fb.actual_label}" is outside YOLO's 103 classes.`)
        : (fb.in_taxonomy ? `Actual "${fb.actual_label}" is a classifier class: counts toward top-5.` : `Actual "${fb.actual_label}" is outside the classifier's 101 classes: excluded from top-5.`);
    } else {
      msg.textContent = fb?.verdict === "incorrect" && !yolo ? "No actual label given: excluded from top-5." : "";
    }
  }
  async function send(verdict) {
    try {
      const r = await api(url, { method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, verdict, actual_label: verdict === "incorrect" && input.value.trim() ? input.value.trim() : null }) });
      seg[field] = r.feedback;
      paint(); onChange(); refreshSummary();
    } catch (e) { msg.textContent = e.message; msg.classList.add("err"); }
  }
  async function clear() {
    try { await api(`${url}?model=${model}`, { method: "DELETE" }); seg[field] = null; input.value = ""; paint(); onChange(); refreshSummary(); }
    catch (e) { msg.textContent = e.message; msg.classList.add("err"); }
  }
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") send("incorrect"); });   // typing a label implies "incorrect"
  const box = h("div", { class: "fb" },
    mk("correct", yolo ? "✓ YOLO class correct" : "✓ Top-1 correct", yolo ? "YOLO's class is the right food/ingredient for this segment" : "The top-1 prediction is the right food"),
    mk("incorrect", "✗ Incorrect", "The prediction is wrong; optionally type the real food, then press Enter or this button"),
    mk("ambiguous", "? Ambiguous / not evaluable", "Exclude this segment from accuracy"),
    input, reset, msg);
  paint();
  return box;
}

function perfTable(res) {
  const t = res.timings_ms, bd = t.yolo_ultralytics_breakdown_ms || {}, compare = res.mode === "compare";
  const rows = [
    ["Mode", MODE_LABEL[res.mode] || res.mode],
    ["Device", res.device],
    [compare ? "Segments detected / classified" : "Segments detected / identified", `${t.num_segments} / ${compare ? t.num_classified : t.num_identified}`],
    ["YOLO inference (predict + masks to CPU)", `${fmtMs(t.yolo_ms)}${bd.inference !== undefined ? `  [Ultralytics: pre ${fmtMs(bd.preprocess)}, net ${fmtMs(bd.inference)}, post ${fmtMs(bd.postprocess)}]` : ""}`],
    [compare ? "Cropping + classifier preprocessing" : "Cropping + masks", fmtMs(t.crop_preprocess_ms)],
    ...(compare
      ? [["MobileNet classification (all segments)", fmtMs(t.classify_ms)], ["MobileNet per classified segment", fmtMs(t.classify_ms_per_segment)]]
      : [["MobileNet", "not run in YOLO-only mode"]]),
    [compare ? "Pipeline inference (YOLO + crop + MobileNet)" : "Pipeline inference (YOLO + crop)", fmtMs(t.inference_ms)],
    ["Total wall clock (adds decode, overlay render, JPEG encode)", fmtMs(t.total_ms)],
  ];
  return h("table", {}, h("tbody", {}, rows.map(([k, v]) => h("tr", {}, h("th", {}, k), h("td", {}, v)))));
}

/* ------------------------------------------------------------------ summary */
function yoloSummary(s) {
  const y = s.yolo, hh = s.head_to_head;
  if (!y.segments_with_yolo_identity) return null;
  return h("div", { class: "yolo-sum" },
    h("h4", {}, "YOLO class accuracy by YOLO confidence (manual verdicts)"),
    h("table", {},
      h("thead", {}, h("tr", {}, ["YOLO confidence", "judged", "correct", "accuracy"].map((t) => h("th", {}, t)))),
      h("tbody", {}, y.accuracy_by_confidence.map((b) => h("tr", {},
        h("td", {}, b.yolo_confidence), h("td", {}, String(b.evaluated)), h("td", {}, String(b.correct)), h("td", {}, fmtPct(b.accuracy)))))),
    hh.segments_judged_for_both
      ? h("p", { class: "info" }, `Head-to-head on the same ${hh.segments_judged_for_both} segments judged for both models: YOLO ${fmtPct(hh.yolo_accuracy)} vs MobileNet ${fmtPct(hh.mobilenet_accuracy)} ` +
          `(both right ${hh.both_correct}, only YOLO ${hh.yolo_only_correct}, only MobileNet ${hh.mobilenet_only_correct}, neither ${hh.both_incorrect}).`)
      : h("p", { class: "info" }, "Head-to-head appears once segments of a YOLO + MobileNet result are judged for both models."));
}

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
        stat(`${s.images_by_mode.yolo_only} / ${s.images_by_mode.compare}`, "images: YOLO-only / YOLO + MobileNet"),
        stat(s.yolo.segments_with_yolo_identity, `segments with a YOLO identity (${c.segments_skipped} skipped)`),
        stat(fmtPct(s.yolo.top1_accuracy), `YOLO class accuracy, n=${s.yolo.top1_denominator} (${s.yolo.ambiguous_or_not_evaluable} ambiguous, ${s.yolo.not_yet_evaluated} pending)`),
        stat(c.segments_classified, "segments classified by MobileNet"),
        stat(fmtPct(s.top1_accuracy), `MobileNet top-1 accuracy, n=${s.top1_denominator}`),
        stat(fmtPct(s.top5_accuracy), `MobileNet top-5 accuracy, n=${s.top5_denominator} (${s.top5_excluded_segments} excluded: no in-taxonomy label)`),
        stat(fmtMs(s.timing.avg_inference_ms), "avg pipeline inference / image"),
        stat(fmtMs(s.timing.avg_total_pipeline_ms), "avg total wall clock / image"),
        stat(fmtMs(s.timing.avg_classification_ms_per_segment), "avg MobileNet time / segment")),
      yoloSummary(s),
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


/* ---- food identity confirmation + USDA nutrition. Models only *suggest*; nothing counts until the user picks. ---- */
const NUT_ROWS = [["energy_kcal", "Energy", "kcal"], ["protein_g", "Protein", "g"], ["carbs_g", "Carbohydrate", "g"], ["fat_g", "Fat", "g"]];
const fmtNut = (v) => (v === null || v === undefined ? "n/a" : String(v));

function nutritionTable(n) {
  if (!n.per_100g) return null;
  const head = ["Nutrient", "per 100 g", n.serving ? `serving (${n.weight_g} g)` : null].filter(Boolean);
  return h("table", { class: "nut" }, h("thead", {}, h("tr", {}, head.map((t) => h("th", {}, t)))),
    h("tbody", {}, NUT_ROWS.map(([k, name, unit]) => h("tr", {}, h("td", {}, name), h("td", {}, `${fmtNut(n.per_100g[k])} ${unit}`),
      n.serving ? h("td", {}, `${fmtNut(n.serving[k])} ${unit}`) : null))));
}

function regionsPanel(res) {
  const wrap = h("div", { class: "regions" });
  const meal = h("div", { class: "meal", role: "status" });
  const base = (r) => `/api/results/${res.result_id}/regions/${r.region_id}`;
  const put = (r, path, body) => api(`${base(r)}/${path}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

  async function refreshMeal() {
    try {
      const m = await api(`/api/results/${res.result_id}/nutrition`);
      const t = m.totals;
      meal.textContent = t ? `Meal total from ${m.included_regions.length} confirmed region(s) with weight: ${fmtNut(t.energy_kcal)} kcal · P ${fmtNut(t.protein_g)} g · C ${fmtNut(t.carbs_g)} g · F ${fmtNut(t.fat_g)} g`
                           : "Meal total: confirm a food, choose its USDA record and enter a weight.";
      if (m.excluded_regions.length) meal.textContent += `  (not counted: ${m.excluded_regions.map((e) => `region ${e.region_id} – ${e.reason}`).join("; ")})`;
    } catch (e) { meal.textContent = e.message; }
  }

  function renderRegion(r, idx, holder) {
    const msg = h("span", { class: "fb-msg", role: "status" });
    const apply = (fresh) => { res.regions[idx] = fresh; holder.replaceChildren(renderRegion(fresh, idx, holder)); refreshMeal(); wrap.dispatchEvent(new CustomEvent("regions-changed", { bubbles: true })); };
    const run = async (fn) => { try { apply(await fn()); } catch (e) { msg.textContent = e.message; msg.classList.add("err"); } };
    const confirmed = r.status === "confirmed";
    const n = r.nutrition;

    const cands = h("div", { class: "cands" }, r.candidates.map((c) => h("button", {
      class: "btn" + (confirmed && r.identity.label === c.label && r.identity.taxonomy === c.taxonomy ? " active" : ""), type: "button",
      title: c.source === "mobilenet" ? c.note : "YOLO ingredient class",
      onclick: () => run(() => put(r, "identity", { label: c.label, taxonomy: c.taxonomy })),
    }, `${c.label} · ${c.source === "yolo" ? "YOLO" : "dish"} ${c.confidence.toFixed(2)} · ${c.usda_mapped ? "USDA ✓" : "no USDA"}`)));

    const custom = h("input", { type: "text", list: "yolo-classes", placeholder: "or type the correct food", autocomplete: "off", "aria-label": `Correct food for region ${r.region_id}` });
    const useCustom = () => {
      const v = custom.value.trim(); if (!v) return;
      const known = state.yoloClasses?.some((c) => c.toLowerCase() === v.toLowerCase());
      run(() => put(r, "identity", { label: v, taxonomy: known ? "foodseg103" : "custom" }));
    };
    custom.addEventListener("keydown", (e) => { if (e.key === "Enter") useCustom(); });

    let usda = null;
    if (confirmed && n.status === "needs_usda_selection") {
      usda = h("div", { class: "usda-pick" }, h("p", {}, n.message), n.usda_candidates.map((c) => h("button", { class: "btn", type: "button",
        onclick: () => run(() => put(r, "identity", { label: r.identity.label, taxonomy: r.identity.taxonomy, fdc_id: c.fdc_id })) },
        `${c.preparation}: ${c.description} (FDC ${c.fdc_id})`)));
    }
    const weight = h("input", { type: "number", min: "1", max: "5000", step: "any", placeholder: "weight (g)", value: r.nutrition.weight_g ?? "", "aria-label": `Portion weight in grams for region ${r.region_id}` });
    const setW = () => run(() => put(r, "portion", { weight_g: weight.value === "" ? null : Number(weight.value) }));
    weight.addEventListener("keydown", (e) => { if (e.key === "Enter") setW(); });

    let detail = null;
    if (confirmed) {
      detail = h("div", { class: "nut-box" },
        n.usda_record ? h("p", {}, `USDA: ${n.usda_record.description} · ${n.usda_record.data_source} · ${n.usda_record.preparation} · FDC ${n.usda_record.fdc_id}`) : h("p", { class: "note" }, n.message || n.status),
        nutritionTable(n),
        n.usda_record ? h("div", { class: "fb" }, weight, h("button", { class: "btn", type: "button", onclick: setW }, "Calculate serving"), n.serving_message ? h("span", { class: "meta" }, n.serving_message) : null) : null);
    }
    const dup = r.duplicate_segment_ids.length ? ` (+ duplicate segment${r.duplicate_segment_ids.length > 1 ? "s" : ""} ${r.duplicate_segment_ids.map((i) => "#" + i).join(", ")} counted once)` : "";
    return h("div", { class: `region ${r.status}` },
      h("h5", {}, `Region ${r.region_id} · ${r.source === "manual" ? "user-drawn outline (not a YOLO mask)" : `segment #${r.primary_segment_id}${dup}`}${r.excluded ? " · EXCLUDED (not food)" : ""}`,
        h("span", { class: `badge ${confirmed ? "compare" : "yolo_only"}` }, confirmed ? `confirmed: ${r.identity.label}` : "needs confirmation")),
      cands, h("div", { class: "fb" }, custom, h("button", { class: "btn", type: "button", onclick: useCustom }, confirmed ? "Correct to this" : "Confirm this"),
        confirmed ? h("button", { class: "link", type: "button", onclick: () => run(() => api(`${base(r)}/identity`, { method: "DELETE" })) }, "un-confirm") : null, msg),
      usda, detail);
  }

  res.regions.forEach((r, i) => { const holder = h("div", {}); holder.append(renderRegion(r, i, holder)); wrap.append(holder); });
  wrap.append(meal);
  refreshMeal();
  return wrap;
}
