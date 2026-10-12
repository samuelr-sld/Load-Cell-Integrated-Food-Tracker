"use strict";
/* Camera and plate detection panel. Loaded after app.js and plate.js and reuses their helpers (h, sv, $, api, enqueue, state).
   Everything shown here comes from /api/camera/*: the browser never fakes a camera result, a detection or a calibration.
   A captured frame is analysed through the ordinary upload queue (enqueue), i.e. the same /api/inference path as a file upload.
   The camera is only opened when a button here is pressed; this page never opens it on load. */

(function cameraPanel() {
  const root = $("#camera-panel");
  if (!root) return;
  const json = (method, path, body) => api(path, { method, headers: { "Content-Type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body) });

  let cap = null;            // latest capture {capture_id, camera_id, width, height, image_url, filename}
  let circle = null;         // the circle currently drawn: {center_x, center_y, radius_px, kind: detected|uncertain|saved|manual, method}
  let points = [];           // manual rim clicks (capture pixels)
  let manualMode = false;
  let selected = null;       // camera status {source, camera_id}
  let hasCal = false;        // a calibration is saved for the selected camera

  const msg = h("div", { class: "fb-msg", role: "status", "data-testid": "cam-msg" });
  const say = (t, err) => { msg.textContent = t; msg.classList.toggle("err", !!err); };
  const guard = (fn) => async () => { try { await fn(); } catch (e) { say(e.message, true); } };
  const busy = async (btn, fn) => { btn.disabled = true; try { await fn(); } finally { btn.disabled = false; sync(); } };

  /* ---- controls ---- */
  const camSel = h("select", { "aria-label": "Camera", "data-testid": "cam-select", onchange: () => select(camSel.value) });
  const modeSel = h("select", { "aria-label": "Plate localization mode", "data-testid": "mode-select", onchange: guard(async () => { await json("PUT", "/api/camera/localization-mode", { mode: modeSel.value }); say(`Localization mode: ${modeSel.value}`); }) });
  const dia = h("input", { type: "number", step: "any", min: "1", placeholder: "plate Ø mm (optional)", "aria-label": "Plate diameter in millimetres", style: "width:150px" });
  const btn = (label, onclick, testid) => h("button", { class: "btn", type: "button", "data-testid": testid, onclick }, label);
  const refreshBtn = btn("Refresh camera list", () => busy(refreshBtn, guard(refreshDevices)), "cam-refresh");
  const testBtn = btn("Test camera connection", () => busy(testBtn, guard(testCamera)), "cam-test");
  const startPrevBtn = btn("Start preview", () => busy(startPrevBtn, guard(startPreview)), "cam-preview-start");
  const stopPrevBtn = btn("Stop preview", () => busy(stopPrevBtn, guard(stopPreview)), "cam-preview-stop");
  const captureBtn = btn("Capture frame", () => busy(captureBtn, guard(capture)), "cam-capture");
  const captureAnalyzeBtn = btn("Capture & analyze", () => busy(captureAnalyzeBtn, guard(async () => { await capture(); await analyze(); })), "cam-capture-analyze");
  const detectBtn = btn("Auto detect plate", () => busy(detectBtn, guard(detect)), "cam-detect");
  const saveBtn = btn("Save circle as calibration", () => busy(saveBtn, guard(saveCalibration)), "cam-save");
  const rimBtn = btn("Click 3 rim points (manual)", () => { manualMode = true; points = []; draw(); say("Click 3 points on the plate rim in the preview: 0/3"); }, "cam-manual");
  const clearBtn = btn("Clear calibration", () => busy(clearBtn, guard(clearCalibration)), "cam-clear");
  const analyzeBtn = btn("Analyze captured image", () => busy(analyzeBtn, guard(analyze)), "cam-analyze");
  const autoBox = h("input", { type: "checkbox", checked: true, "data-testid": "auto-localize", onchange: () => { state.autoLocalize = autoBox.checked; } });
  const calText = h("span", { class: "meta", "data-testid": "cam-calibration" });
  const statusLine = h("div", { class: "statusline", "data-testid": "cam-status" });
  const evidence = h("div", { class: "hint" });

  /* ---- live preview (MJPEG from /api/camera/preview/stream) ---- */
  let previewOn = false, pollTimer = null;
  const liveImg = h("img", { alt: "Live camera preview", class: "stage-img", "data-testid": "cam-live-img", onerror: () => { if (previewOn) pollPreview(); } });
  const liveBox = h("div", { class: "stage cam-stage", hidden: true }, liveImg);
  const liveStatus = h("div", { class: "statusline", "data-testid": "cam-live-status" });
  function showLive(st) {
    const ok = st.active && st.state === "running";
    const cls = ok ? "yolo_only" : st.state === "error" || st.state === "stalled" ? "bad" : "compare";
    liveStatus.replaceChildren(h("span", { class: `badge ${cls}`, "data-state": st.state }, ok ? "● live" : `preview: ${st.state}`),
      st.camera_id ? h("span", { class: "meta" }, `${st.camera_id}${st.width ? ` · camera ${st.width}×${st.height}` : ""}${st.active ? ` · stream ${st.preview_width}px @ ${st.measured_fps} fps (target ${st.target_fps})` : ""}`) : null,
      st.error ? h("span", { class: "meta w-warning" }, st.error.message) : null);
  }
  async function pollPreview() {
    let st;
    try { st = await api("/api/camera/preview/status"); } catch (e) { return say(`Preview status unavailable: ${e.message}`, true); }
    showLive(st);
    if (!previewOn && st.active) {                                   // page reloaded while the server-side preview was still alive: reattach
      previewOn = true; liveBox.hidden = false; liveImg.src = `/api/camera/preview/stream?t=${Date.now()}`;
      clearInterval(pollTimer); pollTimer = setInterval(pollPreview, 1500); sync();
    }
    if (previewOn && !st.active)endPreviewUi(st.error ? `Preview ended: ${st.error.message}` : `Preview ended (${st.stop_reason || "stopped"}).`, !!st.error);
  }
  function endPreviewUi(text, isErr) {
    previewOn = false; clearInterval(pollTimer); pollTimer = null;
    liveImg.removeAttribute("src"); liveBox.hidden = true; sync();
    if (text) say(text, isErr);
  }
  async function startPreview() {
    say("Starting preview (the camera is opened once and kept open)…");
    const st = await json("POST", "/api/camera/preview/start", {});
    previewOn = true; liveBox.hidden = false; showLive(st);
    liveImg.src = `${st.stream_url}?t=${Date.now()}`;
    clearInterval(pollTimer); pollTimer = setInterval(pollPreview, 1500);
    say(`Live preview on ${st.camera_id}. Capture frame / Capture & analyze use this same open camera.`); sync();
  }
  async function stopPreview() {
    await json("POST", "/api/camera/preview/stop", {}).catch((e) => say(e.message, true));
    endPreviewUi("Preview stopped; the camera was released."); showLive({ active: false, state: "stopped" });
  }
  // closing/leaving the page: tell the server right away (it also releases the camera by itself when the stream disconnects)
  window.addEventListener("pagehide", () => { if (previewOn) navigator.sendBeacon("/api/camera/preview/stop"); });

  /* ---- preview with overlay ---- */
  const img = h("img", { alt: "Captured camera frame", class: "stage-img", hidden: true });
  const svg = sv("svg", { class: "stage-svg", preserveAspectRatio: "none" });
  const stage = h("div", { class: "stage cam-stage", hidden: true }, img, svg);
  svg.addEventListener("click", (e) => {
    if (!manualMode || !cap) return;
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const q = pt.matrixTransform(svg.getScreenCTM().inverse());
    points.push([Math.round(q.x * 10) / 10, Math.round(q.y * 10) / 10]);
    if (points.length === 3) guard(finishManual)(); else { draw(); say(`Click points on the plate rim: ${points.length}/3`); }
  });
  function draw() {
    svg.replaceChildren();
    if (!cap) return;
    svg.setAttribute("viewBox", `0 0 ${cap.width} ${cap.height}`);
    if (circle) {
      const cls = { detected: "plate-ring", saved: "plate-ring", manual: "plate-ring", uncertain: "plate-ring uncertain" }[circle.kind] || "plate-ring";
      svg.append(sv("circle", { cx: circle.center_x, cy: circle.center_y, r: circle.radius_px, class: cls, "data-testid": "cam-ring", "data-kind": circle.kind }),
        sv("circle", { cx: circle.center_x, cy: circle.center_y, r: 3, class: "plate-ring-dot" }));
    }
    svg.append(...points.map((q) => sv("circle", { cx: q[0], cy: q[1], r: 4, class: "draft-pt" })));
  }
  function sync() {
    const have = !!cap;
    detectBtn.disabled = !have; analyzeBtn.disabled = !have; rimBtn.disabled = !have;
    startPrevBtn.disabled = previewOn; stopPrevBtn.disabled = !previewOn;
    saveBtn.disabled = !circle || !have;
    clearBtn.disabled = !hasCal;
    stage.hidden = !have; img.hidden = !have;
  }
  function badge(text, cls) { return h("span", { class: `badge ${cls}`, "data-state": text }, text); }
  function showStatus(label, state, extra = []) {
    const cls = { localized: "yolo_only", detected: "yolo_only", needs_manual: "bad", failed: "bad", uncertain: "compare" }[state] || "compare";
    statusLine.replaceChildren(badge(label, cls), ...extra.filter(Boolean));
  }
  function showEvidence(ev, warnings, candidates) {
    evidence.replaceChildren();
    if (ev) evidence.append(`edge support ${(ev.edge_support * 100).toFixed(0)}% · rim arc ${ev.arc_span_deg}° · fit rms ${ev.fit_rms_px}px · visible ${(ev.visible_fraction * 100).toFixed(0)}% · evidence score ${ev.evidence_score} (heuristic, not a probability)`);
    for (const w of warnings || []) evidence.append(h("div", { class: "w-warning" }, "Warning: " + w));
    if (candidates && candidates.length > 1) evidence.append(h("div", {}, `${candidates.length} candidate circle(s) considered.`));
  }

  /* ---- actions ---- */
  async function refreshStatus() {
    const s = await api("/api/camera/status");
    selected = s.selected_camera;
    modeSel.replaceChildren(...s.localization_modes.map((m) => h("option", { value: m, selected: m === s.localization_mode }, m)));
    if (![...camSel.options].some((o) => o.value === String(selected.source))) camSel.append(h("option", { value: String(selected.source) }, `${selected.camera_id} (not probed)`));
    camSel.value = String(selected.source);
    const c = s.calibration;
    calText.textContent = c ? `saved calibration for ${c.camera_id}: centre (${c.center_x.toFixed(1)}, ${c.center_y.toFixed(1)}) r ${c.radius_px.toFixed(1)} px @ ${c.image_width}×${c.image_height}, ${c.method}, ${c.calibrated_at}`
      : `no saved calibration for ${selected.camera_id}`;
    hasCal = !!c; sync();
    return s;
  }
  async function refreshDevices() {
    say("Probing cameras (each is opened briefly)…");
    const r = await api("/api/camera/devices");
    camSel.replaceChildren(...r.devices.map((d) => h("option", { value: String(d.source) },
      `${d.camera_id} — ${d.in_preview ? `previewing, ${d.width}×${d.height}` : d.capture_ok ? `${d.width}×${d.height}` : `unavailable (${d.error_code})`}`)));
    say(r.available.length ? `Available: ${r.available.join(", ")}. Pick the external webcam; index 0 is often the built-in camera.` : "No camera answered. Check the cable, close other apps using the camera, or try another USB port.", !r.available.length);
    await refreshStatus();
  }
  async function select(source) {
    if (previewOn) await stopPreview();
    await json("PUT", "/api/camera/selection", { source: /^\d+$/.test(source) ? Number(source) : source }).catch((e) => say(e.message, true));
    await refreshStatus();
    cap = null; circle = null; points = []; draw(); sync(); showStatus("no capture", "needs_manual");
    say(`Selected ${selected.camera_id}. Calibration is stored per camera.`);
  }
  async function testCamera() {
    say("Testing…");
    const r = await json("POST", "/api/camera/test", {});
    say(r.capture_ok ? `${r.camera_id}: connected, frame ${r.width}×${r.height}, captured in ${r.capture_ms} ms` : `${r.camera_id}: ${r.error}`, !r.capture_ok);
  }
  async function capture() {
    say("Capturing…");
    cap = await json("POST", "/api/camera/capture", {});
    img.src = `${cap.image_url}&t=${Date.now()}`;
    circle = null; points = []; manualMode = false; draw(); sync();
    say(`Captured ${cap.width}×${cap.height} from ${cap.camera_id} (${cap.capture_ms} ms).`);
    await localize();
  }
  async function localize() {
    const r = await json("POST", "/api/camera/localize", {});
    const used = r.steps.map((s) => `${s.source}: ${s.outcome}${s.reason ? ` (${s.reason})` : ""}`).join(" → ");
    if (r.status === "localized") {
      circle = { ...r.plate, kind: r.source === "saved_calibration" ? "saved" : "detected" };
      showStatus(`plate localized: ${r.source}`, "localized", [h("span", { class: "meta" }, `mode ${r.mode}: ${used}`)]);
      showEvidence(r.detection?.evidence, r.detection?.warnings, r.detection?.candidates);
    } else {
      const sug = r.suggestion;
      circle = sug ? { ...sug, kind: "uncertain" } : null;
      showStatus("needs manual calibration", "needs_manual", [h("span", { class: "meta" }, `mode ${r.mode}: ${used}`)]);
      showEvidence(r.detection?.evidence, r.detection?.warnings, r.detection?.candidates);
      say(r.message, true);
    }
    draw(); sync();
  }
  async function detect() {
    const r = await json("POST", "/api/camera/detect-plate", {});
    circle = r.plate ? { ...r.plate, kind: r.status === "detected" ? "detected" : "uncertain", method: "automatic_opencv" } : null;
    showStatus(`detection: ${r.status}${r.reason ? ` (${r.reason})` : ""}`, r.status, [h("span", { class: "meta" }, `${r.timing_ms} ms`)]);
    showEvidence(r.evidence, r.warnings, r.candidates);
    say(r.message, r.status !== "detected");
    draw(); sync();
  }
  async function finishManual() {
    const rim = points; manualMode = false; points = [];
    const body = { rim_points: rim, image_width: cap.width, image_height: cap.height, camera_id: cap.camera_id };
    if (dia.value) body.plate_diameter_mm = Number(dia.value);
    const r = await json("PUT", "/api/camera/calibration", body);
    circle = { ...r.geometry, kind: "manual" };
    say("Calibration saved from 3 rim points."); showStatus("plate localized: manual calibration saved", "localized");
    await refreshStatus(); draw(); sync();
  }
  async function saveCalibration() {
    if (!circle) throw new Error("No circle to save: run detection or click 3 rim points first");
    const body = { camera_id: cap.camera_id, center_x: circle.center_x, center_y: circle.center_y, radius_px: circle.radius_px,
      image_width: cap.width, image_height: cap.height, method: circle.method || "manual_numeric" };
    if (dia.value) body.plate_diameter_mm = Number(dia.value);
    await json("PUT", "/api/camera/calibration", body);
    say(`Saved the circle as the calibration for ${cap.camera_id}. Later captures verify it against the image before using it.`);
    await refreshStatus();
  }
  async function clearCalibration() {
    await json("DELETE", `/api/camera/calibration?camera_id=${encodeURIComponent(selected.camera_id)}`);
    say("Calibration cleared. Re-run detection or calibrate manually.");
    await refreshStatus();
  }
  async function analyze() {
    const blob = await (await fetch(cap.image_url)).blob();
    enqueue([new File([blob], cap.filename, { type: "image/jpeg" })], { cameraId: cap.camera_id });   // the normal upload queue / inference path
    say("Sent to the normal analysis queue below.");
  }

  /* ---- layout ---- */
  root.append(
    h("p", { class: "info" }, "Pick the external camera, capture one frame and let OpenCV find the plate. A saved calibration is re-checked against every new image " +
      "(plate or camera moved ⇒ it is not used). If nothing reliable is found you are asked to click 3 rim points; no circle is ever invented."),
    h("div", { class: "fb" }, h("label", {}, "Camera: ", camSel), refreshBtn, testBtn, h("label", {}, "Localization: ", modeSel)),
    h("div", { class: "fb" }, startPrevBtn, stopPrevBtn, captureBtn, captureAnalyzeBtn),
    liveStatus, liveBox,
    h("div", { class: "fb" }, detectBtn, rimBtn, dia, saveBtn, clearBtn, calText),
    h("div", { class: "fb" }, analyzeBtn, h("label", {}, autoBox, " localize the plate automatically for new results (uploads and captures)")),
    statusLine, evidence, msg, stage);
  state.autoLocalize = true;
  sync();
  pollPreview();                       // a preview that survived a page reload is shown as stopped/live truthfully
  refreshStatus().then(() => showStatus("no capture", "needs_manual")).catch((e) => say(`Camera API unavailable: ${e.message}`, true));
})();
