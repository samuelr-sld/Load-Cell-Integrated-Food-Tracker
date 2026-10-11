"use strict";
/* Plate analysis UI: plate calibration, food-area distribution, ONE total weight -> estimated weights + USDA nutrition,
   meal finalization and the saved-meal list. Loaded after app.js and reuses its helpers (h, $, api).
   Every number shown here comes from GET /api/results/{id}/plate-analysis; the pie chart only draws what the backend counted. */

const SVG_NS = "http://www.w3.org/2000/svg";
function sv(tag, attrs = {}, ...kids) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) el.setAttribute(k, v);
  for (const kid of kids.flat()) if (kid) el.append(kid);
  return el;
}
const fmtG = (v, d = 1) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));
const rgb = (c) => `rgb(${c.join(",")})`;
const NUT_KEYS = ["energy_kcal", "protein_g", "carbs_g", "fat_g"];

/* Pie of the food-mask area distribution. Angles come from the backend's attributed_area_px (counted from mask pixels). */
function areaPie(distribution) {
  const total = distribution.reduce((a, d) => a + d.attributed_area_px, 0);
  if (!total) return h("p", { class: "empty" }, "No food-mask area on the plate yet.");
  const svg = sv("svg", { viewBox: "-1.05 -1.05 2.1 2.1", class: "pie", role: "img", "aria-label": "Food area distribution by region" });
  let a0 = -Math.PI / 2;
  for (const d of distribution) {
    const frac = d.attributed_area_px / total, a1 = a0 + frac * 2 * Math.PI;
    const title = sv("title", {});
    title.textContent = `Region ${d.region_id}${d.label ? ` (${d.label})` : ""}: ${d.food_area_pct.toFixed(1)}% of food area, ${Math.round(d.attributed_area_px)} px`;
    const fill = rgb(d.color_rgb);
    if (frac > 0.9999) svg.append(sv("circle", { r: 1, fill, class: "slice", "data-region": d.region_id }, title));
    else {
      const [x0, y0, x1, y1] = [Math.cos(a0), Math.sin(a0), Math.cos(a1), Math.sin(a1)];
      svg.append(sv("path", { d: `M0 0 L${x0} ${y0} A1 1 0 ${frac > 0.5 ? 1 : 0} 1 ${x1} ${y1} Z`, fill, class: "slice", "data-region": d.region_id }, title));
    }
    a0 = a1;
  }
  const legend = h("ul", { class: "legend" }, distribution.map((d) => h("li", {}, h("span", { class: "swatch", style: `background:${rgb(d.color_rgb)}` }),
    `Region ${d.region_id} ${d.label || "(unconfirmed)"}: ${d.food_area_pct.toFixed(1)}% · ${Math.round(d.attributed_area_px)} px`)));
  return h("div", { class: "pie-wrap" }, svg, legend);
}

/* onStructureChange(): the card must re-render (a region was added/removed/excluded, so the identity list changes). */
function plateSection(res, onStructureChange) {
  const rid = res.result_id, W = res.image.width, H = res.image.height;
  const send = (method, path, body) => api(`/api/results/${rid}/${path}`, { method, headers: { "Content-Type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body) });
  const put = (path, body) => send("PUT", path, body);
  let data = null, mode = null, points = [];
  const msg = h("div", { class: "fb-msg", role: "status" });
  const say = (t, err) => { msg.textContent = t; msg.classList.toggle("err", !!err); };
  const guard = (fn) => async () => { try { await fn(); } catch (e) { say(e.message, true); } };

  /* --- photo with YOLO masks and the plate boundary --- */
  const img = h("img", { src: res.image_urls.overlay, alt: "Meal photo with YOLO masks", width: W, height: H, class: "stage-img" });
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, class: "stage-svg", preserveAspectRatio: "none" });
  const stage = h("div", { class: "stage" }, img, svg);
  const masksBox = h("input", { type: "checkbox", checked: true, onchange: () => { img.src = masksBox.checked ? res.image_urls.overlay : res.image_urls.original; } });
  function drawOverlay() {
    svg.replaceChildren();
    const p = data?.plate;
    if (p) svg.append(sv("circle", { cx: p.center_x, cy: p.center_y, r: p.radius_px, class: "plate-ring", "data-testid": "plate-ring" }),
      sv("circle", { cx: p.center_x, cy: p.center_y, r: 3, class: "plate-ring-dot" }));
    if (points.length) {
      if (mode === "outline") svg.append(sv("polyline", { points: points.map((q) => q.join(",")).join(" "), class: "draft-line" }));
      svg.append(...points.map((q) => sv("circle", { cx: q[0], cy: q[1], r: 4, class: "draft-pt" })));
    }
  }
  svg.addEventListener("click", (e) => {
    if (!mode) return;
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const q = pt.matrixTransform(svg.getScreenCTM().inverse());
    points.push([Math.round(q.x * 10) / 10, Math.round(q.y * 10) / 10]);
    if (mode === "rim" && points.length === 3) finishRim(); else { drawOverlay(); updateModeUi(); }
  });

  /* --- calibration + manual-add controls --- */
  const num = (ph, w = 70) => h("input", { type: "number", step: "any", placeholder: ph, "aria-label": ph, style: `width:${w}px` });
  const cx = num("centre x"), cy = num("centre y"), rr = num("radius px"), dia = num("plate Ø mm", 90);
  const rimBtn = h("button", { class: "btn", type: "button", "data-testid": "rim-mode", onclick: () => startMode("rim") }, "Click 3 points on the plate rim");
  const autoBtn = h("button", { class: "btn", type: "button", "data-testid": "auto-plate", onclick: guard(autoPlate) }, "Auto-detect plate");
  const outlineBtn = h("button", { class: "btn", type: "button", "data-testid": "outline-mode", onclick: () => startMode("outline") }, "Add a missed food (draw outline)");
  const finishBtn = h("button", { class: "btn", type: "button", onclick: guard(finishOutline) }, "Finish outline");
  const cancelBtn = h("button", { class: "link", type: "button", onclick: () => { mode = null; points = []; say(""); drawOverlay(); updateModeUi(); } }, "cancel");
  function updateModeUi() {
    finishBtn.hidden = mode !== "outline";
    cancelBtn.hidden = !mode;
    if (mode === "rim") say(`Click the plate rim: ${points.length}/3 points`);
    else if (mode === "outline") say(`Click around the food YOLO missed: ${points.length} point(s) (at least 3), then "Finish outline"`);
  }
  function startMode(m) { mode = m; points = []; drawOverlay(); updateModeUi(); }
  const withDia = () => (dia.value ? { plate_diameter_mm: Number(dia.value) } : {});
  async function finishRim() {
    const rim = points; mode = null; points = [];
    try { await put("plate", { rim_points: rim, ...withDia() }); say("Plate calibrated from 3 rim points."); await load(); }
    catch (e) { say(e.message, true); drawOverlay(); updateModeUi(); }
  }
  async function finishOutline() {
    if (points.length < 3) throw new Error("Click at least 3 points around the food first");
    await send("POST", "regions/manual", { polygon: points });
    mode = null; points = [];
    await onStructureChange();
  }
  const applyNumeric = guard(async () => {
    await put("plate", { center_x: Number(cx.value), center_y: Number(cy.value), radius_px: Number(rr.value), image_width: W, image_height: H, ...withDia() });
    say("Plate set."); await load();
  });
  async function autoPlate() {
    const r = await send("POST", "plate/auto", state.meta[rid]?.cameraId ? { camera_id: state.meta[rid].cameraId } : {});
    say(r.message, !r.applied);
    await load();
  }
  const clearPlate = guard(async () => { await send("DELETE", "plate"); await load(); });

  /* --- one total weight (simulation of the future load cell) --- */
  const wIn = h("input", { type: "number", min: "0.1", max: "5000", step: "0.1", placeholder: "total food weight (g)", "aria-label": "Total meal weight in grams", id: `w-${rid}`, style: "width:170px" });
  const setW = guard(async () => { await put("total-weight", { net_weight_g: Number(wIn.value), source: "manual_simulation" }); say("Total weight set."); await load(); });
  const clearW = guard(async () => { await send("DELETE", "total-weight"); wIn.value = ""; await load(); });
  wIn.addEventListener("keydown", (e) => { if (e.key === "Enter") setW(); });

  /* --- finalize --- */
  const incompleteBox = h("input", { type: "checkbox" });
  const finalizeMsg = h("div", { class: "fb-msg", role: "status", "data-testid": "finalize-msg" });
  const finalize = async () => {
    try {
      const r = await send("POST", "meal", { accept_incomplete: incompleteBox.checked });
      finalizeMsg.classList.remove("err");
      finalizeMsg.textContent = r.duplicate ? `Already saved as ${r.meal.meal_id} (no duplicate created).` : `Saved ${r.meal.meal_id} as ${r.meal.completeness_status}.`;
      refreshMeals();
    } catch (e) { finalizeMsg.textContent = e.message; finalizeMsg.classList.add("err"); }
  };

  const status = h("div", { class: "statusline", "data-testid": "plate-status" });
  const pieHolder = h("div", {});
  const tableHolder = h("div", { class: "tbl-scroll" });
  const excludedHolder = h("div", { class: "fb" });
  const warnHolder = h("div", {});
  const totalsHolder = h("div", { class: "meal", role: "status", "data-testid": "plate-totals" });

  const setCount = async (r, val) => { try { await put(`regions/${r.region_id}/entity-count`, { user_entity_count: val === "" ? null : Number(val) }); } catch (e) { say(e.message, true); } await load(); };
  const setExcluded = async (id, on) => { try { await put(`regions/${id}/excluded`, { excluded: on }); await onStructureChange(); } catch (e) { say(e.message, true); } };
  const delManual = async (r) => { try { await send("DELETE", `regions/${r.region_id}`); await onStructureChange(); } catch (e) { say(e.message, true); } };

  function render() {
    const a = data;
    const badge = (t, cls) => h("span", { class: `badge ${cls}`, "data-state": t }, t);
    status.replaceChildren(...[
      badge(a.allocation_status || "awaiting_inputs", a.allocation_status === "allocation_complete" ? "yolo_only" : a.allocation_status === "allocation_rejected" ? "bad" : "compare"),
      a.plate_source ? badge(`plate: ${a.plate_source}`, a.plate_source === "manual" ? "compare" : "yolo_only") : null,
      badge(a.validation_status, "compare"),
      badge(`meal_${a.completeness_status}`, a.completeness_status === "complete" ? "yolo_only" : "compare"),
      a.food_area ? h("span", { class: "meta" }, `food covers ${fmtG(a.food_area.plate_coverage_pct)}% of the plate (${a.food_area.unique_food_area_px} unique mask px, overlaps counted once)`) : null,
      a.missing_inputs.length ? h("span", { class: "meta" }, `missing: ${a.missing_inputs.join(", ")}`) : null].filter(Boolean));
    if (a.plate && document.activeElement !== cx && document.activeElement !== cy && document.activeElement !== rr) {
      cx.value = fmtG(a.plate.center_x); cy.value = fmtG(a.plate.center_y); rr.value = fmtG(a.plate.radius_px);
    }
    if (a.plate?.plate_diameter_mm && !dia.value) dia.value = a.plate.plate_diameter_mm;
    if (a.total_weight && document.activeElement !== wIn) wIn.value = a.total_weight.net_weight_g;
    pieHolder.replaceChildren(a.food_area ? areaPie(a.food_area.distribution) : h("p", { class: "empty" }, "Calibrate the plate to measure the food area."));

    const head = ["Region", "Food identity", "Detected segments", "Your item count", "Not food", "Mask area px", "% of food area", "Est. weight g", "g / item", "kcal", "Protein g", "Carbs g", "Fat g", "USDA record"];
    const rows = a.regions.map((r) => {
      const n = r.nutrition, al = r.allocation;
      const count = h("input", { type: "number", min: "1", max: "100", step: "1", value: r.entity_count.user_count ?? "", "aria-label": `Item count for region ${r.region_id}`, style: "width:60px",
        onchange: (e) => setCount(r, e.target.value) });
      const unres = al?.status === "unresolved";
      return h("tr", { "data-region": r.region_id },
        h("td", {}, h("span", { class: "swatch", style: `background:${rgb(r.color_rgb)}` }), `#${r.region_id}${r.source === "manual" ? " (drawn)" : ""}`,
          r.source === "manual" ? h("button", { class: "link", type: "button", onclick: () => delManual(r) }, " delete") : null),
        h("td", { class: "label" }, r.label ? r.label : h("em", {}, "needs confirmation (section E)")),
        h("td", {}, String(r.entity_count.detected_segments)), h("td", {}, count),
        h("td", {}, h("input", { type: "checkbox", "aria-label": `Region ${r.region_id} is not food`, onchange: (e) => setExcluded(r.region_id, e.target.checked) })),
        h("td", {}, r.area ? String(Math.round(r.area.attributed_area_px)) : "–"), h("td", {}, r.area ? fmtG(r.area.food_area_pct) : "–"),
        h("td", { class: "w" }, al?.status === "allocated" ? fmtG(al.estimated_weight_g) : unres ? h("span", { class: "unres", title: "held back until this region is confirmed" }, `${fmtG(al.unresolved_weight_g)} unresolved`) : "–"),
        h("td", {}, fmtG(al?.weight_per_entity_g)),
        ...NUT_KEYS.map((k) => h("td", {}, n ? fmtG(n[k], 1) : "–")),
        h("td", {}, r.usda ? `${r.usda.description} (FDC ${r.usda.fdc_id})` : r.usda_message || "–"));
    });
    tableHolder.replaceChildren(a.regions.length
      ? h("table", { class: "plate-table" }, h("thead", {}, h("tr", {}, head.map((t) => h("th", {}, t)))), h("tbody", {}, rows))
      : h("p", { class: "empty" }, "No active food regions."));

    const out = res.regions.filter((r) => r.excluded);
    excludedHolder.replaceChildren(...(out.length ? [h("span", { class: "meta" }, "Excluded as not food:"),
      ...out.map((r) => h("button", { class: "btn", type: "button", onclick: () => setExcluded(r.region_id, false) }, `restore region ${r.region_id}`))] : []));

    const t = a.nutrition.totals, al = a.allocation;
    totalsHolder.textContent = al && al.status !== "rejected"
      ? `Estimated meal: ${fmtG(al.allocated_weight_g)} g allocated of ${fmtG(a.total_weight.net_weight_g)} g` +
        (al.unresolved_weight_g ? ` (${fmtG(al.unresolved_weight_g)} g unresolved)` : "") +
        (t ? ` · ${fmtG(t.energy_kcal)} kcal · P ${fmtG(t.protein_g)} g · C ${fmtG(t.carbs_g)} g · F ${fmtG(t.fat_g)} g` : " · no nutrition yet") +
        (al.status === "complete" ? "" : "  — PARTIAL: covers confirmed foods only")
      : al ? "Allocation rejected: see the warnings below. No weights or nutrition are shown."
        : "Estimated meal totals appear once the plate, the total weight and the food identities are set.";

    warnHolder.replaceChildren(h("ul", { class: "warns" }, a.warnings.map((w) =>
      h("li", { class: `w-${w.severity}`, "data-code": w.code }, h("b", {}, w.severity === "info" ? "Note: " : w.severity === "blocking" ? "Blocking: " : "Warning: "), w.message))));
  }

  async function load() {
    data = await api(`/api/results/${rid}/plate-analysis`);
    drawOverlay(); render();
  }
  const element = h("div", { class: "plate" },
    h("p", { class: "info" }, "Food area is counted from the actual segmentation-mask pixels inside the plate circle (overlaps counted once). It is a 2-D footprint, not volume or mass. " +
      "The total weight below is a manual simulation of the future load cell."),
    h("div", { class: "plate-grid" },
      h("div", {}, stage,
        h("div", { class: "fb" }, h("label", {}, masksBox, " show YOLO masks")),
        h("div", { class: "fb" }, autoBtn, rimBtn, cx, cy, rr, dia, h("button", { class: "btn", type: "button", onclick: applyNumeric }, "Set plate"), h("button", { class: "link", type: "button", onclick: clearPlate }, "clear plate")),
        h("div", { class: "fb" }, outlineBtn, finishBtn, cancelBtn), msg),
      h("div", {}, h("h4", {}, "Food area distribution (from mask pixels)"), pieHolder)),
    status,
    h("div", { class: "fb" }, h("label", { for: `w-${rid}` }, "Total meal weight:"), wIn, h("button", { class: "btn", type: "button", "data-testid": "set-weight", onclick: setW }, "Set weight"),
      h("button", { class: "link", type: "button", onclick: clearW }, "clear"), h("span", { class: "hint" }, "source: manual_simulation (not a live load-cell reading)")),
    tableHolder, excludedHolder, totalsHolder, warnHolder,
    h("div", { class: "fb" }, h("button", { class: "btn", type: "button", "data-testid": "finalize", onclick: finalize }, "Finalize and save meal"),
      h("label", {}, incompleteBox, " save even if incomplete (stored as incomplete)"), finalizeMsg));
  updateModeUi();
  if (state.meta[rid]?.note) say(state.meta[rid].note, true);
  return { element, load };
}

/* Replace a result card with a freshly fetched one (used when the set of regions changes). */
async function refreshCard(resultId) {
  const old = document.querySelector(`.card[data-result-id="${resultId}"]`);
  const fresh = await api(`/api/results/${resultId}`);
  old?.replaceWith(renderCard(fresh));
}

/* ---- saved meals (SQLite-backed, survive restarts) ---- */
async function refreshMeals() {
  const el = $("#meals");
  if (!el) return;
  try {
    const { meals } = await api("/api/meals");
    el.replaceChildren(meals.length ? h("table", {},
      h("thead", {}, h("tr", {}, ["meal", "saved (UTC)", "status", "allocation", "total g", "foods (estimated g)", "kcal", "weight source"].map((t) => h("th", {}, t)))),
      h("tbody", {}, meals.map((m) => h("tr", { "data-meal": m.meal_id },
        h("td", { class: "mono" }, m.meal_id), h("td", {}, m.created_at), h("td", {}, m.completeness_status), h("td", {}, m.allocation_status),
        h("td", {}, fmtG(m.total_weight_g)),
        h("td", {}, m.items.map((i) => `${i.label} ${fmtG(i.estimated_weight_g)} g${i.user_entity_count ? ` (${i.user_entity_count}×)` : ""}`).join(", ")),
        h("td", {}, m.totals ? fmtG(m.totals.energy_kcal) : "–"), h("td", {}, m.weight_source))))) : h("p", { class: "empty" }, "No meals saved yet."));
  } catch (e) { el.textContent = `Saved meals unavailable: ${e.message}`; }
}
refreshMeals();
