"""Classical-OpenCV plate localization (no learned model): where is the circular plate rim in a top-down image?

Pipeline: grayscale -> CLAHE + blur -> Canny -> candidate circles from HoughCircles AND from contour/arc fits ->
every candidate is refined by a least-squares fit to its own rim edge points and then SCORED by the same measurable evidence:

  edge_support   fraction of the (visible) rim angles that have an edge pixel on the candidate circle whose gradient is
                 perpendicular to the rim (i.e. a real boundary, not texture), counting only runs of consecutive angles
  arc_span_deg   angular extent of that supported rim (a partially visible plate has a short arc)
  fit_rms_px     RMS distance of the supporting edge points from the fitted circle
  visible_fraction  part of the rim that lies inside the image

`evidence_score` combines them into 0..1. It is a HEURISTIC quality indicator, not a calibrated probability.

Contract: status is "detected" (safe to apply), "uncertain" (a plausible circle exists but the evidence is weak, partial or
ambiguous: show it to the user, never apply it silently) or "failed" (nothing usable; no geometry is returned).
Geometry is never invented: a number appears in `plate` only if it was measured from edges in this image.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .plate_analysis import PlateGeometry

EVIDENCE_NOTE = ("evidence_score is a heuristic measure of edge evidence along the fitted rim (support x arc x fit). "
                 "It is NOT a calibrated probability of being correct.")
ANGLES = 360


@dataclass(frozen=True)
class DetectionConfig:
    work_side: int = 640                 # detection runs on a copy whose longest side is this; results are scaled back
    min_radius_frac: float = 0.20        # plausible plate radius as a fraction of min(width, height)
    max_radius_frac: float = 0.75
    min_contrast_std: float = 6.0        # grayscale std below this = blank / washed-out image: not suitable
    accept_support: float = 0.70         # "detected" needs at least this much supported rim ...
    accept_score: float = 0.70           # ... and this evidence score ...
    accept_arc_deg: float = 150.0        # ... over at least this angular span
    min_candidate_support: float = 0.50  # weaker candidates are discarded entirely
    min_run_samples: int = 8             # supported angles count only inside runs at least this long (kills scattered texture hits)
    max_gap_samples: int = 3             # ... bridging gaps up to this long (food occluding the rim)
    ambiguity_margin: float = 0.10       # a distinct non-concentric rival within this score gap makes the result "uncertain"
    max_candidates: int = 5              # returned in diagnostics


class _Ctx:
    """Per-image precomputation shared by every candidate."""
    def __init__(self, gray: np.ndarray, edges: np.ndarray, gx: np.ndarray, gy: np.ndarray):
        self.gray, self.edges = gray, edges.astype(bool)
        self.gx, self.gy = gx, gy
        self.gmag = np.hypot(gx, gy)
        self.h, self.w = gray.shape
        a = np.deg2rad(np.arange(ANGLES) * (360.0 / ANGLES))
        self.cos, self.sin = np.cos(a), np.sin(a)


def _fit_circle(x: np.ndarray, y: np.ndarray) -> Optional[tuple[float, float, float]]:
    """Algebraic (Kasa) least-squares circle fit; None when the points are (near) collinear."""
    if len(x) < 5:
        return None
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    r2 = c + cx * cx + cy * cy
    if not np.isfinite(r2) or r2 <= 0:
        return None
    return float(cx), float(cy), float(math.sqrt(r2))


def _runs_mask(hit: np.ndarray, min_len: int, max_gap: int) -> np.ndarray:
    """Circular: bridge gaps <= max_gap, then keep only runs >= min_len. Returns the filtered hit mask (of the ORIGINAL hits)."""
    n = len(hit)
    if not hit.any():
        return hit.copy()
    if hit.all():
        return hit.copy()
    start = int(np.argmin(hit))                       # rotate so index 0 is a miss: no run straddles the seam
    h = np.roll(hit, -start)
    filled = h.copy()
    i = 0
    while i < n:
        if not h[i]:
            j = i
            while j < n and not h[j]:
                j += 1
            if j - i <= max_gap and 0 < i and j < n:   # interior gap between two hits
                filled[i:j] = True
            i = j
        else:
            i += 1
    keep = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if filled[i]:
            j = i
            while j < n and filled[j]:
                j += 1
            if j - i >= min_len:
                keep[i:j] = True
            i = j
        else:
            i += 1
    return np.roll(keep & h, start)


def _arc_span_deg(hit: np.ndarray) -> float:
    """360 minus the largest circular gap between supported angles."""
    idx = np.flatnonzero(hit)
    if len(idx) == 0:
        return 0.0
    if len(idx) == 1:
        return 0.0
    gaps = np.diff(np.concatenate([idx, [idx[0] + len(hit)]]))
    return float((len(hit) - gaps.max() + 1) * (360.0 / len(hit)))


def _measure(ctx: _Ctx, cx: float, cy: float, r: float, cfg: DetectionConfig, tol: Optional[float] = None) -> dict:
    """Edge evidence for one circle, plus the rim points (best edge per supported angle) for refitting."""
    tol = tol if tol is not None else max(2.0, 0.02 * r)
    offs = np.arange(-tol, tol + 1e-6, 1.0)
    rr = r + offs
    xs = cx + ctx.cos[:, None] * rr[None, :]
    ys = cy + ctx.sin[:, None] * rr[None, :]
    xi, yi = np.rint(xs).astype(int), np.rint(ys).astype(int)
    inside = (xi >= 1) & (xi < ctx.w - 1) & (yi >= 1) & (yi < ctx.h - 1)
    visible = inside.all(axis=1)
    xi, yi = np.clip(xi, 0, ctx.w - 1), np.clip(yi, 0, ctx.h - 1)
    gx, gy, gm = ctx.gx[yi, xi], ctx.gy[yi, xi], ctx.gmag[yi, xi]
    radial = gx * ctx.cos[:, None] + gy * ctx.sin[:, None]
    aligned = ctx.edges[yi, xi] & inside & (gm > 1e-6) & (np.abs(radial) >= 0.75 * gm)
    raw_hit = aligned.any(axis=1) & visible
    hit = _runs_mask(raw_hit, cfg.min_run_samples, cfg.max_gap_samples)
    n_vis = int(visible.sum())
    score_k = np.where(aligned, np.abs(radial), -1.0)
    best_k = score_k.argmax(axis=1)
    rows = np.arange(ANGLES)
    px, py = xi[rows, best_k], yi[rows, best_k]
    sel = hit
    pol = np.sign(radial[rows, best_k])[sel]
    polarity = float(max((pol > 0).mean(), (pol < 0).mean())) if sel.any() else 0.0
    return {"support": float(hit.sum()) / n_vis if n_vis else 0.0, "visible_fraction": n_vis / ANGLES,
            "arc_span_deg": _arc_span_deg(hit), "points": (px[sel].astype(float), py[sel].astype(float)), "hit": hit,
            "polarity_consistency": polarity, "n_supported": int(hit.sum())}


def _refine(ctx: _Ctx, cx: float, cy: float, r: float, cfg: DetectionConfig) -> Optional[dict]:
    """Alternate measure -> robust refit a few times so a Hough circle that is a few pixels off locks onto the real rim."""
    best = None
    for _ in range(3):
        m = _measure(ctx, cx, cy, r, cfg)
        if best is None or m["support"] > best[0]["support"]:
            best = (m, cx, cy, r)
        x, y = m["points"]
        fit = _fit_circle(x, y)
        if fit is None:
            break
        fcx, fcy, fr = fit
        d = np.abs(np.hypot(x - fcx, y - fcy) - fr)                          # drop outlier points once, refit
        keep = d <= max(1.5, 2.5 * 1.4826 * float(np.median(d)))
        if keep.sum() >= 8 and keep.sum() < len(x):
            fit = _fit_circle(x[keep], y[keep]) or fit
            fcx, fcy, fr = fit
        if not (0 < fr < 4 * max(ctx.h, ctx.w)):
            break
        if abs(fcx - cx) < 0.3 and abs(fcy - cy) < 0.3 and abs(fr - r) < 0.3:
            cx, cy, r = fcx, fcy, fr
            break
        cx, cy, r = fcx, fcy, fr
    m = _measure(ctx, cx, cy, r, cfg)
    if best is not None and best[0]["support"] > m["support"]:
        m, cx, cy, r = best
    x, y = m["points"]
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2))) if len(x) else float("inf")
    return {"cx": cx, "cy": cy, "r": r, "rms": rms, **m}


def _score(support: float, arc: float, rms: float, r: float) -> float:
    fit = max(0.0, 1.0 - (rms / r) / 0.03) if np.isfinite(rms) else 0.0       # an rms of 3% of the radius or worse scores 0
    arc_f = min(1.0, arc / 270.0)                                              # a full rim is not required: 270 degrees is "enough"
    return float(support * (0.5 + 0.5 * arc_f) * (0.5 + 0.5 * fit))


class AutoPlateDetectionService:
    def __init__(self, config: Optional[DetectionConfig] = None):
        self.cfg = config or DetectionConfig()

    # ------------------------------------------------------------------ preparation
    def _prepare(self, rgb: np.ndarray) -> tuple[_Ctx, float, np.ndarray]:
        h, w = rgb.shape[:2]
        scale = min(1.0, self.cfg.work_side / max(h, w))
        small = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1.0 else rgb
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        blur = cv2.GaussianBlur(cv2.medianBlur(eq, 5), (5, 5), 1.5)
        high, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        high = float(np.clip(high, 40, 200))
        edges = cv2.Canny(blur, 0.5 * high, high)
        f = blur.astype(np.float32)
        gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
        ctx = _Ctx(gray, edges, gx, gy)
        ctx.blur, ctx.canny_high = blur, high
        return ctx, scale, edges

    def _candidates(self, ctx: _Ctx, edges: np.ndarray) -> list[tuple[float, float, float, str]]:
        h, w = ctx.h, ctx.w
        m = min(h, w)
        rmin, rmax = self.cfg.min_radius_frac * m, self.cfg.max_radius_frac * m
        out: list[tuple[float, float, float, str]] = []
        for p2 in (40, 25):                                                    # strict accumulator first, then permissive
            found = cv2.HoughCircles(ctx.blur, cv2.HOUGH_GRADIENT, dp=1.0, minDist=max(8.0, 0.05 * m), param1=ctx.canny_high,
                                     param2=p2, minRadius=int(rmin), maxRadius=int(rmax))
            if found is not None:
                out += [(float(x), float(y), float(r), "hough") for x, y, r in found[0][:15]]
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        long_enough = [c for c in contours if len(c) >= 2 * math.pi * rmin * 0.25]    # at least a quarter of the smallest plausible rim
        for c in sorted(long_enough, key=len, reverse=True)[:25]:
            pts = c[:, 0, :].astype(float)
            fit = _fit_circle(pts[:, 0], pts[:, 1])
            if fit:
                out.append((*fit, "contour"))
            (ex, ey), er = cv2.minEnclosingCircle(c)
            out.append((float(ex), float(ey), float(er), "contour_enclosing"))
        return out

    # ------------------------------------------------------------------ public
    def detect(self, rgb: np.ndarray) -> dict:
        t0 = time.perf_counter()
        h, w = rgb.shape[:2]
        base = {"image": {"width": w, "height": h}, "evidence_note": EVIDENCE_NOTE, "plate": None, "evidence": None,
                "candidates": [], "warnings": []}

        def finish(status, reason, message, **extra):
            return {**base, "status": status, "reason": reason, "message": message, **extra,
                    "timing_ms": round((time.perf_counter() - t0) * 1000.0, 1)}

        if rgb.ndim != 3 or rgb.shape[2] != 3 or min(h, w) < 64:
            return finish("failed", "image_unsuitable", "The image is too small or not a colour image; plate detection needs at least 64 px on each side.")
        ctx, scale, edges = self._prepare(rgb)
        if float(ctx.gray.std()) < self.cfg.min_contrast_std:
            return finish("failed", "low_contrast_image", f"The image has almost no contrast (grayscale std {float(ctx.gray.std()):.1f}); "
                          "it is blank, washed out or covered. Fix the lighting or calibrate manually.", diagnostics={"gray_std": round(float(ctx.gray.std()), 2)})
        m = min(ctx.h, ctx.w)
        rmin, rmax = self.cfg.min_radius_frac * m, self.cfg.max_radius_frac * m

        refined: list[dict] = []
        raw = []
        for c in self._candidates(ctx, edges):                                  # the same circle is usually proposed several times: refine it once
            if not any(math.hypot(c[0] - k[0], c[1] - k[1]) < 0.02 * k[2] and abs(c[2] - k[2]) < 0.02 * k[2] for k in raw):
                raw.append(c)
        rejected = {"implausible_geometry": 0, "weak_edge_support": 0}
        for cx, cy, r, origin in raw:
            if not (np.isfinite([cx, cy, r]).all() and rmin * 0.7 <= r <= rmax * 1.3 and 0 <= cx < ctx.w and 0 <= cy < ctx.h):
                rejected["implausible_geometry"] += 1
                continue
            ref = _refine(ctx, cx, cy, r, self.cfg)
            if ref is None:
                continue
            ref["origin"] = origin
            if not (rmin <= ref["r"] <= rmax and 0 <= ref["cx"] < ctx.w and 0 <= ref["cy"] < ctx.h):
                rejected["implausible_geometry"] += 1
                continue
            if ref["support"] < self.cfg.min_candidate_support:
                rejected["weak_edge_support"] += 1
                continue
            ref["score"] = _score(ref["support"], ref["arc_span_deg"], ref["rms"], ref["r"])
            refined.append(ref)

        # --- merge near-duplicates (same circle found by several routes), keep the best-scoring one ---
        refined.sort(key=lambda c: -c["score"])
        uniq: list[dict] = []
        for c in refined:
            if not any(math.hypot(c["cx"] - u["cx"], c["cy"] - u["cy"]) < 0.08 * u["r"] and abs(c["r"] - u["r"]) < 0.08 * u["r"] for u in uniq):
                uniq.append(c)
        diag = {"candidates_proposed": len(raw), "candidates_scored": len(refined), "distinct_candidates": len(uniq), "rejected": rejected,
                "canny_high": ctx.canny_high, "work_scale": round(scale, 4), "gray_std": round(float(ctx.gray.std()), 2)}
        base["diagnostics"] = diag
        if not uniq:
            return finish("failed", "no_circle_found", "No circular boundary with enough edge evidence was found in a plausible size range. "
                          "The plate may be out of frame, too low-contrast against the surface, or not circular. Calibrate manually.")

        best = uniq[0]
        warnings: list[str] = []
        reasons: list[str] = []
        # --- circles that CONTAIN the best one with comparable evidence: plate rim vs inner well, plate vs placemat/platform ring. ---
        # A slightly larger one (<12 %) is the same rim seen twice (outer edge + shadow line): take the outer silently.
        # A clearly larger one cannot be told apart from "a bigger thing the plate sits on", so the outer is only SUGGESTED (uncertain).
        def encloses(big: dict, small: dict) -> bool:
            return big["r"] > small["r"] and math.hypot(big["cx"] - small["cx"], big["cy"] - small["cy"]) + small["r"] <= big["r"] * 1.03
        # Symmetric: nested circles of comparable evidence are ambiguous whether the strongest one is the inner or the outer of them.
        nested = [u for u in uniq if u is not best and (encloses(u, best) or encloses(best, u)) and u["score"] >= 0.85 * best["score"]]
        if nested:
            group = [best] + nested
            outer, inner = max(group, key=lambda u: u["r"]), min(group, key=lambda u: u["r"])
            if outer["r"] < 1.12 * inner["r"]:
                warnings.append("Nearly identical circles (rim edge and its shadow line) were merged; the outer one was used.")
            else:
                reasons.append("enclosing_candidates")
                warnings.append(f"Nested circles with comparable edge evidence were found (outer {outer['r'] / inner['r']:.2f}x the inner radius): "
                                "plate rim vs inner well, or plate vs placemat/platform ring (the outer circle encloses the inner). "
                                "The outer circle is suggested; confirm it.")
            best = outer
        rivals = [u for u in uniq if u is not best and not encloses(best, u) and not encloses(u, best)
                  and u["score"] >= best["score"] - self.cfg.ambiguity_margin and u["score"] >= self.cfg.accept_score]

        inv = 1.0 / scale
        def out_c(c: dict) -> dict:
            return {"center_x": round(c["cx"] * inv, 2), "center_y": round(c["cy"] * inv, 2), "radius_px": round(c["r"] * inv, 2),
                    "edge_support": round(c["support"], 3), "arc_span_deg": round(c["arc_span_deg"], 1), "fit_rms_px": round(c["rms"] * inv, 2),
                    "visible_fraction": round(c["visible_fraction"], 3), "evidence_score": round(c["score"], 3), "origin": c["origin"]}

        evidence = {**out_c(best), "polarity_consistency": round(best["polarity_consistency"], 3),
                    "partially_visible": best["visible_fraction"] < 0.98}
        cx, cy, r = best["cx"] * inv, best["cy"] * inv, best["r"] * inv
        try:   # PlateGeometry is the single authority on what a valid plate is (centre inside the frame, positive radius)
            geo = PlateGeometry(cx, cy, r, w, h)
        except ValueError as exc:
            return finish("failed", "center_outside_frame", f"The best circle's centre lies outside the image, which cannot be represented as a plate ({exc}). "
                          "Re-frame so more than half of the plate is visible, or calibrate manually.", candidates=[out_c(u) for u in uniq[:self.cfg.max_candidates]], diagnostics=diag)
        if evidence["partially_visible"]:
            warnings.append(f"Only {best['visible_fraction'] * 100:.0f}% of the rim lies inside the image; centre and radius come from a partial arc.")
        if best["polarity_consistency"] < 0.7:
            warnings.append("The rim edge changes polarity along its length (mixed light/dark transitions); the boundary may be a texture or shadow.")

        if best["support"] < self.cfg.accept_support:
            reasons.append("weak_edge_support")
        if best["arc_span_deg"] < self.cfg.accept_arc_deg:
            reasons.append("insufficient_arc")
        if best["score"] < self.cfg.accept_score:
            reasons.append("low_evidence_score")
        if rivals:
            reasons.append("ambiguous_candidates")
            warnings.append(f"{len(rivals)} other distinct circle(s) have similar evidence; confirm which one is the plate.")
        plate = {**geo.to_dict(), "plate_diameter_mm": None}
        cands = [out_c(u) for u in uniq[:self.cfg.max_candidates]]
        if reasons:
            return finish("uncertain", reasons[0], "A plausible circle was found but the evidence is not strong enough to use it automatically ("
                          + ", ".join(reasons) + "). Review it or calibrate manually.", plate=plate, evidence={**evidence, "reasons": reasons},
                          candidates=cands, warnings=warnings, diagnostics=diag)
        return finish("detected", None, "Plate boundary detected from image edges.", plate=plate, evidence=evidence, candidates=cands,
                      warnings=warnings, diagnostics=diag)

    def verify(self, rgb: np.ndarray, geometry: PlateGeometry, min_support: float) -> dict:
        """Does a given (saved) circle still sit on a real rim edge in THIS image? Measures only; never moves the circle."""
        h, w = rgb.shape[:2]
        if (geometry.image_width, geometry.image_height) != (w, h):
            geometry = geometry.scaled_to(w, h)
        ctx, scale, _ = self._prepare(rgb)
        r = geometry.radius_px * scale
        m = _measure(ctx, geometry.center_x * scale, geometry.center_y * scale, r, self.cfg, tol=max(3.0, 0.03 * r))
        ok = m["support"] >= min_support and m["visible_fraction"] >= 0.30 and m["arc_span_deg"] >= 90.0
        return {"verified": bool(ok), "edge_support": round(m["support"], 3), "arc_span_deg": round(m["arc_span_deg"], 1),
                "visible_fraction": round(m["visible_fraction"], 3), "min_support": min_support,
                "note": "fraction of the saved rim that has an edge on it in this image; a low value means the plate, camera or framing moved"}
