"""Deterministic synthetic top-down scenes for plate-detection tests. Images are RGB uint8."""
from __future__ import annotations

import cv2
import numpy as np


def table(h: int, w: int, seed: int = 0, base=(70, 100, 140), tex: float = 10) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.float32)
    img[:] = base
    grain = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.2) * tex * 2
    stripes = np.sin(np.arange(w)[None, :] / 9.0 + rng.uniform(0, 6)) * tex * 0.6
    img += (grain + stripes)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def plate_scene(h=480, w=640, cx=None, cy=None, r=None, seed=0, rim=(235, 235, 232), well=True, food=True, noise=3, base=(70, 100, 140), tex=10) -> np.ndarray:
    cx = w / 2 if cx is None else cx
    cy = h / 2 if cy is None else cy
    r = 0.42 * min(h, w) if r is None else r
    img = table(h, w, seed, base, tex)
    cv2.circle(img, (int(cx + r * 0.03), int(cy + r * 0.04)), int(r * 1.02), (40, 55, 80), -1, cv2.LINE_AA)   # soft shadow
    cv2.circle(img, (int(cx), int(cy)), int(r), rim, -1, cv2.LINE_AA)
    if well:
        cv2.circle(img, (int(cx), int(cy)), int(r * 0.72), (215, 215, 212), -1, cv2.LINE_AA)
    if food:
        for col, (dx, dy, rr) in zip([(190, 150, 60), (60, 140, 70), (200, 90, 80)], [(-0.25, -0.1, .25), (0.2, -0.2, .2), (0.05, 0.28, .22)]):
            cv2.circle(img, (int(cx + dx * r), int(cy + dy * r)), int(rr * r), col, -1, cv2.LINE_AA)
    return np.clip(img.astype(np.float32) + np.random.default_rng(seed + 2).normal(0, noise, img.shape), 0, 255).astype(np.uint8)


def flat_gray(h=480, w=640, value=128) -> np.ndarray:
    return np.full((h, w, 3), value, np.uint8)
