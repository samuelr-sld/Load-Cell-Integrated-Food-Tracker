"""Plate geometry and per-region food AREA on a circular plate. Pure numpy: no models, no UI, no hardware.

What this measures is the visible 2-D footprint of each food region in the image. Area is NOT volume or mass; the weight
allocator (weight_allocation.py) uses it only as a proportional key and says so in its output.

Coordinates follow OpenCV/Ultralytics image convention: x = column, y = row, pixel (row r, col c) is centred at (x=c, y=r).
A pixel is on the plate when (c - cx)^2 + (r - cy)^2 <= radius^2.

Overlaps: two different regions can claim the same pixel (a topping inside a dish, two masks bleeding into each other).
Every pixel is counted once in the total food area. Shared pixels are attributed either by an equal split between the
claimants ("split", default) or wholly to the first claimant in an explicit priority order ("priority").
Same-food duplicates are handled earlier (regions.group_by_mask_overlap); region_mask() unions a region's member masks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

OVERLAP_POLICIES = ("split", "priority")


# ------------------------------------------------------------------ geometry
@dataclass(frozen=True)
class PlateGeometry:
    center_x: float                   # image pixels (column)
    center_y: float                   # image pixels (row)
    radius_px: float
    image_width: int
    image_height: int
    plate_diameter_mm: Optional[float] = None   # physical rim diameter, only for reporting areas in cm^2 (assumes a top-down view)

    def __post_init__(self):
        for name in ("center_x", "center_y", "radius_px"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number")
        if self.radius_px <= 0:
            raise ValueError("radius_px must be greater than 0")
        if int(self.image_width) != self.image_width or int(self.image_height) != self.image_height or self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image_width and image_height must be positive integers")
        if not (0 <= self.center_x < self.image_width and 0 <= self.center_y < self.image_height):
            raise ValueError("plate centre must lie inside the image")
        if self.plate_diameter_mm is not None and (not math.isfinite(self.plate_diameter_mm) or self.plate_diameter_mm <= 0):
            raise ValueError("plate_diameter_mm must be a positive finite number")

    @classmethod
    def from_rim_points(cls, points: Sequence[tuple[float, float]], image_width: int, image_height: int,
                        plate_diameter_mm: Optional[float] = None) -> "PlateGeometry":
        """Circle through three (x, y) points on the plate rim, e.g. three clicks in a calibration UI."""
        if len(points) != 3:
            raise ValueError("exactly three rim points are required")
        (x1, y1), (x2, y2), (x3, y3) = (tuple(map(float, p)) for p in points)
        d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        scale = max(abs(x1 - x2), abs(x2 - x3), abs(y1 - y2), abs(y2 - y3), 1.0)
        if abs(d) < 1e-9 * scale * scale:
            raise ValueError("rim points are collinear (or repeated); they do not define a circle")
        s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
        cx = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
        cy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
        return cls(cx, cy, math.hypot(x1 - cx, y1 - cy), image_width, image_height, plate_diameter_mm)

    def scaled_to(self, image_width: int, image_height: int) -> "PlateGeometry":
        """Same plate expressed in a resized copy of the image (e.g. calibrated on the original, masks on the downscaled one).
        Only uniform scaling is meaningful for a circle, so the aspect ratio must be preserved."""
        sx, sy = image_width / self.image_width, image_height / self.image_height
        if abs(sx - sy) > 0.01 * max(sx, sy):
            raise ValueError("resize changes the aspect ratio; a circular plate cannot be rescaled non-uniformly")
        s = (sx + sy) / 2.0
        return PlateGeometry(self.center_x * sx, self.center_y * sy, self.radius_px * s, image_width, image_height, self.plate_diameter_mm)

    @property
    def mm_per_px(self) -> Optional[float]:
        return None if self.plate_diameter_mm is None else self.plate_diameter_mm / (2.0 * self.radius_px)

    def mask(self) -> np.ndarray:
        yy, xx = np.ogrid[:self.image_height, :self.image_width]
        return (xx - self.center_x) ** 2 + (yy - self.center_y) ** 2 <= self.radius_px ** 2

    @property
    def fully_in_frame(self) -> bool:
        return (self.center_x - self.radius_px >= 0 and self.center_y - self.radius_px >= 0
                and self.center_x + self.radius_px <= self.image_width - 1 and self.center_y + self.radius_px <= self.image_height - 1)

    def to_dict(self) -> dict:
        return {"center_x": self.center_x, "center_y": self.center_y, "radius_px": self.radius_px, "image_width": self.image_width,
                "image_height": self.image_height, "plate_diameter_mm": self.plate_diameter_mm, "mm_per_px": self.mm_per_px,
                "fully_in_frame": self.fully_in_frame}


# ------------------------------------------------------------------ inputs
@dataclass
class FoodRegion:
    """One physical food region as detected: the union mask of its (duplicate-grouped) YOLO segments.
    detected_segment_ids is what the detector produced; it is NOT a count of physical items."""
    region_id: int
    mask: np.ndarray                                   # bool H x W in the same image coordinates as the PlateGeometry
    detected_segment_ids: list[int] = field(default_factory=list)


def region_mask(segment_masks: Sequence[np.ndarray]) -> np.ndarray:
    """Union of a region's member segment masks (its duplicates describe the same food, so their union is its footprint)."""
    if not segment_masks:
        raise ValueError("a region needs at least one segment mask")
    out = np.zeros(segment_masks[0].shape, dtype=bool)
    for m in segment_masks:
        if m.shape != out.shape:
            raise ValueError("segment masks have different shapes")
        out |= m.astype(bool)
    return out


# ------------------------------------------------------------------ outputs
@dataclass
class RegionArea:
    region_id: int
    detected_segment_count: int
    mask_area_px: int                  # raw mask pixels in the frame
    on_plate_px: int                   # after clipping to the plate circle
    off_plate_px: int                  # clipped away
    exclusive_px: int                  # on-plate pixels no other region claims
    shared_px: int                     # on-plate pixels also claimed by another region
    attributed_area_px: float          # this region's share of the unique food area (fractional under "split")
    food_area_pct: float               # attributed / total unique food area * 100
    plate_coverage_pct: float          # attributed / in-frame plate area * 100
    attributed_area_cm2: Optional[float] = None
    flags: list[str] = field(default_factory=list)

    @property
    def has_area(self) -> bool:
        return self.attributed_area_px > 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class PlateAnalysis:
    plate: PlateGeometry
    overlap_policy: str
    plate_area_px: int                 # plate pixels inside the frame
    plate_in_frame_fraction: float     # plate_area_px / (pi r^2)
    food_area_px: int                  # unique (union) on-plate food pixels
    raw_on_plate_area_sum_px: int      # sum of per-region on-plate areas (would double-count overlaps)
    overlap_px: int                    # pixels claimed by two or more regions
    plate_coverage_pct: float
    regions: list[RegionArea]
    coverage_flags: list[str] = field(default_factory=list)

    @property
    def coverage_uncertain(self) -> bool:
        return bool(self.coverage_flags)

    def region(self, region_id: int) -> Optional[RegionArea]:
        return next((r for r in self.regions if r.region_id == region_id), None)

    def to_dict(self) -> dict:
        return {"plate": self.plate.to_dict(), "overlap_policy": self.overlap_policy, "plate_area_px": self.plate_area_px,
                "plate_in_frame_fraction": self.plate_in_frame_fraction, "food_area_px": self.food_area_px,
                "raw_on_plate_area_sum_px": self.raw_on_plate_area_sum_px, "overlap_px": self.overlap_px,
                "plate_coverage_pct": self.plate_coverage_pct, "coverage_uncertain": self.coverage_uncertain,
                "coverage_flags": list(self.coverage_flags), "regions": [r.to_dict() for r in self.regions],
                "note": "Areas are 2-D image footprints, not volume or mass."}


# ------------------------------------------------------------------ service
@dataclass(frozen=True)
class PlateAnalysisConfig:
    overlap_policy: str = "split"
    off_plate_flag_fraction: float = 0.20     # flag a region when more than this share of its mask lies off the plate
    shared_flag_fraction: float = 0.50        # flag a region when more than this share of its on-plate area is shared
    min_region_px: int = 50                   # flag regions smaller than this after clipping (likely noise)

    def __post_init__(self):
        if self.overlap_policy not in OVERLAP_POLICIES:
            raise ValueError(f"overlap_policy must be one of {', '.join(OVERLAP_POLICIES)}")


class PlateAnalysisService:
    def __init__(self, config: PlateAnalysisConfig | None = None):
        self.config = config or PlateAnalysisConfig()

    def analyze(self, plate: PlateGeometry, regions: Sequence[FoodRegion], priority: Sequence[int] | None = None) -> PlateAnalysis:
        """priority: region ids, highest first; required by (and only used with) the "priority" overlap policy."""
        cfg = self.config
        shape = (plate.image_height, plate.image_width)
        ids = [r.region_id for r in regions]
        if len(set(ids)) != len(ids):
            raise ValueError("region ids must be unique")
        for r in regions:
            if r.mask is None or r.mask.shape != shape:
                raise ValueError(f"region {r.region_id}: mask shape {None if r.mask is None else r.mask.shape} does not match image {shape}")
        if cfg.overlap_policy == "priority":
            if priority is None or sorted(priority) != sorted(ids):
                raise ValueError("the priority policy needs a priority list containing every region id exactly once")

        plate_mask = plate.mask()
        plate_px = int(plate_mask.sum())
        if plate_px == 0:
            raise ValueError("the plate circle contains no image pixels")
        clipped = {r.region_id: r.mask.astype(bool) & plate_mask for r in regions}
        claims = np.zeros(shape, dtype=np.uint16)
        for m in clipped.values():
            claims += m
        food_px = int((claims > 0).sum())
        overlap_px = int((claims > 1).sum())

        attributed: dict[int, float] = {}
        if cfg.overlap_policy == "split":
            inv = np.zeros(shape, dtype=np.float64)
            np.divide(1.0, claims, out=inv, where=claims > 0)
            attributed = {rid: float(inv[m].sum()) for rid, m in clipped.items()}
        else:
            taken = np.zeros(shape, dtype=bool)
            for rid in priority:
                attributed[rid] = float((clipped[rid] & ~taken).sum())
                taken |= clipped[rid]

        mm = plate.mm_per_px
        out: list[RegionArea] = []
        for r in regions:
            m = clipped[r.region_id]
            raw = int(r.mask.astype(bool).sum())
            on = int(m.sum())
            excl = int((m & (claims == 1)).sum())
            att = attributed[r.region_id]
            flags = []
            if on == 0:
                flags.append("no_area_on_plate")
            elif on < cfg.min_region_px:
                flags.append("tiny_region")
            if raw and (raw - on) / raw > cfg.off_plate_flag_fraction:
                flags.append("extends_beyond_plate")
            if on and (on - excl) / on > cfg.shared_flag_fraction:
                flags.append("mostly_overlapping")
            out.append(RegionArea(r.region_id, len(r.detected_segment_ids), raw, on, raw - on, excl, on - excl, att,
                                  round(100.0 * att / food_px, 4) if food_px else 0.0, round(100.0 * att / plate_px, 4),
                                  round(att * mm * mm / 100.0, 3) if mm is not None else None, flags))

        cov = []
        if not plate.fully_in_frame:
            cov.append("plate_partially_out_of_frame")
        if any("extends_beyond_plate" in r.flags for r in out):
            cov.append("food_beyond_plate_rim")
        if any("mostly_overlapping" in r.flags for r in out):
            cov.append("heavy_mask_overlap")
        if regions and food_px == 0:
            cov.append("no_food_area_on_plate")
        return PlateAnalysis(plate, cfg.overlap_policy, plate_px, round(plate_px / (math.pi * plate.radius_px ** 2), 4), food_px,
                             sum(r.on_plate_px for r in out), overlap_px, round(100.0 * food_px / plate_px, 4), out, cov)
