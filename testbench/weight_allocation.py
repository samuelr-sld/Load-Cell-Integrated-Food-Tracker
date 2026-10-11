"""Split ONE measured net food weight across the food regions of a PlateAnalysis, in proportion to deduplicated area.

    weight_i = W * (area_i * k_i) / sum_j (area_j * k_j)

area_i is the region's attributed on-plate area (overlaps counted once) and k_i an optional food-specific calibration
coefficient (default 1.0, i.e. pure area-proportional). Area is not volume or mass, and no accuracy margin has been
validated for this method; results are labelled as estimates.

Rules this module enforces:
* Unconfirmed regions keep their share of the weight as UNRESOLVED mass. It is never handed to confirmed foods.
* A confirmed food without mask area, an invalid weight, an invalid coefficient or a non-food region still in the analysis
  rejects the allocation: any number produced would silently move that mass onto other foods.
* Detected segment count and user-corrected entity count are kept apart; per-item weight uses only the user's count.
* Mass is conserved (allocated == W) exactly when every region is confirmed and valid; otherwise allocated + unresolved == W.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from .nutrition import norm_label
from .plate_analysis import PlateAnalysis
from .regions import MAX_WEIGHT_G, valid_weight

WEIGHT_SOURCES = ("manual", "load_cell")
IDENTITY_STATUSES = ("confirmed", "needs_confirmation", "not_food")
METHOD = "area_proportional"
DISCLAIMER = ("Estimated by splitting the measured total weight in proportion to visible segmentation area. Segmentation area "
              "is not food volume or mass, and this method has no validated accuracy margin.")


# ------------------------------------------------------------------ inputs
@dataclass(frozen=True)
class TotalWeightMeasurement:
    """Net food weight for the whole plate (plate/tare already removed). Manual today; a load cell later."""
    net_weight_g: float
    source: str = "manual"
    measured_at: Optional[str] = None     # ISO-8601, informational
    tare_g: Optional[float] = None        # informational only; net_weight_g is what gets allocated

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class RegionIdentity:
    region_id: int
    status: str = "needs_confirmation"            # see IDENTITY_STATUSES
    label: Optional[str] = None
    taxonomy: Optional[str] = None
    user_entity_count: Optional[int] = None       # user-corrected number of physical items; None = not stated


@dataclass(frozen=True)
class EntityCount:
    detected_segments: int                        # what YOLO produced for this region (duplicates included)
    user_count: Optional[int]                     # what the user says is physically there; never inferred

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# ------------------------------------------------------------------ outputs
@dataclass
class AllocationIssue:
    code: str
    severity: str                                 # "blocking" | "warning"
    message: str
    region_id: Optional[int] = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class EstimatedIngredientWeight:
    region_id: int
    status: str                                   # "allocated" | "unresolved" | "not_allocated" (whole allocation rejected)
    label: Optional[str]
    taxonomy: Optional[str]
    attributed_area_px: float
    coefficient: float
    weight_share: float                           # fraction of the total weight this region's weighted area represents
    estimated_weight_g: Optional[float]           # only for confirmed foods
    unresolved_weight_g: Optional[float]          # only for unconfirmed regions: mass waiting for an identity
    entity_count: EntityCount
    weight_per_entity_g: Optional[float]          # only when the user gave an entity count
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["entity_count"] = self.entity_count.to_dict()
        return d


@dataclass
class AllocationResult:
    status: str                                   # "complete" | "partial" | "rejected"
    measurement: TotalWeightMeasurement
    allocations: list[EstimatedIngredientWeight]
    allocated_weight_g: float
    unresolved_weight_g: float
    issues: list[AllocationIssue]
    coverage_flags: list[str]
    detected_region_count: int
    method: str = METHOD
    disclaimer: str = DISCLAIMER

    @property
    def mass_conserved(self) -> bool:
        return self.status == "complete"

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"

    def to_dict(self) -> dict:
        return {"status": self.status, "mass_conserved": self.mass_conserved, "measurement": self.measurement.to_dict(),
                "allocated_weight_g": self.allocated_weight_g, "unresolved_weight_g": self.unresolved_weight_g,
                "allocations": [a.to_dict() for a in self.allocations], "issues": [i.to_dict() for i in self.issues],
                "coverage_uncertain": bool(self.coverage_flags), "coverage_flags": list(self.coverage_flags),
                "detected_region_count": self.detected_region_count, "method": self.method, "disclaimer": self.disclaimer}


def round_preserving_total(values: Sequence[float], decimals: int = 1) -> list[float]:
    """Largest-remainder rounding: each value rounded to `decimals`, and the rounded values sum to round(sum(values))."""
    if not values:
        return []
    q = 10 ** decimals
    scaled = [v * q for v in values]
    floors = [math.floor(s) for s in scaled]
    short = round(sum(scaled)) - sum(floors)
    order = sorted(range(len(values)), key=lambda i: -(scaled[i] - floors[i]))
    for i in order[:max(0, short)]:
        floors[i] += 1
    return [f / q for f in floors]


# ------------------------------------------------------------------ service
@dataclass(frozen=True)
class WeightAllocationConfig:
    reject_uncertain_coverage: bool = False       # True: plate-coverage warnings block the allocation instead of flagging it
    decimals: int = 1


class WeightAllocationService:
    def __init__(self, config: WeightAllocationConfig | None = None):
        self.config = config or WeightAllocationConfig()

    def allocate(self, analysis: PlateAnalysis, measurement: TotalWeightMeasurement, identities: Sequence[RegionIdentity],
                 coefficients: Mapping[str, float] | None = None) -> AllocationResult:
        """coefficients: normalized food label -> relative weight-per-area factor (default 1.0). Applied to confirmed foods only."""
        issues: list[AllocationIssue] = []
        block = lambda code, msg, rid=None: issues.append(AllocationIssue(code, "blocking", msg, rid))
        warn = lambda code, msg, rid=None: issues.append(AllocationIssue(code, "warning", msg, rid))
        coef_map = {norm_label(k): v for k, v in (coefficients or {}).items()}

        try:
            w = measurement.net_weight_g
            if isinstance(w, bool) or not isinstance(w, (int, float)):
                raise TypeError("net_weight_g must be a number")
            total = valid_weight(w)
        except (TypeError, ValueError):
            total = None
            block("invalid_weight", f"Net weight must be a number greater than 0 and at most {MAX_WEIGHT_G:g} g.")
        if total is not None and abs(round(total, self.config.decimals) - total) > 1e-9:
            # The estimates are rounded to `decimals`; a finer total could not be reproduced by their sum, so mass would not be conserved.
            block("weight_precision", f"Net weight must have at most {self.config.decimals} decimal place(s) so the estimated weights add up to it exactly.")
            total = None
        if measurement.source not in WEIGHT_SOURCES:
            block("invalid_weight_source", f"Weight source must be one of {', '.join(WEIGHT_SOURCES)}.")

        by_id: dict[int, RegionIdentity] = {}
        for ident in identities:
            if ident.region_id in by_id:
                block("duplicate_identity", "More than one identity was given for this region.", ident.region_id)
            by_id[ident.region_id] = ident
            if ident.status not in IDENTITY_STATUSES:
                block("invalid_identity_status", f"Unknown identity status '{ident.status}'.", ident.region_id)
            elif ident.status == "confirmed" and not (ident.label and ident.label.strip()):
                block("incomplete_identity", "A confirmed region needs a food label.", ident.region_id)
            if ident.user_entity_count is not None and (isinstance(ident.user_entity_count, bool) or not isinstance(ident.user_entity_count, int)
                                                         or ident.user_entity_count < 1):
                block("invalid_entity_count", "User entity count must be a whole number of at least 1.", ident.region_id)

        area_ids = {r.region_id for r in analysis.regions}
        for rid, ident in by_id.items():
            if rid not in area_ids and ident.status == "confirmed":
                block("missing_mask", "This confirmed food has no mask in the plate analysis, so its share of the weight is unknown.", rid)

        if not analysis.regions:
            block("no_food_regions", "The plate analysis contains no food regions.")
        elif analysis.food_area_px == 0:
            block("no_food_area", "No food mask area lies on the plate.")

        rows = []
        for ra in analysis.regions:
            ident = by_id.get(ra.region_id)
            if ident is None:
                warn("missing_identity", "No identity was supplied for this region; it is treated as unconfirmed.", ra.region_id)
                ident = RegionIdentity(ra.region_id)
            if ident.status == "not_food":
                block("not_food_region_in_analysis", "This region was marked as not food; re-run the plate analysis without it "
                      "so its area is not counted.", ra.region_id)
            confirmed = ident.status == "confirmed"
            if not ra.has_area:
                if confirmed:
                    block("missing_mask", "This confirmed food has no attributed mask area on the plate, so its share of the weight is unknown.", ra.region_id)
                else:
                    warn("empty_region", "This unconfirmed region has no area on the plate and receives no weight.", ra.region_id)
            k = 1.0
            if confirmed and ident.label:
                k = coef_map.get(norm_label(ident.label), 1.0)
                if isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(k) or k <= 0:
                    block("invalid_coefficient", f"Calibration coefficient for '{ident.label}' must be a positive finite number.", ra.region_id)
                    k = 1.0
            for f in ra.flags:
                if f in ("tiny_region", "extends_beyond_plate", "mostly_overlapping"):
                    warn(f, f"Region area is uncertain ({f.replace('_', ' ')}).", ra.region_id)
            rows.append((ra, ident, float(k), confirmed))

        cov = list(analysis.coverage_flags)
        for f in cov:
            (block if self.config.reject_uncertain_coverage else warn)("uncertain_coverage", f"Plate coverage is uncertain: {f.replace('_', ' ')}.")

        count = len(analysis.regions)
        if total is None or any(i.severity == "blocking" for i in issues):
            allocs = [self._row("not_allocated", ra, ident, k, 0.0, None, None) for ra, ident, k, _ in rows]
            return AllocationResult("rejected", measurement, allocs, 0.0, 0.0, issues, cov, count)

        weighted = [ra.attributed_area_px * k for ra, _, k, _ in rows]
        denom = sum(weighted)
        shares = [w / denom for w in weighted]
        rounded = round_preserving_total([total * s for s in shares], self.config.decimals)
        allocs = [self._row("allocated", ra, ident, k, share, g, None) if confirmed else self._row("unresolved", ra, ident, k, share, None, g)
                  for (ra, ident, k, confirmed), share, g in zip(rows, shares, rounded)]
        allocated = round(sum(a.estimated_weight_g or 0.0 for a in allocs), self.config.decimals)
        unresolved = round(sum(a.unresolved_weight_g or 0.0 for a in allocs), self.config.decimals)
        status = "complete" if all(c for *_, c in rows) else "partial"
        return AllocationResult(status, measurement, allocs, allocated, unresolved, issues, cov, count)

    @staticmethod
    def _row(status: str, ra, ident: RegionIdentity, k: float, share: float, weight: Optional[float],
             unresolved: Optional[float]) -> EstimatedIngredientWeight:
        confirmed = ident.status == "confirmed"
        per_entity = round(weight / ident.user_entity_count, 2) if weight is not None and ident.user_entity_count else None
        return EstimatedIngredientWeight(
            ra.region_id, status, norm_label(ident.label) if confirmed and ident.label else None,
            ident.taxonomy if confirmed else None, ra.attributed_area_px, k, round(share, 6), weight, unresolved,
            EntityCount(ra.detected_segment_count, ident.user_entity_count), per_entity, list(ra.flags))
