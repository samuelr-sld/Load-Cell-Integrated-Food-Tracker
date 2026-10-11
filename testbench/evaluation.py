"""Manual evaluation: per-segment verdicts (held in memory) and the preliminary experimental summary."""
from __future__ import annotations

import re
from datetime import datetime, timezone

from .records import Feedback, InferenceRecord, SegmentRecord

VERDICTS = ("correct", "incorrect", "ambiguous")


class FeedbackError(ValueError):
    pass


def normalize_label(raw: str) -> str:
    """'Apple Pie' / 'apple-pie' -> 'apple_pie' (the Food-101 / ImageFolder spelling)."""
    return re.sub(r"[\s\-]+", "_", raw.strip().lower())


def make_feedback(segment: SegmentRecord, verdict: str, actual_label: str | None, class_names: list[str]) -> Feedback:
    if segment.status != "classified":
        raise FeedbackError("this segment was not classified, so there is no prediction to evaluate")
    if verdict not in VERDICTS:
        raise FeedbackError(f"verdict must be one of {', '.join(VERDICTS)}")
    raw = (actual_label or "").strip()
    if raw and verdict != "incorrect":
        raise FeedbackError("an actual label can only be given together with verdict 'incorrect'")
    normalized, in_tax = (normalize_label(raw), None) if raw else (None, None)
    if normalized:
        in_tax = normalized in class_names
        if normalized == segment.predictions[0]["label"]:
            raise FeedbackError("the actual label equals the top-1 prediction; mark the prediction as correct instead")
    return Feedback(verdict, normalized, raw or None, in_tax, datetime.now(timezone.utc).isoformat(timespec="seconds"))


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def build_summary(records: list[InferenceRecord], evicted_total: int = 0) -> dict:
    done = [r for r in records if r.status == "completed"]
    segs = [s for r in done for s in r.segments]
    classified = [s for s in segs if s.status == "classified"]

    correct = [s for s in classified if s.feedback and s.feedback.verdict == "correct"]
    incorrect = [s for s in classified if s.feedback and s.feedback.verdict == "incorrect"]
    ambiguous = [s for s in classified if s.feedback and s.feedback.verdict == "ambiguous"]
    unevaluated = len(classified) - len(correct) - len(incorrect) - len(ambiguous)
    n_eval = len(correct) + len(incorrect)

    # Top-5 needs a ground-truth label inside the classifier's taxonomy:
    #  - 'correct' segments: truth is the top-1 label, so it is trivially inside the top-5.
    #  - 'incorrect' segments: usable only when the user entered an actual label that is one of the classifier's classes.
    inc_in_tax = [s for s in incorrect if s.feedback.in_taxonomy]
    top5_hits = len(correct) + sum(1 for s in inc_in_tax if s.feedback.actual_label in {p["label"] for p in s.predictions})
    top5_n = len(correct) + len(inc_in_tax)

    classify_ms = sum(r.timings_ms.get("classify_ms", 0.0) for r in done if any(s.status == "classified" for s in r.segments))
    return {
        "label": "PRELIMINARY MANUAL EVALUATION - small, hand-labelled sample; not a benchmark",
        "counts": {
            "images_processed": len(done), "images_failed": len(records) - len(done),
            "segments_detected": len(segs), "segments_classified": len(classified),
            "segments_skipped": len(segs) - len(classified),
            "segments_manually_evaluated": n_eval,   # correct + incorrect
            "correct": len(correct), "incorrect": len(incorrect), "ambiguous_or_not_evaluable": len(ambiguous),
            "not_yet_evaluated": unevaluated,
            "incorrect_with_label_outside_classifier_taxonomy": sum(1 for s in incorrect if s.feedback.in_taxonomy is False),
        },
        "top1_accuracy": round(len(correct) / n_eval, 4) if n_eval else None,
        "top1_denominator": n_eval,
        "top5_accuracy": round(top5_hits / top5_n, 4) if top5_n else None,
        "top5_denominator": top5_n,
        "top5_excluded_segments": len(incorrect) - len(inc_in_tax),
        "timing": {
            "avg_total_pipeline_ms": _mean([r.timings_ms["total_ms"] for r in done if "total_ms" in r.timings_ms]),
            "avg_inference_ms": _mean([r.timings_ms["inference_ms"] for r in done if "inference_ms" in r.timings_ms]),
            "avg_classification_ms_per_segment": round(classify_ms / len(classified), 2) if classified else None,
        },
        "results_evicted_from_memory": evicted_total,
        "caveats": [
            "Unevaluated and ambiguous segments are never counted as correct.",
            "Accuracy is computed only from manual verdicts, never from model confidence.",
            "Top-5 excludes 'incorrect' segments without an actual label inside the classifier's 101-class taxonomy, "
            "so it is optimistic; check top5_excluded_segments.",
            "The detector predicts FoodSeg103 ingredient classes while the classifier knows Food-101 dish classes; "
            "many ingredient crops have no valid label in the classifier's taxonomy.",
            "Summary covers results currently held in memory." + (f" {evicted_total} older result(s) were evicted and are not included." if evicted_total else ""),
        ],
    }
