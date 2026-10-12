"""Manual evaluation: per-segment verdicts (held in memory) and the preliminary experimental summary."""
from __future__ import annotations

import re
from datetime import datetime, timezone

from .records import USABLE_STATUSES, Feedback, InferenceRecord, SegmentRecord

VERDICTS = ("correct", "incorrect", "ambiguous")
EVAL_MODELS = ("mobilenet", "yolo")
YOLO_CONF_BANDS = ((0.25, 0.5), (0.5, 0.7), (0.7, 1.01))     # last edge > 1 so confidence 1.0 lands in the top band


class FeedbackError(ValueError):
    pass


def normalize_label(raw: str) -> str:
    """'Apple Pie' / 'apple-pie' -> 'apple_pie' (the Food-101 / ImageFolder spelling)."""
    return re.sub(r"[\s\-]+", "_", raw.strip().lower())


def make_feedback(segment: SegmentRecord, verdict: str, actual_label: str | None, class_names: list[str],
                  model: str = "mobilenet") -> Feedback:
    """`model` picks what is being judged: the MobileNet top-1 (needs a classified segment) or the YOLO class (any segment
    that has a valid crop). `class_names` is that model's own taxonomy. YOLO names contain spaces ('egg tart'), so both
    sides are compared in the normalized lower_snake_case spelling."""
    if model not in EVAL_MODELS:
        raise FeedbackError(f"model must be one of {', '.join(EVAL_MODELS)}")
    if model == "yolo":
        if segment.status not in USABLE_STATUSES:
            raise FeedbackError("this segment was skipped, so there is no YOLO identity to evaluate")
        predicted, taxonomy, what = normalize_label(segment.yolo_class_name), {normalize_label(n) for n in class_names}, "YOLO class"
    else:
        if segment.status != "classified":
            raise FeedbackError("this segment was not classified, so there is no prediction to evaluate")
        predicted, taxonomy, what = segment.predictions[0]["label"], class_names, "top-1 prediction"
    if verdict not in VERDICTS:
        raise FeedbackError(f"verdict must be one of {', '.join(VERDICTS)}")
    raw = (actual_label or "").strip()
    if raw and verdict != "incorrect":
        raise FeedbackError("an actual label can only be given together with verdict 'incorrect'")
    normalized, in_tax = (normalize_label(raw), None) if raw else (None, None)
    if normalized:
        in_tax = normalized in taxonomy
        if normalized == predicted:
            raise FeedbackError(f"the actual label equals the {what}; mark the prediction as correct instead")
    return Feedback(verdict, normalized, raw or None, in_tax, datetime.now(timezone.utc).isoformat(timespec="seconds"))


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def _yolo_summary(segs: list[SegmentRecord]) -> dict:
    """Manual verdicts on YOLO's own class. Accuracy is also split by YOLO confidence band, which answers
    'is the confidence score a usable gate for trusting the class?'."""
    usable = [s for s in segs if s.status in USABLE_STATUSES]
    fbs = [(s, s.yolo_feedback) for s in usable if s.yolo_feedback]
    correct = [s for s, f in fbs if f.verdict == "correct"]
    incorrect = [s for s, f in fbs if f.verdict == "incorrect"]
    n_eval = len(correct) + len(incorrect)
    bands = []
    for lo, hi in YOLO_CONF_BANDS:
        c = sum(1 for s in correct if lo <= s.yolo_confidence < hi)
        i = sum(1 for s in incorrect if lo <= s.yolo_confidence < hi)
        bands.append({"yolo_confidence": f"{lo:.2f}-{min(hi, 1.0):.2f}", "evaluated": c + i, "correct": c,
                      "accuracy": round(c / (c + i), 4) if c + i else None})
    return {
        "segments_with_yolo_identity": len(usable), "manually_evaluated": n_eval, "correct": len(correct), "incorrect": len(incorrect),
        "ambiguous_or_not_evaluable": sum(1 for _, f in fbs if f.verdict == "ambiguous"),
        "not_yet_evaluated": len(usable) - len(fbs),
        "top1_accuracy": round(len(correct) / n_eval, 4) if n_eval else None, "top1_denominator": n_eval,
        "accuracy_by_confidence": bands,
        "incorrect_with_label_outside_yolo_taxonomy": sum(1 for s in incorrect if s.yolo_feedback.in_taxonomy is False),
    }


def _head_to_head(segs: list[SegmentRecord]) -> dict:
    """Only segments where BOTH models were judged correct/incorrect (compare mode), so the two accuracies are measured
    on exactly the same segments."""
    both = [s for s in segs if s.status == "classified" and s.feedback and s.yolo_feedback
            and s.feedback.verdict in ("correct", "incorrect") and s.yolo_feedback.verdict in ("correct", "incorrect")]
    y = lambda s: s.yolo_feedback.verdict == "correct"
    m = lambda s: s.feedback.verdict == "correct"
    n = len(both)
    return {
        "segments_judged_for_both": n,
        "both_correct": sum(1 for s in both if y(s) and m(s)), "yolo_only_correct": sum(1 for s in both if y(s) and not m(s)),
        "mobilenet_only_correct": sum(1 for s in both if m(s) and not y(s)), "both_incorrect": sum(1 for s in both if not y(s) and not m(s)),
        "yolo_accuracy": round(sum(1 for s in both if y(s)) / n, 4) if n else None,
        "mobilenet_accuracy": round(sum(1 for s in both if m(s)) / n, 4) if n else None,
    }


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

    classify_ms = sum(r.timings_ms.get("classify_ms") or 0.0 for r in done if any(s.status == "classified" for s in r.segments))
    return {
        "label": "PRELIMINARY MANUAL EVALUATION - small, hand-labelled sample; not a benchmark",
        "counts": {
            "images_processed": len(done), "images_failed": len(records) - len(done),
            "segments_detected": len(segs), "segments_classified": len(classified),
            "segments_skipped": sum(1 for s in segs if s.status == "skipped"),   # invalid crop; YOLO-only segments are not 'skipped'
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
        "images_by_mode": {m: sum(1 for r in done if r.mode == m) for m in ("yolo_only", "compare")},
        "yolo": _yolo_summary(segs),
        "head_to_head": _head_to_head(segs),
        "caveats": [
            "The MobileNet figures above judge the Food-101 top-1; the 'yolo' block judges YOLO's FoodSeg103 ingredient class. "
            "Verdicts on the two models are independent, and 'head_to_head' only uses segments judged for both.",
            "Unevaluated and ambiguous segments are never counted as correct.",
            "Accuracy is computed only from manual verdicts, never from model confidence.",
            "Top-5 excludes 'incorrect' segments without an actual label inside the classifier's 101-class taxonomy, "
            "so it is optimistic; check top5_excluded_segments.",
            "The detector predicts FoodSeg103 ingredient classes while the classifier knows Food-101 dish classes; "
            "many ingredient crops have no valid label in the classifier's taxonomy.",
            "Summary covers results currently held in memory." + (f" {evicted_total} older result(s) were evicted and are not included." if evicted_total else ""),
        ],
    }
