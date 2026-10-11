import pytest

from testbench.evaluation import FeedbackError, build_summary, make_feedback, normalize_label
from testbench.records import Feedback, InferenceRecord, SegmentRecord
from .conftest import CLASS_NAMES


def _seg(sid, top5=("red", "green", "blue", "filler_0", "filler_1"), status="classified", fb=None):
    preds = [{"rank": i + 1, "class_index": i, "label": l, "softmax_score": 0.5 / (i + 1)} for i, l in enumerate(top5)] if status == "classified" else []
    return SegmentRecord(sid, 0, "x", 0.9, [0, 0, 10, 10], [0, 0, 10, 10], [255, 0, 0], [10, 10], 80, 0.8, status, None, preds, {}, fb)


def _fb(verdict, actual=None, in_tax=None):
    return Feedback(verdict, actual, actual, in_tax, "t")


def _rec(rid, segs, total=100.0, classify=20.0, status="completed"):
    return InferenceRecord(rid, rid + ".jpg", "t", status, None, {}, "cpu", segs,
                           {"total_ms": total, "inference_ms": total - 10, "classify_ms": classify})


def test_unevaluated_predictions_are_never_counted_as_correct():
    s = build_summary([_rec("a", [_seg(1), _seg(2)])])
    assert s["top1_accuracy"] is None and s["top5_accuracy"] is None
    assert s["counts"]["not_yet_evaluated"] == 2 and s["counts"]["segments_manually_evaluated"] == 0


def test_top1_uses_only_correct_and_incorrect_ambiguous_is_excluded():
    segs = [_seg(1, fb=_fb("correct")), _seg(2, fb=_fb("correct")), _seg(3, fb=_fb("incorrect")),
            _seg(4, fb=_fb("ambiguous")), _seg(5)]
    s = build_summary([_rec("a", segs)])
    assert s["top1_accuracy"] == pytest.approx(2 / 3, abs=1e-4) and s["top1_denominator"] == 3
    c = s["counts"]
    assert (c["correct"], c["incorrect"], c["ambiguous_or_not_evaluable"], c["not_yet_evaluated"]) == (2, 1, 1, 1)


def test_top5_requires_a_label_inside_the_classifier_taxonomy():
    segs = [
        _seg(1, fb=_fb("correct")),                                          # hit (truth = top-1)
        _seg(2, fb=_fb("incorrect", "blue", True)),                          # truth in top-5 -> hit
        _seg(3, fb=_fb("incorrect", "filler_9", True)),                      # in taxonomy, not in top-5 -> miss
        _seg(4, fb=_fb("incorrect", "broccoli", False)),                     # outside taxonomy -> excluded
        _seg(5, fb=_fb("incorrect")),                                        # no label -> excluded
    ]
    s = build_summary([_rec("a", segs)])
    assert s["top5_denominator"] == 3 and s["top5_accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["top5_excluded_segments"] == 2
    assert s["top1_accuracy"] == pytest.approx(1 / 5)
    assert s["counts"]["incorrect_with_label_outside_classifier_taxonomy"] == 1


def test_timing_averages_and_counts_and_skipped_segments():
    r1 = _rec("a", [_seg(1), _seg(2, status="skipped")], total=100, classify=30)
    r2 = _rec("b", [_seg(1), _seg(2), _seg(3)], total=200, classify=60)
    s = build_summary([r1, r2, _rec("c", [], status="failed")], evicted_total=4)
    assert s["counts"]["images_processed"] == 2 and s["counts"]["images_failed"] == 1
    assert s["counts"]["segments_detected"] == 5 and s["counts"]["segments_classified"] == 4 and s["counts"]["segments_skipped"] == 1
    assert s["timing"]["avg_total_pipeline_ms"] == 150.0
    assert s["timing"]["avg_classification_ms_per_segment"] == round(90 / 4, 2)
    assert s["results_evicted_from_memory"] == 4 and "PRELIMINARY" in s["label"]
    assert any("evicted" in c for c in s["caveats"])


def test_normalize_label():
    assert normalize_label("  Apple Pie ") == "apple_pie" and normalize_label("fish-and chips") == "fish_and_chips"


def test_make_feedback_rules():
    seg = _seg(1)
    assert make_feedback(seg, "correct", None, CLASS_NAMES).verdict == "correct"
    fb = make_feedback(seg, "incorrect", "Filler 7", CLASS_NAMES)
    assert fb.actual_label == "filler_7" and fb.in_taxonomy is True
    assert make_feedback(seg, "incorrect", "broccoli", CLASS_NAMES).in_taxonomy is False
    assert make_feedback(seg, "incorrect", "", CLASS_NAMES).actual_label is None
    with pytest.raises(FeedbackError):
        make_feedback(seg, "maybe", None, CLASS_NAMES)
    with pytest.raises(FeedbackError):
        make_feedback(seg, "correct", "blue", CLASS_NAMES)             # label only allowed with 'incorrect'
    with pytest.raises(FeedbackError):
        make_feedback(seg, "incorrect", "red", CLASS_NAMES)            # equals top-1 -> contradiction
    with pytest.raises(FeedbackError):
        make_feedback(_seg(2, status="skipped"), "correct", None, CLASS_NAMES)
