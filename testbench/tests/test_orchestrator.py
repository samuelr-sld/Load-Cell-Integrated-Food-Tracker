"""Orchestration with real crop/preprocess/record code and fake networks."""
import numpy as np
import pytest
import torch

from .conftest import color_scene, detection, jpeg_bytes, make_pipeline, rect_mask


def _upload_bytes(scene):
    return jpeg_bytes(scene, quality=100)


def test_empty_segmentation_completes_without_calling_the_classifier():
    pipe, seg, cls = make_pipeline(detections=[])
    rec = pipe.process_upload(jpeg_bytes(np.full((80, 80, 3), 120, np.uint8)), "blank.jpg")
    assert rec.status == "completed" and rec.segments == []
    assert cls.calls == 0 and seg.calls == 1
    assert rec.timings_ms["num_segments"] == 0 and rec.timings_ms["classify_ms_per_segment"] is None
    assert any("classifier was not called" in n for n in rec.notes)
    assert {"original", "overlay"} <= set(rec.images)


def test_single_instance_gets_top5_and_all_debug_images():
    img, dets = color_scene()
    pipe, _, cls = make_pipeline(detections=[dets[0]])
    rec = pipe.process_upload(_upload_bytes(img), "one.jpg")
    assert len(rec.segments) == 1
    s = rec.segments[0]
    assert s.id == 1 and s.status == "classified" and cls.batch_sizes == [1]
    assert len(s.predictions) == 5 and s.predictions[0]["label"] == "red"
    assert set(s.image_keys) == {"masked", "bbox", "model_input"} and all(k in rec.images for k in s.image_keys.values())
    assert s.crop_size_wh == [70, 70]


def test_multiple_instances_each_prediction_belongs_to_its_own_segment():
    img, dets = color_scene()
    pipe, _, cls = make_pipeline(detections=dets)           # red square, tiny green speck (skipped), blue disc
    rec = pipe.process_upload(_upload_bytes(img), "three.jpg")
    red, speck, blue = rec.segments
    assert (red.status, speck.status, blue.status) == ("classified", "skipped", "classified")
    assert speck.predictions == [] and speck.skip_reason.startswith("mask_too_small")
    assert cls.batch_sizes == [2]                           # skipped crop never reaches the classifier
    # the skipped segment sits BETWEEN the two classified ones: a positional bug would hand 'blue' to the speck
    assert red.predictions[0]["label"] == "red" and blue.predictions[0]["label"] == "blue"
    assert [s.id for s in rec.segments] == [1, 2, 3]


def test_timings_are_recorded_and_consistent():
    img, dets = color_scene()
    pipe, _, _ = make_pipeline(detections=dets)
    t = pipe.process_upload(_upload_bytes(img), "t.jpg").timings_ms
    for k in ("decode_ms", "yolo_ms", "crop_preprocess_ms", "classify_ms", "render_encode_ms", "inference_ms", "total_ms"):
        assert t[k] >= 0
    assert abs(t["inference_ms"] - (t["yolo_ms"] + t["crop_preprocess_ms"] + t["classify_ms"])) < 0.1
    assert t["total_ms"] >= t["inference_ms"]
    assert t["num_segments"] == 3 and t["num_classified"] == 2
    assert t["classify_ms_per_segment"] == pytest.approx(t["classify_ms"] / 2, abs=0.011)
    assert t["yolo_ultralytics_breakdown_ms"] == {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0}


def test_classifier_receives_the_masked_crop_not_the_bbox_crop():
    """A disc inside a square box on a gray background: the box corners lie outside the mask. With the masked crop
    they are black; the bbox crop would show the gray (128) background there."""
    img, dets = color_scene()
    pipe, _, cls = make_pipeline(detections=[dets[2]])
    seen = []
    orig = cls.classify
    cls.classify = lambda ts: (seen.extend(ts), orig(ts))[1]
    pipe.process_upload(_upload_bytes(img), "m.jpg")
    corner = seen[0][:, :6, :6]                                           # top-left corner of the 224x224 tensor
    black = torch.tensor([(0 - m) / s for m, s in zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]).view(3, 1, 1)
    assert torch.allclose(corner, black.expand(3, 6, 6), atol=1e-4)


def test_inference_error_becomes_failed_record_without_leaking_internals():
    pipe, _, _ = make_pipeline(error=RuntimeError("secret C:\\internal\\path exploded"))
    rec = pipe.process_upload(jpeg_bytes(np.zeros((60, 60, 3), np.uint8)), "x.jpg")
    assert rec.status == "failed" and "secret" not in rec.error and "RuntimeError" in rec.error
    assert rec.segments == [] and rec.images == {}


def test_downscale_is_noted_and_shapes_are_consistent():
    from dataclasses import replace
    from testbench.config import Settings
    s = replace(Settings(), max_image_side=100)
    pipe, _, _ = make_pipeline(detections=[], settings=s)
    rec = pipe.process_upload(jpeg_bytes(np.zeros((200, 400, 3), np.uint8)), "wide.jpg")
    assert rec.image["downscaled"] and (rec.image["width"], rec.image["height"]) == (100, 50)
    assert any("downscaled" in n for n in rec.notes)
