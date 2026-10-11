"""YOLO-only mode, the compare mode that is kept beside it, the extended result schema and per-model evaluation.
Real orchestrator/API/store; only the two networks are fakes (see conftest)."""
import io
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from testbench.config import Settings
from testbench.evaluation import FeedbackError, make_feedback
from testbench.main import create_app
from testbench.records import Feedback, SegmentRecord
from .conftest import YOLO_NAMES, color_scene, jpeg_bytes, make_pipeline


@pytest.fixture
def scene():
    img, dets = color_scene()
    return jpeg_bytes(img, 100), dets


@pytest.fixture
def client(scene):
    data, dets = scene
    pipe, seg, cls = make_pipeline(detections=dets)
    with TestClient(create_app(pipeline=pipe)) as c:
        c.scene, c.seg, c.cls = data, seg, cls
        yield c


def _post(client, mode=None, name="meal.jpg"):
    r = client.post("/api/inference", files=[("files", (name, client.scene, "image/jpeg"))], data={"mode": mode} if mode else None)
    return r, (r.json()["items"][0] if r.status_code == 200 else None)


# ------------------------------------------------------------------ orchestrator
def test_yolo_only_never_calls_the_classifier_and_uses_yolo_as_identity(scene):
    data, dets = scene
    pipe, _, cls = make_pipeline(detections=dets)
    rec = pipe.process_upload(data, "m.jpg", "yolo_only")
    assert rec.mode == "yolo_only" and cls.calls == 0
    assert [s.status for s in rec.segments] == ["identified", "skipped", "identified"]
    red, speck, blue = rec.segments
    assert red.predictions == [] and blue.predictions == []
    assert (red.yolo_class_name, red.yolo_class_id) == ("red-thing", 0) and (blue.yolo_class_name, blue.yolo_class_id) == ("blue-thing", 0)
    # crops exist for identified segments (a mask reference, no 224x224 classifier input); skipped ones have none
    assert set(red.image_keys) == {"masked", "bbox", "mask"} and speck.image_keys == {}
    assert "seg1_input" not in rec.images and "seg3_input" not in rec.images


def test_yolo_only_timings_have_no_classifier_stage(scene):
    data, dets = scene
    pipe, _, _ = make_pipeline(detections=dets)
    t = pipe.process_upload(data, "t.jpg", "yolo_only").timings_ms
    assert t["classify_ms"] is None and t["classify_ms_per_segment"] is None
    assert t["num_segments"] == 3 and t["num_identified"] == 2 and t["num_classified"] == 0
    assert abs(t["inference_ms"] - (t["yolo_ms"] + t["crop_preprocess_ms"])) < 0.1


def test_compare_mode_is_the_unchanged_original_pipeline(scene):
    data, dets = scene
    pipe, _, cls = make_pipeline(detections=dets)
    rec = pipe.process_upload(data, "c.jpg", "compare")
    assert rec.mode == "compare" and cls.batch_sizes == [2]
    assert [s.status for s in rec.segments] == ["classified", "skipped", "classified"]
    assert rec.segments[0].predictions[0]["label"] == "red" and rec.segments[2].predictions[0]["label"] == "blue"
    assert rec.timings_ms["num_classified"] == 2 and rec.timings_ms["classify_ms"] > 0


def test_default_mode_comes_from_settings_and_bad_values_fail_loudly(scene):
    data, dets = scene
    pipe, _, cls = make_pipeline(detections=dets)
    assert pipe.process_upload(data, "d.jpg").mode == "compare"                       # shipped default = original behaviour
    pipe, _, cls = make_pipeline(detections=dets, settings=replace(Settings(), default_mode="yolo_only"))
    assert pipe.process_upload(data, "d.jpg").mode == "yolo_only" and cls.calls == 0
    with pytest.raises(ValueError):
        pipe.process_upload(data, "d.jpg", "mobilenet_only")
    with pytest.raises(ValueError):
        Settings(default_mode="nonsense")


def test_no_detections_note_is_mode_specific():
    pipe, _, _ = make_pipeline(detections=[])
    blank = jpeg_bytes(np.full((60, 60, 3), 120, np.uint8))
    assert any("classifier was not called" in n for n in pipe.process_upload(blank, "b.jpg", "compare").notes)
    notes = pipe.process_upload(blank, "b.jpg", "yolo_only").notes
    assert notes and not any("classifier" in n for n in notes)


# ------------------------------------------------------------------ result schema over HTTP
def test_yolo_only_schema_over_http(client):
    r, item = _post(client, "yolo_only")
    assert r.status_code == 200 and item["result"]["mode"] == "yolo_only" and item["result"]["identity_source"] == "yolo"
    assert item["result"]["classifier_preprocessing"] is None
    seg = item["result"]["segments"][0]
    assert seg["segment_id"] == seg["id"] == 1
    assert (seg["yolo_class_id"], seg["yolo_class_name"], seg["yolo_confidence"]) == (0, "red-thing", 0.9)
    assert seg["primary_identity"] == {"source": "yolo", "class_id": 0, "class_name": "red-thing", "confidence": 0.9,
                                       "taxonomy": "FoodSeg103 ingredients"}
    assert seg["mobilenet"] is None and seg["top1"] is None and seg["predictions"] == []
    assert set(seg["image_urls"]) == {"masked", "bbox", "mask"}
    assert client.get(seg["image_urls"]["masked"]).status_code == 200
    assert item["result"]["num_identified"] == 2 and item["result"]["num_classified"] == 0
    assert client.get("/api/results").json()["results"][0]["mode"] == "yolo_only"


def test_compare_schema_exposes_optional_mobilenet_next_to_yolo(client):
    _, item = _post(client, "compare")
    seg = item["result"]["segments"][0]
    assert seg["primary_identity"]["class_name"] == "red-thing"                      # identity is YOLO in both modes
    assert seg["mobilenet"]["top1"]["label"] == "red" and len(seg["mobilenet"]["predictions"]) == 5
    assert seg["top1"] == seg["mobilenet"]["top1"]                                    # legacy alias still present
    assert set(seg["image_urls"]) == {"masked", "bbox", "mask", "model_input"}
    assert item["result"]["classifier_preprocessing"]["random_augmentation"] is False


def test_omitted_mode_uses_the_server_default_and_invalid_mode_is_rejected(client):
    assert _post(client)[1]["result"]["mode"] == "compare"
    r = client.post("/api/inference", files=[("files", ("m.jpg", client.scene, "image/jpeg"))], data={"mode": "banana"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_mode"
    assert len(client.get("/api/results").json()["results"]) == 1                     # the rejected request stored nothing


def test_health_lists_modes_and_yolo_taxonomy(client):
    h = client.get("/api/health").json()
    assert h["modes"] == ["yolo_only", "compare"] and h["default_mode"] == "compare"
    assert h["yolo_class_names"] == ["red-thing", "speck", "blue-thing", "egg tart"]   # stray -1 'background' is not offered


def test_mask_reference_is_a_lossless_png_aligned_with_the_crop(client):
    _, item = _post(client, "yolo_only")
    seg = item["result"]["segments"][0]
    resp = client.get(seg["image_urls"]["mask"])
    assert resp.status_code == 200 and resp.headers["content-type"] == "image/png"
    mask = Image.open(io.BytesIO(resp.content))
    assert mask.format == "PNG" and mask.mode == "L" and mask.size == tuple(seg["crop_size_wh"])
    arr = np.asarray(mask)
    assert set(np.unique(arr)) <= {0, 255} and int((arr == 255).sum()) == seg["mask_area_px"]
    # other debug images are still JPEG
    assert client.get(seg["image_urls"]["masked"]).headers["content-type"] == "image/jpeg"


# ------------------------------------------------------------------ per-model evaluation
def test_yolo_feedback_roundtrip_in_yolo_only_mode(client):
    rid = _post(client, "yolo_only")[1]["result_id"]
    url = f"/api/results/{rid}/segments/1/feedback"
    r = client.put(url, json={"verdict": "correct", "model": "yolo"})
    assert r.status_code == 200 and r.json()["model"] == "yolo" and r.json()["feedback"]["verdict"] == "correct"
    r = client.put(f"/api/results/{rid}/segments/3/feedback", json={"verdict": "incorrect", "actual_label": "Egg Tart", "model": "yolo"})
    assert r.json()["feedback"]["actual_label"] == "egg_tart" and r.json()["feedback"]["in_taxonomy"] is True   # 'egg tart' has a space in the checkpoint

    seg = client.get(f"/api/results/{rid}").json()["segments"][0]
    assert seg["yolo_feedback"]["verdict"] == "correct" and seg["feedback"] is None

    s = client.get("/api/evaluation/summary").json()
    assert s["yolo"]["top1_accuracy"] == 0.5 and s["yolo"]["manually_evaluated"] == 2 and s["yolo"]["not_yet_evaluated"] == 0
    band = {b["yolo_confidence"]: b for b in s["yolo"]["accuracy_by_confidence"]}["0.70-1.00"]    # both fake detections have conf 0.9
    assert band["evaluated"] == 2 and band["correct"] == 1 and band["accuracy"] == 0.5
    assert s["top1_accuracy"] is None and s["images_by_mode"] == {"yolo_only": 1, "compare": 0}   # MobileNet stats untouched
    assert s["counts"]["segments_skipped"] == 1 and s["yolo"]["segments_with_yolo_identity"] == 2    # identified != skipped

    assert client.delete(url, params={"model": "yolo"}).json()["feedback"] is None
    assert client.get("/api/evaluation/summary").json()["yolo"]["manually_evaluated"] == 1


def test_mobilenet_feedback_is_refused_where_mobilenet_did_not_run(client):
    rid = _post(client, "yolo_only")[1]["result_id"]
    r = client.put(f"/api/results/{rid}/segments/1/feedback", json={"verdict": "correct"})        # model defaults to mobilenet
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_feedback"
    r = client.put(f"/api/results/{rid}/segments/2/feedback", json={"verdict": "correct", "model": "yolo"})
    assert r.status_code == 422 and "skipped" in r.json()["error"]["message"]                   # skipped segment: no identity to judge


def test_head_to_head_uses_only_segments_judged_for_both_models(client):
    rid = _post(client, "compare")[1]["result_id"]
    url = f"/api/results/{rid}/segments/{{}}/feedback"
    put = lambda seg, model, verdict: client.put(url.format(seg), json={"verdict": verdict, "model": model}).status_code
    assert [put(1, "yolo", "correct"), put(1, "mobilenet", "correct"),            # both right
            put(3, "yolo", "correct"), put(3, "mobilenet", "incorrect")] == [200] * 4   # only YOLO right
    s = client.get("/api/evaluation/summary").json()
    assert s["head_to_head"] == {"segments_judged_for_both": 2, "both_correct": 1, "yolo_only_correct": 1,
                                 "mobilenet_only_correct": 0, "both_incorrect": 0, "yolo_accuracy": 1.0, "mobilenet_accuracy": 0.5}
    assert s["top1_accuracy"] == 0.5 and s["yolo"]["top1_accuracy"] == 1.0
    # a segment judged for one model only must not enter the head-to-head
    rid2 = _post(client, "compare")[1]["result_id"]
    client.put(f"/api/results/{rid2}/segments/1/feedback", json={"verdict": "incorrect", "model": "yolo"})
    assert client.get("/api/evaluation/summary").json()["head_to_head"]["segments_judged_for_both"] == 2


def test_make_feedback_yolo_rules():
    seg = SegmentRecord(1, 3, "egg tart", 0.8, [0, 0, 9, 9], [0, 0, 9, 9], [1, 2, 3], [9, 9], 80, 0.9, "identified")
    names = [n for k, n in sorted(YOLO_NAMES.items()) if k >= 0]
    assert make_feedback(seg, "correct", None, names, "yolo").verdict == "correct"
    fb = make_feedback(seg, "incorrect", "Red Thing", names, "yolo")
    assert fb.actual_label == "red_thing" and fb.in_taxonomy is True                   # 'red-thing', 'Red Thing' -> red_thing
    assert make_feedback(seg, "incorrect", "broccoli", names, "yolo").in_taxonomy is False
    with pytest.raises(FeedbackError, match="YOLO class"):
        make_feedback(seg, "incorrect", "Egg Tart", names, "yolo")                       # equals the prediction
    with pytest.raises(FeedbackError):
        make_feedback(seg, "correct", None, names, "resnet")
    with pytest.raises(FeedbackError):                                                   # MobileNet verdict needs a MobileNet prediction
        make_feedback(seg, "correct", None, names, "mobilenet")
    seg.status = "skipped"
    with pytest.raises(FeedbackError):
        make_feedback(seg, "correct", None, names, "yolo")
