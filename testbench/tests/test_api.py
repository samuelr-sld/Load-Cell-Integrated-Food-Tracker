"""HTTP contract tests. The real FastAPI app, store and orchestrator run; only the two networks are fakes."""
import io
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from testbench.main import create_app
from testbench.model_loader import ModelLoadError
from .conftest import color_scene, jpeg_bytes, make_pipeline


@pytest.fixture
def client():
    img, dets = color_scene()
    pipe, seg, cls = make_pipeline(detections=dets)
    with TestClient(create_app(pipeline=pipe)) as c:
        c.scene = jpeg_bytes(img, 100)
        c.seg, c.cls = seg, cls
        yield c


def _post(client, *files):
    return client.post("/api/inference", files=[("files", f) for f in files])


def _upload_one(client, name="meal.jpg"):
    r = _post(client, (name, client.scene, "image/jpeg"))
    assert r.status_code == 200, r.text
    return r.json()["items"][0]


def test_health_reports_models_device_and_limits(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["models_loaded"] is True and h["device"]["resolved"] == "cpu"
    assert h["limits"]["accepted_extensions"] == [".jpg", ".jpeg", ".png"]


def test_health_and_inference_when_models_failed_to_load(monkeypatch):
    monkeypatch.setattr("testbench.main.load_models", lambda s: (_ for _ in ()).throw(ModelLoadError("checkpoint missing")))
    with TestClient(create_app()) as c:
        h = c.get("/api/health").json()
        assert h["status"] == "error" and h["models_loaded"] is False and "checkpoint missing" in h["load_error"]
        r = _post(c, ("a.jpg", jpeg_bytes(np.zeros((20, 20, 3), np.uint8)), "image/jpeg"))
        assert r.status_code == 503 and r.json()["error"]["code"] == "models_not_loaded"


def test_inference_returns_result_id_and_structured_predictions(client):
    item = _upload_one(client)
    assert item["status"] == "completed" and item["error"] is None
    res = item["result"]
    assert res["result_id"] == item["result_id"] and res["filename"] == "meal.jpg"
    assert [s["status"] for s in res["segments"]] == ["classified", "skipped", "classified"]
    top1 = res["segments"][0]["top1"]
    assert top1["label"] == "red" and len(res["segments"][0]["predictions"]) == 5
    assert res["timings_ms"]["num_segments"] == 3 and "score_note" in res and res["device"] == "cpu"


def test_multiple_files_are_processed_independently(client):
    r = _post(client, ("a.jpg", client.scene, "image/jpeg"), ("b.jpg", client.scene, "image/jpeg"))
    body = r.json()
    assert r.status_code == 200 and body["summary"] == {"received": 2, "completed": 2, "failed": 0, "rejected": 0}
    assert body["items"][0]["result_id"] != body["items"][1]["result_id"]
    assert len(client.get("/api/results").json()["results"]) == 2


def test_list_detail_and_newest_first(client):
    a, b = _upload_one(client, "first.jpg"), _upload_one(client, "second.jpg")
    results = client.get("/api/results").json()["results"]
    assert [r["filename"] for r in results] == ["second.jpg", "first.jpg"]
    d = client.get(f"/api/results/{a['result_id']}").json()
    assert d["filename"] == "first.jpg" and d["image_urls"]["original"].endswith("/images/original")
    assert d["segments"][0]["image_urls"]["masked"].endswith("/images/seg1_masked")


def test_images_are_served_from_memory_as_jpeg(client):
    item = _upload_one(client)
    rid = item["result_id"]
    for key in ("original", "overlay", "seg1_masked", "seg1_bbox", "seg1_input", "seg3_masked"):
        r = client.get(f"/api/results/{rid}/images/{key}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert Image.open(io.BytesIO(r.content)).format == "JPEG"
    assert Image.open(io.BytesIO(client.get(f"/api/results/{rid}/images/seg1_input").content)).size == (224, 224)
    assert client.get(f"/api/results/{rid}/images/seg2_masked").status_code == 404      # skipped segment has no crop
    miss = client.get(f"/api/results/{rid}/images/nope")
    assert miss.status_code == 404 and miss.json()["error"]["code"] == "image_not_found"


def test_unknown_result_is_404_with_error_envelope(client):
    r = client.get("/api/results/doesnotexist")
    assert r.status_code == 404 and r.json()["error"]["code"] == "result_not_found"
    assert client.get("/api/results/doesnotexist/images/original").status_code == 404
    assert client.get("/api/no-such-route").json()["error"]["code"] == "http_error"


def test_delete_clears_everything(client):
    rid = _upload_one(client)["result_id"]
    _upload_one(client)
    r = client.delete("/api/results")
    assert r.status_code == 200 and r.json() == {"cleared": 2}
    assert client.get("/api/results").json()["results"] == []
    assert client.get(f"/api/results/{rid}").status_code == 404
    assert client.delete("/api/results").json() == {"cleared": 0}


def test_missing_files_field_is_422_envelope(client):
    r = client.post("/api/inference")
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request" and "files" in r.json()["error"]["message"]


def test_rejections_mixed_and_all_invalid(client):
    gif = io.BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).convert("P").save(gif, "GIF")
    r = _post(client, ("ok.jpg", client.scene, "image/jpeg"), ("anim.gif", gif.getvalue(), "image/gif"),
              ("bad.png", b"not a png", "image/png"))
    body = r.json()
    assert r.status_code == 200
    assert [i["status"] for i in body["items"]] == ["completed", "rejected", "rejected"]
    assert body["items"][1]["error"]["code"] == "unsupported_extension" and body["items"][2]["error"]["code"] == "invalid_image"
    assert len(client.get("/api/results").json()["results"]) == 1          # rejected uploads are not stored

    r = _post(client, ("anim.gif", gif.getvalue(), "image/gif"))
    assert r.status_code == 415 and r.json()["summary"]["rejected"] == 1
    r = _post(client, ("bad.jpg", b"garbage", "image/jpeg"))
    assert r.status_code == 422


def test_size_and_count_limits(small_limits):
    img, dets = color_scene()
    pipe, _, _ = make_pipeline(detections=dets, settings=small_limits)
    big = jpeg_bytes(np.random.default_rng(0).integers(0, 255, (600, 600, 3), dtype=np.uint8), 100)
    assert len(big) > small_limits.max_upload_bytes
    with TestClient(create_app(small_limits, pipe)) as c:
        r = _post(c, ("big.jpg", big, "image/jpeg"))
        assert r.status_code == 413 and r.json()["items"][0]["error"]["code"] == "file_too_large"
        r = _post(c, *[(f"{i}.jpg", jpeg_bytes(img), "image/jpeg") for i in range(4)])
        assert r.status_code == 413 and r.json()["error"]["code"] == "too_many_files"
        # a body far beyond files*limit is refused before multipart parsing
        huge = b"\0" * (small_limits.max_files_per_request * small_limits.max_upload_bytes + 2 * 1024 * 1024)
        r = _post(c, ("huge.jpg", huge, "image/jpeg"))
        assert r.status_code == 413 and r.json()["error"]["code"] == "request_too_large"


def test_inference_failure_returns_500_item_without_stack_trace():
    pipe, _, _ = make_pipeline(error=RuntimeError("secret C:\\internal\\path"))
    with TestClient(create_app(pipeline=pipe)) as c:
        r = _post(c, ("a.jpg", jpeg_bytes(np.zeros((30, 30, 3), np.uint8)), "image/jpeg"))
        assert r.status_code == 500
        assert "secret" not in r.text and "Traceback" not in r.text
        assert r.json()["items"][0]["status"] == "failed"
        assert c.get("/api/results").json()["results"] == []             # failed runs are not retained


def test_unhandled_exception_hides_internals(client, monkeypatch):
    monkeypatch.setattr(client.app.state.store, "list", lambda: (_ for _ in ()).throw(KeyError("secret internals")))
    r = TestClient(client.app, raise_server_exceptions=False).get("/api/results")
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal_error" and "secret" not in r.text


def test_store_evicts_oldest_beyond_result_limit(small_limits):          # max_results = 2
    img, dets = color_scene()
    pipe, _, _ = make_pipeline(detections=dets, settings=small_limits)
    with TestClient(create_app(small_limits, pipe)) as c:
        ids = [_post(c, (f"{i}.jpg", jpeg_bytes(img), "image/jpeg")).json()["items"][0]["result_id"] for i in range(3)]
        assert c.get(f"/api/results/{ids[0]}").status_code == 404
        assert c.get(f"/api/results/{ids[1]}").status_code == 200 and c.get(f"/api/results/{ids[2]}").status_code == 200
        assert c.get("/api/results").json()["store"]["evicted_total"] == 1
        assert c.get("/api/evaluation/summary").json()["results_evicted_from_memory"] == 1


def test_store_enforces_memory_budget(small_limits):
    img, dets = color_scene()
    roomy = replace(small_limits, max_results=50, max_store_bytes=512 * 2**20)
    pipe, _, _ = make_pipeline(detections=dets, settings=roomy)
    with TestClient(create_app(roomy, pipe)) as c:
        _post(c, ("probe.jpg", jpeg_bytes(img), "image/jpeg"))
        one = c.get("/api/results").json()["store"]["bytes"]          # measured size of one record

    tight = replace(roomy, max_store_bytes=int(one * 2.5))             # room for two records, not three
    pipe, _, _ = make_pipeline(detections=dets, settings=tight)
    with TestClient(create_app(tight, pipe)) as c:
        for i in range(6):
            assert _post(c, (f"{i}.jpg", jpeg_bytes(img), "image/jpeg")).status_code == 200
        st = c.get("/api/results").json()["store"]
        assert st["bytes"] <= tight.max_store_bytes and st["count"] == 2 and st["evicted_total"] == 4


# --------------------------------------------------------------- manual evaluation endpoints
def test_feedback_roundtrip_and_summary(client):
    rid = _upload_one(client)["result_id"]
    url = lambda seg: f"/api/results/{rid}/segments/{seg}/feedback"
    assert client.get("/api/evaluation/summary").json()["top1_accuracy"] is None

    assert client.put(url(1), json={"verdict": "correct"}).json()["feedback"]["verdict"] == "correct"
    r = client.put(url(3), json={"verdict": "incorrect", "actual_label": "Filler 3"})
    assert r.status_code == 200 and r.json()["feedback"]["actual_label"] == "filler_3" and r.json()["feedback"]["in_taxonomy"] is True

    s = client.get("/api/evaluation/summary").json()
    assert s["top1_accuracy"] == 0.5 and s["counts"]["segments_manually_evaluated"] == 2
    assert s["top5_denominator"] == 2 and s["top5_accuracy"] == 0.5          # 'filler_3' is not in the fake top-5
    detail = client.get(f"/api/results/{rid}").json()
    assert detail["segments"][0]["feedback"]["verdict"] == "correct" and detail["num_evaluated"] == 2

    assert client.delete(url(1)).json()["feedback"] is None
    assert client.get("/api/evaluation/summary").json()["counts"]["segments_manually_evaluated"] == 1


def test_feedback_validation_errors(client):
    rid = _upload_one(client)["result_id"]
    base = f"/api/results/{rid}/segments"
    assert client.put(f"{base}/1/feedback", json={"verdict": "maybe"}).status_code == 422
    assert client.put(f"{base}/1/feedback", json={"verdict": "correct", "actual_label": "x"}).json()["error"]["code"] == "invalid_feedback"
    assert client.put(f"{base}/1/feedback", json={"verdict": "incorrect", "actual_label": "red"}).status_code == 422
    assert client.put(f"{base}/2/feedback", json={"verdict": "correct"}).status_code == 422          # skipped segment
    assert client.put(f"{base}/9/feedback", json={"verdict": "correct"}).json()["error"]["code"] == "segment_not_found"
    assert client.put(f"/api/results/nope/segments/1/feedback", json={"verdict": "correct"}).status_code == 404
