"""Live integration: real YOLO11s-seg + real MobileNet checkpoint + real photographs. No mocks anywhere in this file.

Run only these with:  python -m pytest -c testbench/pytest.ini testbench/tests/test_live_integration.py -s
"""
from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from testbench.config import REPO_ROOT, Settings
from testbench.crops import CropProcessor
from testbench.ingest import decode_image
from testbench.main import create_app
from testbench.model_loader import load_models
from testbench.orchestrator import InferencePipeline
from testbench.scripts.classifier_sanity_check import VAL_DIR, run as classifier_sanity

pytestmark = pytest.mark.live
S = Settings()
IMAGES = sorted((REPO_ROOT / "yolo_pipeline" / "data" / "images" / "val").glob("*.jpg"))[:3]

if not (S.yolo_weights.is_file() and S.mobilenet_weights.is_file() and len(IMAGES) == 3):
    pytest.skip("real checkpoints or sample images are not present", allow_module_level=True)


@pytest.fixture(scope="module")
def models():
    return load_models(S)


@pytest.fixture(scope="module")
def pipeline(models):
    return InferencePipeline.from_models(models, S)


@pytest.fixture(scope="module")
def client(pipeline):
    with TestClient(create_app(S, pipeline)) as c:
        yield c


def _polygon_mask(poly, h, w):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m.astype(bool)


# ------------------------------------------------------------------ end to end through HTTP
def test_real_image_through_http_with_both_checkpoints(client, models):
    path = IMAGES[0]
    r = client.post("/api/inference", files=[("files", (path.name, path.read_bytes(), "image/jpeg"))])
    assert r.status_code == 200, r.text
    item = r.json()["items"][0]
    assert item["status"] == "completed"
    res = item["result"]
    names = set(models.class_names)
    classified = [s for s in res["segments"] if s["status"] == "classified"]
    assert res["timings_ms"]["num_segments"] >= 1 and classified, "YOLO should find food in a FoodSeg103 meal photo"
    assert res["device"] == str(models.device)
    for s in classified:
        p = s["predictions"]
        assert len(p) == 5 and [x["rank"] for x in p] == [1, 2, 3, 4, 5]
        scores = [x["softmax_score"] for x in p]
        assert scores == sorted(scores, reverse=True) and 0 < sum(scores) <= 1.0 + 1e-4
        assert all(x["label"] in names and models.class_names[x["class_index"]] == x["label"] for x in p)
        assert s["yolo_class_name"] in models.yolo_names.values()
    t = res["timings_ms"]
    assert t["yolo_ms"] > 0 and t["classify_ms"] > 0 and t["crop_preprocess_ms"] > 0 and t["total_ms"] >= t["inference_ms"]
    # every visual output is retrievable from memory
    rid = item["result_id"]
    for key in ["original", "overlay", *classified[0]["image_urls"].values()]:
        url = key if key.startswith("/api") else f"/api/results/{rid}/images/{key}"
        img = client.get(url)
        assert img.status_code == 200 and Image.open(io.BytesIO(img.content)).format == "JPEG"
    print(f"\n[live] {path.name}: {len(res['segments'])} segments on {res['device']}, timings {t}")


def test_blank_image_gives_empty_result_and_classifier_is_skipped(client):
    blank = io.BytesIO()
    Image.fromarray(np.full((320, 320, 3), 200, np.uint8)).save(blank, "JPEG")
    r = client.post("/api/inference", files=[("files", ("blank.jpg", blank.getvalue(), "image/jpeg"))])
    res = r.json()["items"][0]["result"]
    assert r.status_code == 200 and res["segments"] == [] and res["timings_ms"]["classify_ms_per_segment"] is None


# ------------------------------------------------------------------ the critical compatibility checks (section 6)
@pytest.mark.parametrize("path", IMAGES, ids=lambda p: p.name)
def test_array_input_matches_file_path_input_so_channel_order_is_right(models, path):
    """Ultralytics given a file path reads with cv2 (BGR) - the way the user already validated YOLO. The testbench feeds a
    numpy array; its detections must match, which proves the RGB->BGR handling."""
    from testbench.segmentation import SegmentationService
    ref = models.yolo.predict(str(path), device=str(models.device), conf=S.yolo_conf, retina_masks=True, verbose=False)[0]
    decoded = decode_image(path.read_bytes(), path.name, S)
    mine = SegmentationService(models.yolo, models.yolo_names, models.device, S).segment(decoded.rgb).detections

    strong = [i for i in range(len(ref.boxes)) if float(ref.boxes.conf[i]) >= 0.45]     # ignore near-threshold flips from JPEG decoder differences
    assert strong
    for i in strong:
        rb, rc = ref.boxes.xyxy[i].cpu().numpy(), int(ref.boxes.cls[i])
        match = [d for d in mine if d.class_id == rc and np.abs(np.array(d.box_xyxy) - rb).max() < 4.0]
        assert match, f"reference detection {i} (class {rc}) missing from array-input run"
        assert abs(match[0].confidence - float(ref.boxes.conf[i])) < 0.05


@pytest.mark.parametrize("path", IMAGES, ids=lambda p: p.name)
def test_masks_are_aligned_with_original_image_regions(models, path):
    """Retina masks must sit exactly where Ultralytics' own original-coordinate polygons are, and inside their boxes."""
    decoded = decode_image(path.read_bytes(), path.name, S)
    h, w = decoded.rgb.shape[:2]
    r = models.yolo.predict(np.ascontiguousarray(decoded.rgb[:, :, ::-1]), device=str(models.device), conf=S.yolo_conf,
                            retina_masks=True, verbose=False)[0]
    assert r.masks.data.shape[1:] == (h, w)                                  # no letterbox-space masks reach the crop stage
    for i in range(len(r.boxes)):
        mask = (r.masks.data[i] > 0.5).cpu().numpy()
        poly = _polygon_mask(r.masks.xy[i], h, w)
        inter, union = (mask & poly).sum(), (mask | poly).sum()
        assert inter / union > 0.9, f"segment {i}: mask/polygon IoU {inter / union:.3f}"
        ys, xs = np.where(mask)
        x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
        assert xs.min() >= x1 - 1.5 and xs.max() <= x2 + 1.5 and ys.min() >= y1 - 1.5 and ys.max() <= y2 + 1.5


def test_crops_come_from_original_pixels_for_every_real_instance(pipeline):
    decoded = decode_image(IMAGES[1].read_bytes(), IMAGES[1].name, S)
    seg = pipeline.segmenter.segment(decoded.rgb)
    assert seg.detections
    cropper = CropProcessor(S.min_mask_pixels, S.min_crop_side)
    for det in seg.detections:
        c = cropper.crop(decoded.rgb, det)
        if c.status != "ok":
            continue
        x1, y1, x2, y2 = c.box_px
        region, mask = decoded.rgb[y1:y2, x1:x2], det.mask[y1:y2, x1:x2]
        out = np.asarray(c.masked_crop)
        assert (out[mask] == region[mask]).all() and (out[~mask] == 0).all()
        assert np.array_equal(np.asarray(c.bbox_crop), region)


def test_every_prediction_maps_to_its_own_segment(pipeline):
    """Classify each masked crop on its own (batch of 1) and compare with the pipeline's batched, id-associated output."""
    data = IMAGES[2].read_bytes()
    rec = pipeline.process_upload(data, IMAGES[2].name)
    decoded = decode_image(data, IMAGES[2].name, S)
    seg = pipeline.segmenter.segment(decoded.rgb)
    cropper = pipeline.cropper
    for s in rec.segments:
        if s.status != "classified":
            continue
        det = seg.detections[s.id - 1]
        assert det.class_name == s.yolo_class_name
        crop = cropper.crop(decoded.rgb, det)
        solo = pipeline.classifier.classify([cropper.to_tensor(cropper.model_input_preview(crop.masked_crop))])[0]
        # Batched vs single-crop scores differ by ~1e-3 on Ampere GPUs because cuDNN uses TF32 by default (verified: with
        # TF32 disabled the difference is ~1e-7). A mis-associated crop would differ by orders of magnitude more.
        batched_top = {p["label"]: p["softmax_score"] for p in s.predictions}
        assert solo[0].label in batched_top
        assert solo[0].softmax_score == pytest.approx(batched_top[solo[0].label], abs=0.01)
        assert solo[0].label == s.predictions[0]["label"] or abs(solo[0].softmax_score - s.predictions[0]["softmax_score"]) < 0.01


def test_classifier_state_for_inference(models):
    assert models.classifier.training is False
    assert not any(p.requires_grad for p in models.classifier.parameters())
    assert not models.classifier.classifier[2].training            # dropout inactive
    x = torch.zeros(1, 3, 224, 224, device=models.device)
    with torch.inference_mode():
        a, b = models.classifier(x), models.classifier(x)
    assert torch.equal(a, b) and not a.requires_grad                # deterministic, no autograd graph


def test_classifier_reproduces_validation_accuracy_so_class_mapping_is_right(models):
    """Whole-dish Food-101 validation photos through testbench preprocessing + loader. Wrong class order or wrong
    preprocessing would give ~1% (chance); a faithful integration lands near the checkpoint's recorded val top-1."""
    if not VAL_DIR.is_dir():
        pytest.skip("Food-101 validation folder not present")
    out = classifier_sanity(per_class=4)
    print(f"\n[live] classifier sanity: {out}")
    assert out["images"] == 404
    assert out["top1"] > 0.72 and out["top5"] > 0.90
    assert abs(out["top1"] - out["checkpoint_best_val_top1"]) < 0.10


def test_cpu_fallback_runs_the_same_pipeline_end_to_end():
    cpu = InferencePipeline.from_models(load_models(replace(S, device="cpu")), replace(S, device="cpu"))
    rec = cpu.process_upload(IMAGES[0].read_bytes(), IMAGES[0].name)
    assert rec.status == "completed" and rec.device == "cpu"
    assert any(s.status == "classified" and len(s.predictions) == 5 for s in rec.segments)
    print(f"\n[live] CPU timings: {rec.timings_ms}")
