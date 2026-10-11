import json
from dataclasses import replace

import pytest
import torch

from testbench.config import REPO_ROOT, Settings
from testbench.model_loader import ModelLoadError, load_classifier, load_models, load_yolo, resolve_device

S = Settings()
needs_artifacts = pytest.mark.skipif(not (S.mobilenet_weights.is_file() and S.yolo_weights.is_file()), reason="checkpoints not present")
CPU = torch.device("cpu")


# ------------------------------------------------------------------ device resolution
def test_device_resolution(monkeypatch):
    assert resolve_device("cpu") == (CPU, None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == (CPU, None)                       # nothing was asked for, so nothing to explain
    dev, reason = resolve_device("cuda")
    assert dev == CPU and "is_available() is False" in reason          # explicit request -> fallback is reported
    with pytest.raises(ModelLoadError):
        resolve_device("tpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert resolve_device("auto")[0] == torch.device("cuda:0")
    dev, reason = resolve_device("cuda:3")
    assert dev == CPU and "only 1" in reason


# ------------------------------------------------------------------ classifier checkpoint handling
@needs_artifacts
def test_real_classifier_loads_strictly_in_eval_mode_without_grads():
    model, names, arch, meta = load_classifier(S.mobilenet_weights, S.class_names_json, CPU)
    assert arch == "mobilenet_v3_large"      # NB: the checkpoint is V3-Large; strict loading into V3-Small is impossible
    assert meta["num_classes"] == len(names) == 101
    assert model.training is False
    assert not any(p.requires_grad for p in model.parameters())
    with torch.inference_mode():
        out = model(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, 101) and not out.requires_grad


@needs_artifacts
def test_checkpoint_weights_are_not_modified_by_loading():
    before = S.mobilenet_weights.stat()
    load_classifier(S.mobilenet_weights, S.class_names_json, CPU)
    after = S.mobilenet_weights.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)


def test_missing_checkpoint_is_reported(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        load_classifier(tmp_path / "nope.pt", None, CPU)
    with pytest.raises(ModelLoadError, match="not found"):
        load_yolo(tmp_path / "nope.pt")


def test_checkpoint_without_class_names_is_a_blocker_not_a_guess(tmp_path):
    p = tmp_path / "bad.pt"
    torch.save({"model_state_dict": {}, "num_classes": 101}, p)
    with pytest.raises(ModelLoadError, match="missing required keys"):
        load_classifier(p, None, CPU)


@needs_artifacts
def test_class_list_disagreement_between_checkpoint_and_json_is_rejected(tmp_path):
    names = json.loads(S.class_names_json.read_text(encoding="utf-8"))
    names[0], names[1] = names[1], names[0]                      # same classes, different order -> wrong labels
    other = tmp_path / "class_names.json"
    other.write_text(json.dumps(names), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="disagrees"):
        load_classifier(S.mobilenet_weights, other, CPU)


@needs_artifacts
def test_head_size_mismatch_is_rejected(tmp_path):
    ck = torch.load(S.mobilenet_weights, map_location="cpu", weights_only=True)
    ck["num_classes"] = 100
    ck["class_names"] = ck["class_names"][:100]
    p = tmp_path / "wrong_head.pt"
    torch.save(ck, p)
    with pytest.raises(ModelLoadError, match="head output size"):
        load_classifier(p, None, CPU)


@pytest.mark.skipif(not (REPO_ROOT / "yolo_pipeline" / "yolo26n.pt").is_file(), reason="detection model not present")
def test_non_segmentation_yolo_checkpoint_is_rejected():
    with pytest.raises(ModelLoadError, match="expected 'segment'"):
        load_yolo(REPO_ROOT / "yolo_pipeline" / "yolo26n.pt")


# ------------------------------------------------------------------ full load (real models) + CPU fallback
@needs_artifacts
@pytest.mark.live
def test_both_models_load_on_cpu_and_report_it():
    m = load_models(replace(S, device="cpu"))
    assert m.device == CPU and m.device_fallback_reason is None
    assert m.yolo.task == "segment" and m.classifier.training is False
    assert m.info["device"]["resolved"] == "cpu" and m.info["classifier"]["architecture"] == "mobilenet_v3_large"
    assert len(m.class_names) == 101 and m.yolo_names[0] == "candy" and m.yolo_names[102] == "other ingredients"


@needs_artifacts
@pytest.mark.live
def test_cuda_request_without_cuda_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    m = load_models(replace(S, device="cuda"))
    assert m.device == CPU and "is_available() is False" in m.device_fallback_reason
    assert m.info["device"]["fallback_reason"] == m.device_fallback_reason


@needs_artifacts
@pytest.mark.live
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_both_models_load_on_cuda():
    m = load_models(replace(S, device="cuda"))
    assert m.device.type == "cuda" and m.device_fallback_reason is None
    assert next(m.classifier.parameters()).is_cuda
    assert m.info["device"]["gpu_name"]
