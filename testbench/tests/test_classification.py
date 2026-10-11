import json

import pytest
import torch
import torch.nn as nn

from testbench.classification import ClassificationService
from testbench.config import Settings
from testbench.model_loader import load_classifier


class StubModel(nn.Module):
    """Emits fixed logits and records the runtime flags that matter for correct inference."""

    def __init__(self, logits_rows):
        super().__init__()
        self.rows = torch.tensor(logits_rows, dtype=torch.float32)
        self.flags = []
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        self.flags.append((self.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        n = x.shape[0]
        out = self.rows[self._cursor:self._cursor + n]
        self._cursor += n
        return out + 0 * x.flatten(1).sum(dim=1, keepdim=True)

    _cursor = 0


NAMES = [f"c{i}" for i in range(8)]


def test_top5_indices_labels_order_and_softmax_values():
    logits = [[0.1, 3.0, 0.2, 2.0, 0.0, 1.0, 5.0, 0.3]]
    svc = ClassificationService(StubModel(logits), NAMES, torch.device("cpu"))
    preds = svc.classify([torch.zeros(3, 4, 4)])[0]
    assert [p.class_index for p in preds] == [6, 1, 3, 5, 7]
    assert [p.label for p in preds] == ["c6", "c1", "c3", "c5", "c7"]
    assert [p.rank for p in preds] == [1, 2, 3, 4, 5]
    expected = torch.softmax(torch.tensor(logits[0]), 0)
    for p in preds:
        assert p.softmax_score == pytest.approx(float(expected[p.class_index]), rel=1e-5)
    assert all(a.softmax_score >= b.softmax_score for a, b in zip(preds, preds[1:]))
    assert sum(p.softmax_score for p in preds) < 1.0 + 1e-6


def test_batching_preserves_input_order():
    rows = [[float(i == k) * 9 for i in range(8)] for k in (4, 0, 7, 2, 5)]       # row k peaks at class k
    svc = ClassificationService(StubModel(rows), NAMES, torch.device("cpu"), batch_size=2)
    out = svc.classify([torch.zeros(3, 4, 4)] * 5)
    assert [o[0].class_index for o in out] == [4, 0, 7, 2, 5]


def test_runs_in_eval_mode_without_gradients_under_inference_mode():
    model = StubModel([[1.0] * 8])
    model.train()                                       # the service must force eval()
    svc = ClassificationService(model, NAMES, torch.device("cpu"))
    out = svc.classify([torch.zeros(3, 4, 4)])
    assert model.flags == [(False, False, True)]        # (training, grad enabled, inference mode)
    assert not model.training


def test_fewer_than_five_classes_and_empty_input():
    svc = ClassificationService(StubModel([[1.0, 2.0, 3.0]]), ["a", "b", "c"], torch.device("cpu"))
    assert len(svc.classify([torch.zeros(3, 4, 4)])[0]) == 3
    assert svc.classify([]) == []


def test_logit_width_must_match_class_list():
    svc = ClassificationService(StubModel([[1.0, 2.0, 3.0]]), ["a", "b"], torch.device("cpu"))
    with pytest.raises(RuntimeError, match="logits"):
        svc.classify([torch.zeros(3, 4, 4)])


# -------- class-index mapping against the real artifacts --------
@pytest.fixture(scope="module")
def real_classifier():
    s = Settings()
    if not s.mobilenet_weights.is_file():
        pytest.skip("MobileNet checkpoint not present")
    return load_classifier(s.mobilenet_weights, s.class_names_json, torch.device("cpu"))


def test_class_index_mapping_matches_training_artifacts(real_classifier):
    model, names, arch, meta = real_classifier
    s = Settings()
    assert names == json.loads(s.class_names_json.read_text(encoding="utf-8"))
    assert len(names) == meta["num_classes"] == 101 == model.classifier[-1].out_features
    # ImageFolder order (what training used) is sorted folder names; verify the stored list really is that order.
    assert names == sorted(names)
    assert names[0] == "apple_pie" and names[-1] == "waffles"


def test_real_classifier_head_decodes_through_service(real_classifier):
    model, names, _, _ = real_classifier
    svc = ClassificationService(model, names, torch.device("cpu"))
    preds = svc.classify([torch.zeros(3, 224, 224)])[0]
    assert len(preds) == 5 and all(p.label == names[p.class_index] for p in preds)
