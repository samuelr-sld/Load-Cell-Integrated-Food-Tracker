import importlib.util
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from testbench.config import REPO_ROOT
from testbench.preprocessing import (IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, LetterboxPad, build_letterbox_resize,
                                     build_tensor_stage, build_val_transform)

TRAIN_SCRIPT = REPO_ROOT / "mobilenet_pipeline" / "scripts" / "train_mobnet.py"


def _img(w, h, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8), "RGB")


def test_output_shape_dtype_and_range():
    t = build_val_transform()(_img(97, 31))
    assert t.shape == (3, IMAGE_SIZE, IMAGE_SIZE) == (3, 224, 224)
    assert t.dtype == torch.float32


# Literal values from train_mobnet.py (Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])). Deliberately NOT imported
# from the module under test, otherwise a changed constant would change the expectation with it.
TRAIN_MEAN, TRAIN_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def test_normalization_matches_training_configuration():
    assert list(IMAGENET_MEAN) == list(TRAIN_MEAN) and list(IMAGENET_STD) == list(TRAIN_STD)
    for pixel, value in (((255, 255, 255), 1.0), ((0, 0, 0), 0.0)):
        img = Image.new("RGB", (224, 224), pixel)
        expected = torch.tensor([(value - m) / s for m, s in zip(TRAIN_MEAN, TRAIN_STD)]).view(3, 1, 1)
        assert torch.allclose(build_val_transform()(img), expected.expand(3, 224, 224), atol=1e-5)


def test_letterbox_preserves_aspect_ratio_with_black_bars():
    padded = LetterboxPad(fill=0)(Image.new("RGB", (100, 50), (255, 255, 255)))
    assert padded.size == (100, 100)
    a = np.asarray(padded)
    assert (a[:25] == 0).all() and (a[75:] == 0).all() and (a[25:75] == 255).all()
    tall = LetterboxPad(fill=0)(Image.new("RGB", (30, 91), (9, 9, 9)))
    assert tall.size == (91, 91)


def test_pipeline_is_deterministic():
    img = _img(120, 80)
    assert torch.equal(build_val_transform()(img), build_val_transform()(img))


def test_split_stages_equal_full_transform():
    img = _img(120, 80)
    preview = build_letterbox_resize()(img)
    assert preview.size == (224, 224)
    assert torch.equal(build_tensor_stage()(preview), build_val_transform()(img))


@pytest.fixture(scope="module")
def training_module():
    if not TRAIN_SCRIPT.is_file():
        pytest.skip("training script not present")
    import os
    cwd = os.getcwd()
    os.chdir(TRAIN_SCRIPT.parent)   # the script does os.makedirs('../models') at import; run it where that already exists
    try:
        spec = importlib.util.spec_from_file_location("_orig_train_mobnet", TRAIN_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    except ImportError as exc:
        pytest.skip(f"training script dependencies unavailable: {exc}")
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize("size", [(100, 50), (37, 211), (224, 224), (300, 299), (11, 10)])
def test_bit_identical_to_original_training_val_transform(training_module, size):
    """The strongest guard against preprocessing drift: run the ORIGINAL script's val transform and compare."""
    img = _img(*size, seed=size[0])
    reference = training_module.data_transforms["val"](img)
    assert training_module.IMAGE_SIZE == IMAGE_SIZE
    assert torch.equal(build_val_transform()(img), reference)
    assert torch.equal(build_tensor_stage()(build_letterbox_resize()(img)), reference)
