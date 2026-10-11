"""Classifier preprocessing, copied from the 'val' pipeline of mobilenet_pipeline/scripts/train_mobnet.py.

train_mobnet.py cannot be imported without side effects (it creates ../models and pulls in sklearn/tqdm), so the
definitions are reproduced here verbatim. tests/test_preprocessing.py imports the original script and asserts that
this module produces bit-identical tensors, so any drift is caught.

Original (train_mobnet.py):
    IMAGE_SIZE = 224
    'val': Compose([LetterboxPad(fill=0), Resize((224, 224)), ToTensor(),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
No random augmentation is applied here.
"""
from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import functional as F

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LetterboxPad:
    """Aspect-ratio preserving square padding (identical to the training script). Pads with black (fill=0)."""

    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        hp = (max_dim - w) // 2
        vp = (max_dim - h) // 2
        return F.pad(img, (hp, vp, max_dim - w - hp, max_dim - h - vp), fill=self.fill)


def build_letterbox_resize() -> transforms.Compose:
    """PIL RGB -> 224x224 PIL RGB (first two stages of the val transform). Used for the debug preview too."""
    return transforms.Compose([LetterboxPad(fill=0), transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))])


def build_tensor_stage() -> transforms.Compose:
    """224x224 PIL RGB -> normalized float32 CHW tensor (last two stages of the val transform)."""
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def build_val_transform() -> transforms.Compose:
    """The complete training-time validation transform."""
    return transforms.Compose([LetterboxPad(fill=0), transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                               transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
