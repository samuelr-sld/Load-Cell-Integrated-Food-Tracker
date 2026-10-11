"""Upload validation and decoding. Decides what the 'original image' is for every later stage."""
from __future__ import annotations

import io
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings

ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")
_ALLOWED_FORMATS = {"JPEG", "PNG"}


class ImageValidationError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 422):
        super().__init__(message)
        self.code, self.message, self.http_status = code, message, http_status


@dataclass
class DecodedImage:
    rgb: np.ndarray            # H x W x 3 uint8, RGB, EXIF orientation applied
    source_format: str         # "JPEG" | "PNG"
    original_size: tuple[int, int]   # (w, h) as uploaded, after EXIF orientation
    downscaled: bool
    decode_ms: float

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]


def check_filename(filename: str) -> None:
    name = (filename or "").lower()
    if not name.endswith(ALLOWED_EXTENSIONS):
        raise ImageValidationError("unsupported_extension", f"'{filename}': only .jpg, .jpeg and .png files are accepted", 415)


def decode_image(data: bytes, filename: str, settings: Settings) -> DecodedImage:
    """Validate and decode upload bytes. The file extension and Content-Type header are not trusted; the content is."""
    start = time.perf_counter()
    check_filename(filename)
    if not data:
        raise ImageValidationError("empty_file", f"'{filename}' is empty", 422)
    if len(data) > settings.max_upload_bytes:
        raise ImageValidationError("file_too_large", f"'{filename}' exceeds the {settings.max_upload_bytes // 2**20} MB upload limit", 413)

    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt, (w, h) = probe.format, probe.size
            if fmt not in _ALLOWED_FORMATS:
                raise ImageValidationError("unsupported_format", f"'{filename}' is a {fmt or 'unknown'} image; only JPEG and PNG are accepted", 415)
            if w * h > settings.max_image_pixels:
                raise ImageValidationError("image_too_large", f"'{filename}' is {w}x{h}; limit is {settings.max_image_pixels // 1_000_000} megapixels", 413)
            probe.verify()  # structural check; the image must be reopened afterwards
        with Image.open(io.BytesIO(data)) as img:
            img.load()  # full decode catches truncated / corrupt pixel data
            img = ImageOps.exif_transpose(img)  # browsers and cv2 honour EXIF orientation; so must we
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                rgba = img.convert("RGBA")
                flat = Image.new("RGB", rgba.size, (255, 255, 255))  # transparent pixels -> white
                flat.paste(rgba, mask=rgba.getchannel("A"))
                img = flat
            else:
                img = img.convert("RGB")
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageValidationError("invalid_image", f"'{filename}' is not a valid, fully decodable JPEG/PNG image ({type(exc).__name__})", 422) from exc

    original_size = img.size
    downscaled = max(img.size) > settings.max_image_side
    if downscaled:
        scale = settings.max_image_side / max(img.size)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)

    rgb = np.asarray(img, dtype=np.uint8).copy()
    return DecodedImage(rgb, fmt, original_size, downscaled, (time.perf_counter() - start) * 1000.0)
