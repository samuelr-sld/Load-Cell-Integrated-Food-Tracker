import io
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from testbench.ingest import ImageValidationError, decode_image
from .conftest import jpeg_bytes, png_bytes


def _rgb(h=40, w=60, value=100):
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_valid_jpeg_and_png_decode_to_rgb_uint8(settings):
    for data, name, fmt in [(jpeg_bytes(_rgb()), "a.jpg", "JPEG"), (jpeg_bytes(_rgb()), "a.JPEG", "JPEG"), (png_bytes(_rgb()), "b.png", "PNG")]:
        d = decode_image(data, name, settings)
        assert d.rgb.shape == (40, 60, 3) and d.rgb.dtype == np.uint8
        assert d.source_format == fmt and not d.downscaled


def test_grayscale_and_palette_inputs_become_three_channel(settings):
    gray = io.BytesIO()
    Image.fromarray(np.full((20, 30), 90, np.uint8), "L").save(gray, "JPEG")
    assert decode_image(gray.getvalue(), "g.jpg", settings).rgb.shape == (20, 30, 3)
    pal = io.BytesIO()
    Image.fromarray(_rgb(20, 30)).convert("P").save(pal, "PNG")
    assert decode_image(pal.getvalue(), "p.png", settings).rgb.shape == (20, 30, 3)


def test_transparent_png_is_flattened_onto_white(settings):
    rgba = np.zeros((10, 10, 4), np.uint8)   # fully transparent black
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    assert (decode_image(buf.getvalue(), "t.png", settings).rgb == 255).all()


def test_exif_orientation_is_applied(settings):
    img = Image.fromarray(_rgb(40, 60))
    exif = Image.Exif()
    exif[0x0112] = 6   # rotate 90 CW when displayed
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif.tobytes())
    d = decode_image(buf.getvalue(), "rot.jpg", settings)
    assert d.rgb.shape[:2] == (60, 40)
    assert d.original_size == (40, 60)


@pytest.mark.parametrize("name,code,status", [("notes.txt", "unsupported_extension", 415), ("anim.gif", "unsupported_extension", 415),
                                              ("noext", "unsupported_extension", 415)])
def test_bad_extension_rejected(settings, name, code, status):
    with pytest.raises(ImageValidationError) as e:
        decode_image(jpeg_bytes(_rgb()), name, settings)
    assert (e.value.code, e.value.http_status) == (code, status)


def test_content_is_checked_not_the_extension(settings):
    gif = io.BytesIO()
    Image.fromarray(_rgb()).convert("P").save(gif, "GIF")
    with pytest.raises(ImageValidationError) as e:
        decode_image(gif.getvalue(), "disguised.jpg", settings)
    assert e.value.code == "unsupported_format" and e.value.http_status == 415


def test_empty_corrupt_and_truncated_files_rejected(settings):
    with pytest.raises(ImageValidationError) as e:
        decode_image(b"", "x.jpg", settings)
    assert e.value.code == "empty_file"
    with pytest.raises(ImageValidationError) as e:
        decode_image(b"this is not an image", "x.jpg", settings)
    assert e.value.code == "invalid_image"
    good = jpeg_bytes(np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8))
    with pytest.raises(ImageValidationError) as e:
        decode_image(good[: len(good) // 2], "cut.jpg", settings)
    assert e.value.code == "invalid_image"


def test_size_limits(settings):
    s = replace(settings, max_upload_bytes=1000)
    with pytest.raises(ImageValidationError) as e:
        decode_image(jpeg_bytes(np.random.default_rng(1).integers(0, 255, (100, 100, 3), dtype=np.uint8)), "big.jpg", s)
    assert (e.value.code, e.value.http_status) == ("file_too_large", 413)
    s = replace(settings, max_image_pixels=1000)
    with pytest.raises(ImageValidationError) as e:
        decode_image(jpeg_bytes(_rgb(100, 100)), "px.jpg", s)
    assert (e.value.code, e.value.http_status) == ("image_too_large", 413)


def test_large_images_are_downscaled_once_and_flagged(settings):
    s = replace(settings, max_image_side=100)
    d = decode_image(jpeg_bytes(_rgb(200, 400)), "wide.jpg", s)
    assert d.downscaled and d.original_size == (400, 200) and d.rgb.shape[:2] == (50, 100)
