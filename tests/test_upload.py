import io

import numpy as np
import pytest
from PIL import Image

from src import upload


def _encode(img: Image.Image, fmt: str, **kwargs) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def test_png_decodes_to_rgb_uint8():
    data = _encode(Image.new("RGBA", (40, 30), (10, 20, 30, 255)), "PNG")
    out = upload.decode_upload(data)
    assert out.shape == (30, 40, 3)
    assert out.dtype == np.uint8
    assert tuple(out[0, 0]) == (10, 20, 30)


def test_long_side_is_capped():
    data = _encode(Image.new("RGB", (2560, 1000)), "JPEG")
    out = upload.decode_upload(data)
    assert max(out.shape[:2]) == upload.MAX_SIDE
    assert out.shape[:2] == (500, 1280)


def test_exif_orientation_is_applied():
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise for display
    data = _encode(Image.new("RGB", (60, 20)), "JPEG", exif=exif)
    assert upload.decode_upload(data).shape[:2] == (60, 20)


def test_other_formats_are_rejected():
    data = _encode(Image.new("RGB", (8, 8)), "GIF")
    with pytest.raises(upload.UploadError, match="Only JPEG and PNG"):
        upload.decode_upload(data)


def test_garbage_bytes_are_rejected():
    with pytest.raises(upload.UploadError, match="could not be decoded"):
        upload.decode_upload(b"not an image at all")


def test_truncated_png_is_rejected():
    data = _encode(Image.new("RGB", (64, 64), (200, 0, 0)), "PNG")
    with pytest.raises(upload.UploadError):
        upload.decode_upload(data[: len(data) // 2])


def test_oversized_files_are_rejected_before_decoding():
    with pytest.raises(upload.UploadError, match="10 MiB"):
        upload.decode_upload(b"\0" * (upload.MAX_UPLOAD_BYTES + 1))


def test_pixel_limit_is_checked_from_the_header(monkeypatch):
    monkeypatch.setattr(upload, "MAX_PIXELS", 100)
    data = _encode(Image.new("RGB", (20, 20)), "PNG")
    with pytest.raises(upload.UploadError, match="million pixels"):
        upload.decode_upload(data)
