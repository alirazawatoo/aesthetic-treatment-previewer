"""Bounded decoding of an uploaded photo into the RGB array the pipeline uses."""

import io

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
MAX_SIDE = 1280
ALLOWED_FORMATS = frozenset({"JPEG", "PNG"})


class UploadError(ValueError):
    """The upload cannot be used; the message is safe to show to the user."""


def resize_max(image_rgb: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        return cv2.resize(image_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image_rgb


def decode_upload(data: bytes) -> np.ndarray:
    """Decode JPEG/PNG bytes, apply EXIF orientation and cap the longest side.

    Size and pixel limits are checked before the pixel data is decoded, so an
    oversized or decompression-bomb image never reaches memory as an array.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError("Choose a JPEG or PNG no larger than 10 MiB.")
    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ALLOWED_FORMATS:
            raise UploadError("Only JPEG and PNG images are supported.")
        if img.width * img.height > MAX_PIXELS:
            raise UploadError("Choose an image with at most 20 million pixels.")
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise UploadError("This file could not be decoded as a JPEG or PNG image.") from e

    try:
        img = ImageOps.exif_transpose(img)
    except (OSError, ValueError):
        pass  # A malformed EXIF block should not block an otherwise valid image.
    return resize_max(np.array(img.convert("RGB")))
