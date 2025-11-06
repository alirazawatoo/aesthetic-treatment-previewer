"""
Procedurally drawn demo face.

Faces are personal data, so the app, tests and screenshots use this drawing
instead of a photo. It is deliberately cartoon-like, yet passes the OpenCV Haar
detector and the blur/lighting checks, so the skin controls can be exercised
without anyone's selfie.
"""

import cv2
import numpy as np

_SKIN = np.array([206, 160, 138], np.float32)
_HAIR = (58, 42, 34)
_BROW = (60, 44, 36)
_LIP_UPPER = (168, 86, 84)
_LIP_LOWER = (178, 96, 92)


def _draw_eye(img: np.ndarray, ex: float, ey: float, fx: float, fy: float) -> None:
    cv2.ellipse(img, (int(ex), int(ey - fy * 0.16)), (int(fx * 0.24), int(fy * 0.04)), 0, 190, 350, _BROW, 6)
    cv2.ellipse(img, (int(ex), int(ey)), (int(fx * 0.17), int(fy * 0.065)), 0, 0, 360, (236, 232, 226), -1)
    cv2.circle(img, (int(ex), int(ey)), int(fy * 0.055), (74, 52, 40), -1)
    cv2.circle(img, (int(ex), int(ey)), int(fy * 0.025), (20, 16, 14), -1)
    cv2.ellipse(img, (int(ex), int(ey)), (int(fx * 0.17), int(fy * 0.065)), 0, 180, 360, (90, 60, 50), 3)
    # Under-eye shadow, so the under-eye control has something to lift.
    cv2.ellipse(img, (int(ex), int(ey + fy * 0.1)), (int(fx * 0.15), int(fy * 0.05)), 0, 0, 180, (170, 120, 104), 4)


def make_demo_face(size: int = 640, seed: int = 7) -> np.ndarray:
    """Return a size x size RGB uint8 drawing of a face with mild skin texture."""
    rng = np.random.default_rng(seed)
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    img = np.dstack([np.full((h, w), c, np.float32) for c in (214, 222, 220)])
    img *= (0.92 + 0.08 * (yy / h))[..., None]

    cx, cy = w / 2, h * 0.52
    fx, fy = w * 0.27, h * 0.36
    cv2.ellipse(img, (int(cx), h + int(h * 0.05)), (int(w * 0.42), int(h * 0.2)), 0, 180, 360, (70, 86, 96), -1)
    cv2.ellipse(img, (int(cx), int(cy - fy * 0.18)), (int(fx * 1.12), int(fy * 1.0)), 0, 0, 360, _HAIR, -1)

    d = ((xx - cx) / fx) ** 2 + ((yy - cy) / fy) ** 2
    face = d <= 1
    shade = np.clip(1.06 - 0.22 * d - 0.06 * (xx - cx) / fx, 0.7, 1.1)
    img[face] = (_SKIN * shade[..., None])[face]

    _draw_eye(img, cx - fx * 0.42, cy - fy * 0.12, fx, fy)
    _draw_eye(img, cx + fx * 0.42, cy - fy * 0.12, fx, fy)

    cv2.line(img, (int(cx + fx * 0.03), int(cy - fy * 0.05)), (int(cx + fx * 0.06), int(cy + fy * 0.2)), (170, 122, 102), 4)
    cv2.ellipse(img, (int(cx), int(cy + fy * 0.24)), (int(fx * 0.14), int(fy * 0.045)), 0, 0, 180, (150, 104, 88), 4)

    my = cy + fy * 0.48
    cv2.ellipse(img, (int(cx), int(my)), (int(fx * 0.3), int(fy * 0.07)), 0, 180, 360, _LIP_UPPER, -1)
    cv2.ellipse(img, (int(cx), int(my)), (int(fx * 0.3), int(fy * 0.09)), 0, 0, 180, _LIP_LOWER, -1)
    cv2.line(img, (int(cx - fx * 0.3), int(my)), (int(cx + fx * 0.3), int(my)), (110, 50, 50), 3)

    # Grain and darker spots give the smoothing and pigment controls visible work.
    img += rng.normal(0, 6, img.shape).astype(np.float32) * face[..., None]
    for _ in range(40):
        px, py = rng.normal(cx, fx * 0.5), rng.normal(cy + fy * 0.1, fy * 0.35)
        if ((px - cx) / fx) ** 2 + ((py - cy) / fy) ** 2 < 0.8:
            cv2.circle(img, (int(px), int(py)), int(rng.integers(2, 5)), (170, 118, 100), -1)
    img = cv2.GaussianBlur(np.clip(img, 0, 255), (0, 0), 0.6)
    img += rng.normal(0, 4, img.shape).astype(np.float32) * face[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)
