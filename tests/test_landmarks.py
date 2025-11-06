"""Landmark path, run with requirements-landmarks.txt; skipped without MediaPipe.

MediaPipe FaceMesh finds a full 478-point mesh on the drawn demo face, so the
lip and under-eye effects are tested on real detector output, not on a photo.
"""

import cv2
import numpy as np
import pytest

pytest.importorskip("mediapipe")

from src import effects, face_utils, segmentation  # noqa: E402
from src.demo_face import make_demo_face  # noqa: E402


@pytest.fixture(scope="module")
def demo():
    image = make_demo_face()
    face = face_utils.detect_face_landmarks(image)
    assert face is not None and face["landmarks_px"] is not None
    return image, face


def test_mesh_covers_the_drawn_face_upright(demo):
    image, face = demo
    assert face["landmarks_px"].shape == (478, 2)
    x, y, w, h = face["bbox"]
    assert x >= 0 and y >= 0 and x + w <= image.shape[1] and y + h <= image.shape[0]
    assert abs(face["roll_deg"]) < 5


def test_lip_and_under_eye_regions_are_found(demo):
    image, face = demo
    regions = segmentation.segment_regions(image, face["landmarks_px"])
    assert set(regions) == {"lip_ring", "under_eye"}
    lip_ys = np.nonzero(regions["lip_ring"])[0]
    eye_ys = np.nonzero(regions["under_eye"])[0]
    assert eye_ys.mean() < lip_ys.mean(), "under-eye region must sit above the lips"


def test_lip_filler_only_changes_the_mouth_area(demo):
    image, face = demo
    out = effects.lip_filler(image, face["landmarks_px"], 80)
    changed = np.any(out != image, axis=2)
    assert changed.any()
    lips = face_utils.lip_points_from_landmarks(face["landmarks_px"])
    y0, y1 = lips[:, 1].min(), lips[:, 1].max()
    rows = np.nonzero(changed)[0]
    pad = (y1 - y0) * 1.5
    assert rows.min() >= y0 - pad and rows.max() <= y1 + pad


def test_under_eye_reduction_lightens_the_under_eye_region(demo):
    image, face = demo
    mask = segmentation.segment_regions(image, face["landmarks_px"])["under_eye"]
    # The effect erodes and feathers its mask, so measure the region's core.
    core = cv2.erode(mask, np.ones((15, 15), np.uint8)) > 0
    out = effects.under_eye_reduction(image, face["landmarks_px"], 80)
    assert out[core].mean() > image[core].mean() + 8
    assert np.array_equal(out[:50], image[:50]), "pixels far from the eyes stay untouched"


def test_zero_intensity_procedures_are_noops(demo):
    image, face = demo
    assert np.array_equal(effects.lip_filler(image, face["landmarks_px"], 0), image)
    assert np.array_equal(effects.under_eye_reduction(image, face["landmarks_px"], 0), image)
