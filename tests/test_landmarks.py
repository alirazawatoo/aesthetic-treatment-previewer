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


def _eye_opening(image, landmarks):
    mask = np.zeros(image.shape[:2], np.uint8)
    for side in ("left", "right"):
        poly = face_utils.eye_polygon_from_landmarks(landmarks, side)
        cv2.fillConvexPoly(mask, cv2.convexHull(poly.astype(np.int32)), 255)
    return mask > 0


def test_under_eye_region_stays_below_the_eye_opening(demo):
    image, face = demo
    lms = face["landmarks_px"]
    region = segmentation.segment_regions(image, lms)["under_eye"] > 0
    assert not (region & _eye_opening(image, lms)).any()


def test_under_eye_reduction_leaves_the_eyes_alone(demo):
    image, face = demo
    lms = face["landmarks_px"]
    eyes = _eye_opening(image, lms)
    out = effects.under_eye_reduction(image, lms, 100)
    diff = np.abs(out.astype(int) - image.astype(int))[eyes]
    assert diff.mean() < 2


def test_under_eye_reduction_does_not_turn_skin_green(demo):
    image, face = demo
    lms = face["landmarks_px"]
    out = effects.under_eye_reduction(image, lms, 100)
    changed = np.any(out != image, axis=2)
    a_before = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[..., 1][changed].astype(float).mean()
    a_after = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)[..., 1][changed].astype(float).mean()
    assert a_after > 128, "OpenCV LAB a=128 is neutral; below it is green"
    assert a_before - a_after < 4
