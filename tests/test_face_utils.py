import numpy as np

from src import face_utils


def test_skin_mask_from_bbox_covers_face_center():
    h, w = 200, 200
    image = np.zeros((h, w, 3), dtype=np.uint8)
    bbox = (50, 50, 100, 100)

    mask = face_utils.skin_mask_from_bbox(image, bbox)

    assert mask.shape[:2] == (h, w)
    # Center of the bbox should be inside the mask.
    cx = bbox[0] + bbox[2] // 2
    cy = bbox[1] + bbox[3] // 2
    assert mask[cy, cx] == 255


def test_under_eye_mask_returns_none_with_too_few_points():
    # Not enough landmarks for an eye polygon.
    landmarks = np.zeros((10, 2), dtype=np.float32)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    mask = face_utils.under_eye_mask(image, landmarks, side="left")
    assert mask is None


def test_lip_ring_mask_from_landmarks_produces_non_empty_mask():
    # Construct synthetic landmarks with valid indices for lip ring.
    num_points = 500
    landmarks = np.zeros((num_points, 2), dtype=np.float32)
    # Distinct nested ellipses form a ring; coincident points do not.
    for indices, rx, ry in [(face_utils._LIP_OUTER_IDX, 40, 20), (face_utils._LIP_INNER_IDX, 15, 7)]:
        angles = np.linspace(0, 2 * np.pi, len(indices), endpoint=False)
        landmarks[indices] = np.column_stack([100 + rx * np.cos(angles), 100 + ry * np.sin(angles)])

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    mask = face_utils.lip_ring_mask_from_landmarks(image, landmarks)

    assert mask is not None
    assert mask.shape[:2] == (200, 200)
    assert np.any(mask > 0)


def test_coincident_lip_points_produce_an_empty_ring():
    landmarks = np.full((500, 2), 100, dtype=np.float32)
    mask = face_utils.lip_ring_mask_from_landmarks(np.zeros((200, 200, 3), dtype=np.uint8), landmarks)
    assert not np.any(mask)


def test_lip_ring_is_unavailable_when_landmark_array_is_too_short():
    # 402 points passes the size sanity check but lacks index 405.
    landmarks = np.zeros((402, 2), dtype=np.float32)
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    assert face_utils.lip_ring_mask_from_landmarks(image, landmarks) is None
