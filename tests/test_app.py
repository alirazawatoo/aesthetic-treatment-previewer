import importlib.util
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app():
    return AppTest.from_file(APP, default_timeout=60)


def test_empty_state_asks_for_a_photo(app):
    app.run()
    assert not app.exception
    assert "Upload a well-lit" in app.info[0].value
    assert not app.slider


def test_demo_face_passes_checks_and_renders_an_edit(app):
    app.run()
    app.radio[0].set_value("Synthetic demo face").run()

    assert not app.exception
    checks = " ".join(m.value for m in app.markdown)
    assert "Sharpness" in checks and "Needs attention" not in checks
    assert [s.label for s in app.slider][:3] == ["Wrinkle smoothing", "Tone evening", "Pigment lightening"]
    assert float(app.metric[0].value) < 1.0


@pytest.mark.skipif(importlib.util.find_spec("mediapipe") is not None, reason="MediaPipe supplies landmarks")
def test_lip_and_eye_controls_are_disabled_without_landmarks(app):
    app.run()
    app.radio[0].set_value("Synthetic demo face").run()

    by_label = {s.label: s for s in app.slider}
    assert by_label["Lip volume"].disabled
    assert by_label["Under-eye lightening"].disabled


def test_zero_intensity_leaves_the_face_crop_unchanged(app):
    app.run()
    app.radio[0].set_value("Synthetic demo face").run()
    for label in ("Wrinkle smoothing", "Tone evening", "Pigment lightening"):
        next(s for s in app.slider if s.label == label).set_value(0)
    app.run()

    assert not app.exception
    assert app.metric[0].value == "1.000"


@pytest.mark.skipif(importlib.util.find_spec("mediapipe") is None, reason="needs requirements-landmarks.txt")
def test_landmarks_enable_lip_and_eye_previews(app):
    app.run()
    app.radio[0].set_value("Synthetic demo face").run()
    by_label = {s.label: s for s in app.slider}
    assert not by_label["Lip volume"].disabled
    by_label["Lip volume"].set_value(60)
    by_label["Under-eye lightening"].set_value(60)
    for label in ("Wrinkle smoothing", "Tone evening", "Pigment lightening"):
        by_label[label].set_value(0)
    app.run()

    assert not app.exception
    assert "with landmarks" in " ".join(m.value for m in app.markdown)
    assert float(app.metric[0].value) < 1.0
