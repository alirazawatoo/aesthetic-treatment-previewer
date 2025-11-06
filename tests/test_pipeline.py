import numpy as np

from src import pipeline


def _face_with_landmarks():
    return {"landmarks_px": np.zeros((478, 2), dtype=np.float32), "bbox": (10, 10, 40, 40), "roll_deg": 0.0}


def _regions(image_rgb, landmarks_px):
    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    mask[20:30, 20:30] = 255
    return {"lip_ring": mask, "under_eye": mask.copy()}


def _run(base, use_gen=True):
    return pipeline.apply_aesthetic_procedures_pipeline(
        base_rgb=base,
        image_for_regions=base,
        face=_face_with_landmarks(),
        use_gen=use_gen,
        gen_strength=0.5,
        gen_steps=4,
        lip_int=50,
        ue_int=50,
        lip_prompt="lips",
        lip_negative_prompt="",
        eye_prompt="eyes",
        eye_negative_prompt="",
        seed=1,
    )


def test_failed_second_inpaint_falls_back_from_the_unedited_base(monkeypatch):
    calls = []

    def flaky_inpaint(image, mask, **kwargs):
        calls.append(kwargs["prompt"])
        if len(calls) == 2:
            raise OSError("weights not cached")
        out = image.copy()
        out[mask > 0] = 200
        return out

    monkeypatch.setattr(pipeline, "segment_regions", _regions)
    monkeypatch.setattr(pipeline, "sdxl_inpaint_edit", flaky_inpaint)
    monkeypatch.setattr(pipeline, "lip_filler", lambda img, lms, k: img)
    monkeypatch.setattr(pipeline, "under_eye_reduction", lambda img, lms, k: img)

    base = np.zeros((64, 64, 3), dtype=np.uint8)
    out, info = _run(base)

    assert calls == ["lips", "eyes"]
    assert np.array_equal(out, base), "the lip inpaint must not survive into the fallback"
    assert info["used_generative"] is False
    assert len(info["warnings"]) == 1


def test_generative_flag_reports_regions_that_were_edited(monkeypatch):
    monkeypatch.setattr(pipeline, "segment_regions", lambda img, lms: {})
    monkeypatch.setattr(pipeline, "sdxl_inpaint_edit", lambda *a, **k: (_ for _ in ()).throw(AssertionError))

    base = np.zeros((64, 64, 3), dtype=np.uint8)
    out, info = _run(base)

    assert info["used_generative"] is False
    assert np.array_equal(out, base)


def test_classical_path_runs_without_generative_editing(monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline, "lip_filler", lambda img, lms, k: seen.append(("lip", k)) or img)
    monkeypatch.setattr(pipeline, "under_eye_reduction", lambda img, lms, k: seen.append(("eye", k)) or img)

    _, info = _run(np.zeros((64, 64, 3), dtype=np.uint8), use_gen=False)

    assert seen == [("lip", 50), ("eye", 50)]
    assert info == {"used_generative": False, "warnings": []}
