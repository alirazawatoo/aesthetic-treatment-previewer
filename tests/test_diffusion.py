import sys
import types

import numpy as np
from PIL import Image

from src import diffusion


class _FakePipeline:
    """Lightweight stand-in for StableDiffusionXLInpaintPipeline.

    Returns a solid-color image so we can verify ROI compositing without
    importing heavy ML dependencies.
    """

    def __call__(self, *args, **kwargs):
        init_image = kwargs["image"]
        width, height = init_image.size
        img = Image.new("RGB", (width, height), (128, 0, 0))
        return types.SimpleNamespace(images=[img])


class _FakeGenerator:
    def __init__(self, device=None):
        self.device = device

    def manual_seed(self, seed: int):
        return self


def test_sdxl_inpaint_edit_preserves_shape_and_edits_roi(monkeypatch):
    # Patch pipeline loader to avoid importing diffusers/torch.
    fake_pipe = _FakePipeline()
    device = "cpu"
    dtype = "float32"
    monkeypatch.setattr(
        diffusion,
        "_ensure_pipeline",
        lambda model_id=None: (fake_pipe, device, dtype),
    )

    # Patch torch module used inside sdxl_inpaint_edit.
    fake_torch = types.SimpleNamespace(Generator=_FakeGenerator)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255

    out = diffusion.sdxl_inpaint_edit(
        image,
        mask,
        prompt="test prompt",
        negative_prompt="",
        strength=0.5,
        steps=4,
        seed=0,
    )

    assert out.shape == image.shape
    assert out.dtype == np.uint8

    # Pixels outside the mask should remain unchanged (still black).
    assert np.all(out[:8, :8] == 0)
    # Inside the masked region we expect edits from the fake pipeline.
    assert np.any(out[20:44, 20:44] != 0)


def test_empty_mask_never_loads_a_model(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Empty mask must not initialize the model")

    monkeypatch.setattr(diffusion, "_ensure_pipeline", forbidden)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    assert np.array_equal(
        diffusion.sdxl_inpaint_edit(image, np.zeros((32, 32), dtype=np.uint8), prompt="synthetic"), image
    )


def test_roi_includes_last_pixel_when_padding_is_zero():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[7, 7] = 255
    assert diffusion._roi_from_mask(mask, pad=0) == (7, 7, 8, 8)


def test_loader_requires_cached_weights(monkeypatch):
    calls = []

    class Pipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(kwargs)
            return cls()

        def to(self, device):
            return self

        def enable_attention_slicing(self):
            pass

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float16="float16"))
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(StableDiffusionXLInpaintPipeline=Pipeline))
    monkeypatch.setattr(diffusion, "_PIPE", None)
    monkeypatch.setattr(diffusion, "_get_device_and_dtype", lambda torch: ("cpu", "float32"))
    diffusion._ensure_pipeline("synthetic-model")
    assert calls[0]["local_files_only"] is True


def test_downscaled_roi_is_composited_with_full_resolution_mask(monkeypatch):
    monkeypatch.setattr(
        diffusion,
        "_ensure_pipeline",
        lambda model_id=None: (_FakePipeline(), "cpu", "float32"),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(Generator=_FakeGenerator))

    # A 1000px ROI is downscaled to 768px for inference; single-pixel columns
    # in the mask are dropped by nearest-neighbour resizing at that scale.
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    mask = np.zeros((1000, 1000), dtype=np.uint8)
    mask[:, ::4] = 255

    out = diffusion.sdxl_inpaint_edit(image, mask, prompt="synthetic", seed=None)

    edited = np.any(out != 0, axis=2)
    assert np.array_equal(edited, mask > 0)
