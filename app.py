import hashlib
import io

import numpy as np
import streamlit as st
from PIL import Image

from src.config import (
    DEFAULT_EYE_NEGATIVE_PROMPT,
    DEFAULT_EYE_PROMPT,
    DEFAULT_LIP_NEGATIVE_PROMPT,
    DEFAULT_LIP_PROMPT,
    QC_BLUR_VARIANCE_THRESHOLD,
    QC_LIGHTING_MEAN_RANGE,
    QC_ORIENTATION_MAX_ABS_DEG,
)
from src.demo_face import make_demo_face
from src.pipeline import (
    apply_adjustments_pipeline,
    apply_aesthetic_procedures_pipeline,
    compute_identity_similarity,
    compute_skin_mask,
    run_qc_pipeline,
)
from src.upload import UploadError, decode_upload

SOURCE_UPLOAD = "Upload a photo"
SOURCE_DEMO = "Synthetic demo face"

st.set_page_config(page_title="Dermal treatment preview", layout="wide")

st.title("Dermal treatment preview")
st.markdown(
    "Try skin smoothing, tone evening and pigment lightening on one photo, on this machine. "
    "This is an image-editing sketch: it **does not predict treatment results and is not medical advice**."
)


def _load_source() -> tuple[str, np.ndarray] | None:
    """Return (cache key, RGB image) for the chosen source, or None while waiting for an upload."""
    source = st.radio("Image source", [SOURCE_UPLOAD, SOURCE_DEMO], horizontal=True)
    if source == SOURCE_DEMO:
        st.caption("A procedurally drawn face, so no real person's photo is needed to try the controls.")
        return "demo", make_demo_face()

    file = st.file_uploader(
        "Photo of a face (JPEG or PNG, up to 10 MiB)",
        type=["jpg", "jpeg", "png"],
        help="Decoded in memory for this session only; the app writes no files.",
    )
    if file is None:
        st.info(
            "Upload a well-lit, front-facing photo to begin, or switch to the synthetic demo face. "
            "Use a photo you have the right to edit."
        )
        return None
    data = file.getvalue()
    try:
        return hashlib.sha256(data).hexdigest(), decode_upload(data)
    except UploadError as e:
        st.error(str(e))
        return None


def _analyse(key: str, image_rgb: np.ndarray) -> dict:
    """Run QC once per image per session; sliders rerun the script but not face detection."""
    cached = st.session_state.get("analysis")
    if cached is None or cached["key"] != key:
        with st.spinner("Checking the photo..."):
            cached = {"key": key, "qc": run_qc_pipeline(image_rgb)}
        st.session_state["analysis"] = cached
    return cached["qc"]


def _status(passed: bool) -> str:
    return ":green[**Pass**]" if passed else ":orange[**Needs attention**]"


def _show_qc(qc: dict) -> None:
    st.subheader("Photo checks")
    if not qc.get("face_present"):
        st.error("No face was found. Use a front-facing photo with the whole face in frame.")
        return
    lo, hi = QC_LIGHTING_MEAN_RANGE
    rows = [
        ("Face", True, "found" + (" with landmarks" if qc["face"].get("landmarks_px") is not None else " (box only)")),
        (
            "Sharpness",
            qc["blur"]["pass"],
            f"variance {qc['blur']['variance']:.0f}, needs {QC_BLUR_VARIANCE_THRESHOLD:.0f}+",
        ),
        (
            "Lighting",
            qc["lighting"]["pass"],
            f"mean {qc['lighting']['mean']:.0f} (aim {lo:.0f}-{hi:.0f}), spread {qc['lighting']['std']:.0f}",
        ),
        (
            "Head tilt",
            qc["orientation"]["pass"],
            f"{qc['orientation']['roll_deg']:.1f} degrees, limit {QC_ORIENTATION_MAX_ABS_DEG:.0f}",
        ),
    ]
    for name, passed, detail in rows:
        st.markdown(f"{name}: {_status(passed)}  \n:gray[{detail}]")
    for message in qc.get("messages", []):
        st.warning(message)


loaded = _load_source()
if loaded is None:
    st.stop()
key, image_rgb = loaded
qc = _analyse(key, image_rgb)

qc_col, img_col = st.columns([1, 2], gap="large")
with qc_col:
    _show_qc(qc)
    proceed_anyway = False
    if qc.get("face_present") and not qc.get("overall_pass"):
        proceed_anyway = st.checkbox("Edit anyway; results may look worse")
with img_col:
    st.image(image_rgb, caption="Original", width="stretch")

face = qc.get("face")
if face is None or not (qc.get("overall_pass") or proceed_anyway):
    st.stop()

try:
    skin_mask = compute_skin_mask(image_rgb, face)
except ValueError:
    st.error("The detected face has no usable region to edit.")
    st.stop()

st.subheader("Skin adjustments")
col1, col2, col3 = st.columns(3)
with col1:
    wrinkle = st.slider("Wrinkle smoothing", 0, 100, 40, step=5, help="Edge-preserving blur inside the face region.")
with col2:
    tone = st.slider("Tone evening", 0, 100, 25, step=5, help="Local contrast equalisation of lightness.")
with col3:
    pigment = st.slider("Pigment lightening", 0, 100, 20, step=5, help="Lifts pixels darker than their surroundings.")

has_landmarks = face.get("landmarks_px") is not None
st.subheader("Lip and under-eye previews")
if not has_landmarks:
    st.caption(
        "These need MediaPipe face landmarks. Install requirements-landmarks.txt; "
        "the box-only fallback detector cannot locate lips or eyes."
    )
lp_col, ue_col = st.columns(2)
with lp_col:
    lip_int = st.slider("Lip volume", 0, 100, 0, step=5, disabled=not has_landmarks)
with ue_col:
    ue_int = st.slider("Under-eye lightening", 0, 100, 0, step=5, disabled=not has_landmarks)

with st.expander("Generative editing (optional, off by default)"):
    st.caption(
        "Uses SDXL inpainting for the lip and under-eye previews. Needs requirements-ml.txt and model weights "
        "already in the local Hugging Face cache; nothing is downloaded. Without them the classical effects are used."
    )
    use_gen = st.toggle("Use generative editing", value=False, disabled=not has_landmarks)
    gen_strength = st.slider("Strength", 0.10, 0.90, 0.55, 0.05, disabled=not use_gen)
    gen_steps = st.slider("Steps", 4, 40, 14, 1, disabled=not use_gen, help="More steps are slower.")
    lip_prompt = st.text_input("Lip prompt", value=DEFAULT_LIP_PROMPT, disabled=not use_gen)
    lip_negative = st.text_input("Lip negative prompt", value=DEFAULT_LIP_NEGATIVE_PROMPT, disabled=not use_gen)
    eye_prompt = st.text_input("Under-eye prompt", value=DEFAULT_EYE_PROMPT, disabled=not use_gen)
    eye_negative = st.text_input("Under-eye negative prompt", value=DEFAULT_EYE_NEGATIVE_PROMPT, disabled=not use_gen)
    seed = st.number_input("Seed", value=42, step=1, disabled=not use_gen)

with st.spinner("Applying adjustments..."):
    result_rgb = apply_adjustments_pipeline(image_rgb, skin_mask, wrinkle=wrinkle, tone=tone, pigment=pigment)
    if has_landmarks:
        result_rgb, proc_info = apply_aesthetic_procedures_pipeline(
            base_rgb=result_rgb,
            image_for_regions=image_rgb,
            face=face,
            use_gen=use_gen,
            gen_strength=gen_strength,
            gen_steps=gen_steps,
            lip_int=lip_int,
            ue_int=ue_int,
            lip_prompt=lip_prompt,
            lip_negative_prompt=lip_negative,
            eye_prompt=eye_prompt,
            eye_negative_prompt=eye_negative,
            seed=int(seed),
        )
        for msg in proc_info["warnings"]:
            st.info(msg)
    ssim_val = compute_identity_similarity(image_rgb, result_rgb, face["bbox"])

st.subheader("Before and after")
before_col, after_col = st.columns(2)
with before_col:
    st.image(image_rgb, caption="Before", width="stretch")
with after_col:
    st.image(result_rgb, caption="After (illustrative edit)", width="stretch")

st.metric(
    "Structural similarity of the face crop",
    f"{ssim_val:.3f}",
    help="SSIM compares pixel structure (1.0 means unchanged). It is not identity verification "
    "and says nothing about how a real treatment would look.",
)

buf = io.BytesIO()
Image.fromarray(result_rgb).save(buf, format="PNG")
png = buf.getvalue()
st.download_button(
    "Download the edited image (PNG)",
    data=png,
    file_name=f"after_{hashlib.sha256(png).hexdigest()[:10]}.png",
    mime="image/png",
)
