"""DR Screening Support — Research Prototype.

Single-file Streamlit app for explainable diabetic retinopathy screening.
Run:  streamlit run app.py
"""
import base64
import io
import json
import os
from datetime import datetime, timezone

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DR Screening Support — Research Prototype",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Inject CSS: mandatory disclaimer always visible, flag colours, badge styles
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .disclaimer-box {
        background: #fff3cd;
        border: 2px solid #d4a017;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 18px 0 10px 0;
        font-size: 15px;
        font-weight: 600;
        color: #664d03;
        line-height: 1.5;
    }
    .grade-badge {
        font-size: 26px;
        font-weight: 700;
        padding: 10px 20px;
        border-radius: 10px;
        display: inline-block;
        margin-bottom: 8px;
    }
    .flag-ROUTINE  { background: #d4edda; color: #155724; border: 2px solid #28a745; }
    .flag-REVIEW   { background: #fff3cd; color: #856404; border: 2px solid #ffc107; }
    .flag-REFER    { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
    .flag-UNGREADABLE { background: #e2e3e5; color: #383d41; border: 2px solid #6c757d; }
    .gradcam-caption {
        font-size: 12px;
        color: #666;
        font-style: italic;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Imports from project modules
# ---------------------------------------------------------------------------
from quality import check_quality  # noqa: E402
from model_utils import (  # noqa: E402
    MODEL_FILE,
    IMG_SIZE,
    REFER_THRESHOLD,
    REVIEW_CONFIDENCE,
    GRADE_NAMES,
    load_model,
    load_calibration,
    apply_temperature,
    preprocess_image,
    predict,
    derive_outputs,
)

# Grad-CAM imports are deferred until model is loaded successfully.

# ---------------------------------------------------------------------------
# Model directory (same directory as this script)
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Cache model across reruns
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def cached_load_model():
    """Load model once and cache; also pre-build the Grad-CAM model."""
    model, err = load_model(MODEL_DIR)
    grad_model = None
    if model is not None:
        try:
            from gradcam import build_grad_model
            grad_model = build_grad_model(model)
        except Exception as exc:
            st.warning(f"Grad-CAM model could not be built: {exc}")
    return model, grad_model, err


@st.cache_resource(show_spinner=False)
def cached_load_calibration():
    return load_calibration(MODEL_DIR)


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("DR Screening Support — Research Prototype")

# ---------------------------------------------------------------------------
# Mandatory disclaimer — always visible, non-dismissable
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="disclaimer-box">'
    "⚠️ <b>Research prototype.</b> Not a diagnostic device. "
    "Outputs are for decision support only and require review by a "
    "qualified eye-care professional. Do not use for patient care."
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model, grad_model, model_err = cached_load_model()
temperature, is_calibrated = cached_load_calibration()

if model_err:
    st.error(f"⚠️ Model failed to load: {model_err}")
    st.info(
        f"Place `{MODEL_FILE}` in `{MODEL_DIR}` and restart the app."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload a retinal fundus image (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"],
)

if uploaded is None:
    st.info("👆 Upload a fundus image to begin.")
    st.stop()

file_bytes = uploaded.read()
filename = uploaded.name

# ---------------------------------------------------------------------------
# Quality check
# ---------------------------------------------------------------------------
pil_img, quality_reason = check_quality(file_bytes)

if pil_img is None:
    # UNGREADABLE path — do NOT run the model
    st.markdown(
        '<div class="grade-badge flag-UNGREADABLE">'
        "UNGREADABLE — Recapture or manual review"
        "</div>",
        unsafe_allow_html=True,
    )
    st.warning(f"Quality check failed: {quality_reason}")

    # Still produce a downloadable JSON for the unreadable case
    result_json = {
        "filename": filename,
        "grade": None,
        "grade_name": None,
        "probabilities": None,
        "confidence": None,
        "calibrated": is_calibrated,
        "p_referable": None,
        "referable_threshold": REFER_THRESHOLD,
        "flag": "UNGREADABLE",
        "quality_reason": quality_reason,
        "gradcam_png_b64": None,
        "model_file": MODEL_FILE,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    st.download_button(
        "📥 Download result JSON",
        data=json.dumps(result_json, indent=2),
        file_name=f"{os.path.splitext(filename)[0]}_result.json",
        mime="application/json",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
try:
    img_array = preprocess_image(pil_img)  # (1, 384, 384, 3), raw [0, 255]
    probs = predict(model, img_array)      # shape (5,)

    # Optional temperature scaling
    if is_calibrated and temperature is not None:
        probs = apply_temperature(probs, temperature)

    outputs = derive_outputs(probs, is_calibrated)
except Exception as exc:
    st.error(f"Inference failed: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
overlay_img = None
cam_array = None
try:
    from gradcam import compute_gradcam, overlay_gradcam

    cam_array = compute_gradcam(model, img_array, grad_model=grad_model)
    overlay_img = overlay_gradcam(pil_img, cam_array, alpha=0.45)
except Exception as exc:
    st.warning(f"Grad-CAM generation failed: {exc}")

# ---------------------------------------------------------------------------
# Layout: left = thumbnail, right = results
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Uploaded Image")
    st.caption(filename)
    st.image(pil_img, use_container_width=True)

with col_right:
    # --- Grade badge ---
    flag = outputs["flag"]
    grade_text = f"Grade {outputs['grade']} — {outputs['grade_name']}"
    st.markdown(
        f'<div class="grade-badge flag-{flag}">{grade_text}</div>',
        unsafe_allow_html=True,
    )

    # --- Confidence line ---
    conf_label = (
        "Calibrated confidence" if is_calibrated else "Model score (uncalibrated)"
    )
    st.markdown(f"**{conf_label}:** {outputs['confidence']:.4f}")

    # --- Referral line ---
    ref_color = "🔴" if flag == "REFER" else "🟢"
    st.markdown(
        f"**Referable probability:** {outputs['p_referable']:.4f} "
        f"(threshold {REFER_THRESHOLD}) — **{flag}** {ref_color}"
    )

    # --- 5-class probability bar chart ---
    st.subheader("Class Probabilities")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 2.5))
    classes = [GRADE_NAMES[i] for i in range(5)]
    bars = ax.barh(classes, outputs["probabilities"], color="#aec7e8")
    # Highlight predicted class
    bars[outputs["grade"]].set_color("#1f77b4")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    for i, v in enumerate(outputs["probabilities"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Grad-CAM panel (full width)
# ---------------------------------------------------------------------------
st.subheader("Grad-CAM Explanation")
if overlay_img is not None:
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.image(
            pil_img.resize((IMG_SIZE, IMG_SIZE)),
            caption="Original (resized to 384×384)",
            use_container_width=True,
        )
    with gcol2:
        st.image(overlay_img, caption="Grad-CAM overlay", use_container_width=True)
    st.markdown(
        '<p class="gradcam-caption">'
        "Grad-CAM indicates coarse image regions that influenced this prediction. "
        "It is not a clinical justification and is not lesion-validated."
        "</p>",
        unsafe_allow_html=True,
    )
else:
    st.info("Grad-CAM overlay could not be generated for this image.")

# ---------------------------------------------------------------------------
# Mandatory disclaimer (repeated at bottom to ensure visibility)
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="disclaimer-box">'
    "⚠️ <b>Research prototype.</b> Not a diagnostic device. "
    "Outputs are for decision support only and require review by a "
    "qualified eye-care professional. Do not use for patient care."
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Collapsible technical details + JSON download
# ---------------------------------------------------------------------------
with st.expander("🔧 Technical Details"):
    st.markdown(f"- **Model file:** `{MODEL_FILE}`")
    st.markdown(f"- **Input size:** {IMG_SIZE}×{IMG_SIZE} RGB")
    st.markdown("- **Grad-CAM target layer:** `conv5_block16_concat`")
    st.markdown(
        f"- **Temperature:** {'T = ' + str(temperature) if is_calibrated else 'None (uncalibrated)'}"
    )
    st.markdown(f"- **Referral threshold:** p_referable ≥ {REFER_THRESHOLD}")
    st.markdown(f"- **Review threshold:** confidence < {REVIEW_CONFIDENCE}")
    st.markdown("- **No network calls. No telemetry.**")

    # Build Grad-CAM base64 for JSON
    gradcam_b64 = None
    if overlay_img is not None:
        buf = io.BytesIO()
        overlay_img.save(buf, format="PNG")
        gradcam_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    result_json = {
        "filename": filename,
        "grade": outputs["grade"],
        "grade_name": outputs["grade_name"],
        "probabilities": outputs["probabilities"],
        "confidence": outputs["confidence"],
        "calibrated": outputs["calibrated"],
        "p_referable": outputs["p_referable"],
        "referable_threshold": outputs["referable_threshold"],
        "flag": outputs["flag"],
        "quality_reason": None,
        "gradcam_png_b64": gradcam_b64,
        "model_file": MODEL_FILE,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    st.download_button(
        "📥 Download result JSON",
        data=json.dumps(result_json, indent=2),
        file_name=f"{os.path.splitext(filename)[0]}_result.json",
        mime="application/json",
    )
