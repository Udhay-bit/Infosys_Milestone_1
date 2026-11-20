"""Model loading, inference, and calibration utilities.

IMPORTANT: Do NOT normalize or preprocess images before feeding to the model.
The DenseNet preprocess_input step is already inside the model graph.
Feeding preprocessed values will silently degrade predictions.
"""
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf


def _apply_deserialization_patches():
    """Ensure Keras layers tolerate extra metadata keys like 'quantization_config'.
    
    When models saved with newer Keras 3 configs are loaded in environments where
    Dense.__init__ does not take quantization_config, deserialization would otherwise fail.
    """
    try:
        import keras
        orig_dense_from_config = keras.layers.Dense.from_config
        def patched_dense_from_config(cls, config):
            config.pop("quantization_config", None)
            return orig_dense_from_config(config)
        keras.layers.Dense.from_config = classmethod(patched_dense_from_config)
    except Exception:
        pass


_apply_deserialization_patches()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_FILE = "densenet121_aptos_50ep_best.keras"
IMG_SIZE = 384
GRADE_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}
REFER_THRESHOLD = 0.40
REVIEW_CONFIDENCE = 0.60


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_dir: str = ".") -> tuple[tf.keras.Model | None, str | None]:
    """Load the Keras model; return (model, None) or (None, error_string)."""
    model_path = os.path.join(model_dir, MODEL_FILE)
    if not os.path.isfile(model_path):
        return None, f"Model file not found: {model_path}"
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model, None
    except Exception as exc:
        return None, f"Failed to load model: {exc}"


# ---------------------------------------------------------------------------
# Calibration (optional temperature scaling)
# ---------------------------------------------------------------------------
def load_calibration(model_dir: str = ".") -> tuple[float | None, bool]:
    """Return (temperature, True) if calibration.json exists, else (None, False)."""
    cal_path = os.path.join(model_dir, "calibration.json")
    if os.path.isfile(cal_path):
        try:
            with open(cal_path, "r") as f:
                data = json.load(f)
            return float(data["T"]), True
        except Exception:
            pass
    return None, False


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Re-scale softmax probabilities via temperature on logits.

    Because the model already outputs softmax, we invert to logits,
    divide by T, and re-apply softmax.
    """
    logits = np.log(np.clip(probs, 1e-12, None))
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - scaled.max())  # numerical stability
    return exp_scaled / exp_scaled.sum()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def preprocess_image(pil_img) -> np.ndarray:
    """Resize a PIL RGB image to (1, 384, 384, 3) float32 in [0, 255].

    NOTE: No normalisation here — it is baked into the model graph.
    """
    resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(arr, axis=0)  # (1, 384, 384, 3)


def predict(model, img_array: np.ndarray) -> np.ndarray:
    """Run inference and return the raw softmax probabilities shape (5,)."""
    preds = model.predict(img_array, verbose=0)
    return preds[0]


# ---------------------------------------------------------------------------
# Derived outputs — these are DERIVED, not trained classification heads
# ---------------------------------------------------------------------------
def derive_outputs(probs: np.ndarray, calibrated: bool) -> dict:
    """Compute grade, confidence, referral flag from softmax probs."""
    grade = int(np.argmax(probs))
    grade_name = GRADE_NAMES[grade]
    confidence = float(np.max(probs))

    # p_referable = P(Moderate) + P(Severe) + P(Proliferative)
    p_referable = float(probs[2] + probs[3] + probs[4])

    # Flag logic (quality check handled upstream; UNGREADABLE set by caller)
    if p_referable >= REFER_THRESHOLD:
        flag = "REFER"
    elif confidence < REVIEW_CONFIDENCE:
        flag = "REVIEW"
    else:
        flag = "ROUTINE"

    return {
        "grade": grade,
        "grade_name": grade_name,
        "probabilities": [float(p) for p in probs],
        "confidence": confidence,
        "calibrated": calibrated,
        "p_referable": p_referable,
        "referable_threshold": REFER_THRESHOLD,
        "flag": flag,
    }
