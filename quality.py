"""Heuristic image quality checks for fundus images.
These are simple sanity checks, NOT clinical quality metrics.
"""
from PIL import Image
import numpy as np
import io


def check_quality(file_bytes: bytes) -> tuple[Image.Image | None, str | None]:
    """Return (rgb_image, None) if OK, or (None, reason_string) if rejected."""
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        return None, "File could not be opened as an image."
    w, h = img.size
    if w < 256 or h < 256:
        return None, f"Resolution too low ({w}×{h}). Minimum is 256×256."
    arr = np.asarray(img, dtype=np.float32)
    mean_brightness = arr.mean()
    if mean_brightness < 20:
        return None, f"Image is too dark (mean brightness {mean_brightness:.1f})."
    if mean_brightness > 235:
        return None, f"Image is overexposed (mean brightness {mean_brightness:.1f})."
    flat = arr.reshape(-1, 3)
    _, counts = np.unique(flat, axis=0, return_counts=True)
    if counts.max() / flat.shape[0] > 0.95:
        return None, "Image appears to be a single solid colour."
    return img, None
