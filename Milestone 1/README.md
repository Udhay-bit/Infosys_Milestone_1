# DR Screening Support — Research Prototype

Offline, local web app for explainable diabetic retinopathy screening.
Research prototype — **not a diagnostic device**.

## Quick Start

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Ensure the model file sits next to app.py
#    Expected: densenet121_aptos_50ep_best.keras

# 3. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload a retinal fundus image (PNG/JPG) to get a DR severity grade (0–4), class probabilities, Grad-CAM overlay, and referral flag. No internet connection required.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI (single file) |
| `model_utils.py` | Model loading, inference, calibration |
| `gradcam.py` | Grad-CAM heatmap generation |
| `quality.py` | Heuristic image quality checks |
| `requirements.txt` | Python dependencies |
