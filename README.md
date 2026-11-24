# DRISHTI AI — Diabetic Retinopathy Screening Portal

AI-assisted clinical decision support system for diabetic retinopathy (DR) screening, powered by DenseNet-121 and Grad-CAM explainability.

---

## 🚀 How to Run the Frontend

You have three flexible ways to run the frontend:

### Method 1: High-Performance Web Server (Recommended)
Runs the full backend with real-time **DenseNet-121 inference**, **heuristic quality checks**, and **Grad-CAM visual attribution**:

```bash
python server.py
```
Then open your browser at:
👉 **[http://localhost:8000](http://localhost:8000)**

---

### Method 2: Direct Browser Open (Offline / Standalone)
You can directly double-click or open [index.html](index.html) in any modern browser:
- Double-click `index.html` or drag it into Chrome/Edge/Firefox.
- Works offline anywhere without any server.

---

### Method 3: Streamlit Interface
If you prefer running via Streamlit:

```bash
streamlit run app.py
```
The Streamlit app launches with the **DRISHTI AI Portal** embedded as the primary view, with a sidebar toggle to access the classical view.

---

## 📋 Screening Workflow Steps

1. **Patient Intake**: Register patient demographics (Name, Age, Gender, Phone, Prior DR history) and generate a verified Clinical ID.
2. **Login Session**: Authenticate active clinical screening session.
3. **Image Capture**: Upload retinal fundus photograph (PNG / JPG / JPEG).
4. **Quality Check**: Automated pre-screening QA (Sharpness, Illumination, Framing, Color Spectrum).
5. **AI Diagnostic Assessment**: DenseNet-121 DR grading (Grade 0–4), confidence score, and full multi-class softmax probability breakdown.
6. **Explainable AI (Grad-CAM)**: Visual attribution heatmap highlighting convolutional feature activations driving the classification.
7. **Doctor Review & Verification**: Ophthalmologist verification portal to confirm grade, request OCT scan, or overrule with clinical notes.
8. **Referral Report**: Official diagnostic summary report with clinical referral tier, print functionality, and JSON export.
9. **Patient Database**: Searchable records table tracking all completed screenings and audit history.

---

## 📁 Project Structure

| File | Purpose |
|------|---------|
| [index.html](index.html) | DRISHTI AI Screening Portal frontend (HTML/CSS/JS) |
| [server.py](server.py) | ASGI API & web server (Starlette + Uvicorn) |
| [app.py](app.py) | Streamlit application integrating the DRISHTI AI frontend |
| [model_utils.py](model_utils.py) | DenseNet-121 model loading, preprocessing, and inference |
| [gradcam.py](gradcam.py) | Grad-CAM gradient calculation and heatmap overlay |
| [quality.py](quality.py) | Retinal image quality assurance checks |
| [densenet121_aptos_50ep_best.keras](densenet121_aptos_50ep_best.keras) | Trained DenseNet-121 model weights |
| [requirements.txt](requirements.txt) | Python dependencies |
