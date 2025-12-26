# backend/main.py
# FastAPI bridge for AFLCP
# - Triggers run_aflcp_sim.py
# - Serves HTML pages
# - Loads trained TensorFlow model for prediction
# - Serves metrics & model files

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates

import subprocess
import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(BASE_DIR, "aflcp_weights")
TEMPLATES_DIR = os.path.join(BASE_DIR, "ui", "templates")

os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# APP INIT
# -----------------------------
app = FastAPI(title="AFLCP Backend")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# -----------------------------
# GLOBAL ARTIFACTS (lazy load)
# -----------------------------
_model = None
_scaler = None
_feature_cols = None
_n_classes = None


def load_artifacts():
    """Load trained AFLCP artifacts if not already loaded"""
    global _model, _scaler, _feature_cols, _n_classes

    if _model is None:
        model_path = os.path.join(SAVE_DIR, "global_model.h5")
        scaler_path = os.path.join(SAVE_DIR, "scaler.pkl")
        feat_path = os.path.join(SAVE_DIR, "feature_columns.npy")
        ncls_path = os.path.join(SAVE_DIR, "n_classes.npy")

        if not os.path.exists(model_path):
            raise FileNotFoundError("Trained model not found. Run training first.")

        _model = tf.keras.models.load_model(model_path)
        _scaler = pickle.load(open(scaler_path, "rb"))
        _feature_cols = np.load(feat_path, allow_pickle=True).tolist()
        _n_classes = int(np.load(ncls_path)[0])


# -----------------------------
# PREPROCESS (same logic as training)
# -----------------------------
def preprocess_input_dict(input_dict: dict):
    df = pd.DataFrame([input_dict])

    # numeric / categorical coercion
    coerced = pd.DataFrame()
    for c in df.columns:
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().all():
            coerced[c] = num
        else:
            coerced[c] = df[c].astype(str).fillna("missing")

    dummies = pd.get_dummies(coerced, drop_first=True)

    aligned = pd.DataFrame(columns=_feature_cols)
    aligned.loc[0] = 0.0

    for c in dummies.columns:
        if c in aligned.columns:
            aligned.at[0, c] = dummies.at[0, c]

    X = _scaler.transform(aligned.values)
    return X


# =====================================================
# PAGE ROUTES (HTML)
# =====================================================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/predict-page", response_class=HTMLResponse)
def predict_page(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})


@app.get("/global-model", response_class=HTMLResponse)
def global_model_page(request: Request):
    return templates.TemplateResponse("global.html", {"request": request})


# =====================================================
# TRAINING (RUN AFLCP SIMULATION)
# =====================================================
@app.post("/train")
def train():
    """
    Starts run_aflcp_sim.py in background.
    Training is NOT blocking the API.
    """
    cmd = [
        "python",
        "run_aflcp_sim.py",
        "--use_fedprox",
        "--use_topk"
    ]

    subprocess.Popen(cmd, cwd=BASE_DIR)
    return {"status": "AFLCP training started"}


# =====================================================
# METRICS
# =====================================================
@app.get("/metrics")
def get_metrics():
    metrics_path = os.path.join(SAVE_DIR, "metrics.csv")
    if not os.path.exists(metrics_path):
        return {"error": "metrics.csv not found. Run training first."}

    df = pd.read_csv(metrics_path)
    return df.to_dict(orient="records")


# =====================================================
# PREDICTION
# =====================================================
@app.post("/predict")
async def predict(request: Request):
    """
    Takes raw patient JSON, returns probability
    """
    try:
        load_artifacts()
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    data = await request.json()
    X = preprocess_input_dict(data)

    probs = _model.predict(X)[0]

    if _n_classes == 2:
        prob = float(probs) if np.isscalar(probs) else float(probs[0])
        return {"probability": prob}
    else:
        return {
            "probabilities": probs.tolist(),
            "predicted_class": int(np.argmax(probs))
        }


# =====================================================
# DOWNLOADS
# =====================================================
@app.get("/download-model")
def download_model():
    path = os.path.join(SAVE_DIR, "global_model.h5")
    return FileResponse(path, filename="global_model.h5")


@app.get("/download-metrics")
def download_metrics():
    path = os.path.join(SAVE_DIR, "metrics.csv")
    return FileResponse(path, filename="metrics.csv")
