# backend/main.py
"""
AFLCP Backend (FastAPI)
- User authentication with SQLite
- Serves UI pages
- Uploads CSV
- Starts AFLCP training (background) with configurable parameters
- Streams logs
- Serves metrics
- Runs predictions (single + batch CSV)
- Save/Load trained models
"""

import os
import sys
import json
import shutil
import threading
import io
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Cookie, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from aflcp_core import train_aflcp, predict_aflcp, load_metrics, AFLCPPredictor
from backend.database import (
    init_db, verify_user, create_user, create_session,
    get_user_from_session, delete_session, save_training_record
)

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, "ui", "templates")

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
SAVE_DIR = os.path.join(BASE_DIR, "aflcp_weights")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
LOG_FILE = os.path.join(BASE_DIR, "training.log")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# -------------------------------------------------
# APP INIT
# -------------------------------------------------
app = FastAPI(title="AFLCP Backend")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

training_process = None
training_status = {"running": False, "completed": False, "error": None}
training_config = {"rounds": 50}  # Track current training config
last_batch_predictions = []  # Store last batch prediction results

# Clear stale training log on startup
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)


# -------------------------------------------------
# AUTH HELPERS
# -------------------------------------------------
def get_current_user(request: Request):
    """Get current user from session cookie"""
    token = request.cookies.get("session_token")
    if not token:
        return None
    return get_user_from_session(token)


def require_auth(request: Request):
    """Check if user is authenticated, return user or None"""
    user = get_current_user(request)
    return user


# -------------------------------------------------
# AUTH ROUTES
# -------------------------------------------------
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    user = get_current_user(request)
    if user:
        return RedirectResponse("/home", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/api/login")
async def api_login(request: Request):
    body = await request.json()
    username = body.get("username", "").strip()
    password = body.get("password", "")

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")

    user = verify_user(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_session(user["id"])
    response = JSONResponse({"message": "Login successful", "user": {
        "username": user["username"],
        "full_name": user["full_name"],
        "role": user["role"]
    }})
    response.set_cookie("session_token", token, httponly=True, max_age=86400 * 7)
    return response


@app.post("/api/register")
async def api_register(request: Request):
    body = await request.json()
    username = body.get("username", "").strip()
    password = body.get("password", "")
    full_name = body.get("full_name", "").strip()

    if not username or not password or not full_name:
        raise HTTPException(status_code=400, detail="All fields required")
    if len(password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters")

    success = create_user(username, password, full_name)
    if not success:
        raise HTTPException(status_code=409, detail="Username already exists")

    user = verify_user(username, password)
    token = create_session(user["id"])
    response = JSONResponse({"message": "Registration successful"})
    response.set_cookie("session_token", token, httponly=True, max_age=86400 * 7)
    return response


@app.get("/api/logout")
def api_logout(request: Request):
    token = request.cookies.get("session_token")
    delete_session(token)
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("session_token")
    return response


@app.get("/api/me")
def api_me(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {
        "username": user["username"],
        "full_name": user["full_name"],
        "role": user["role"]
    }


# -------------------------------------------------
# UI ROUTES (Auth Required)
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return RedirectResponse("/home", status_code=302)


@app.get("/home", response_class=HTMLResponse)
def home(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("main.html", {"request": request, "user": user})


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user})


@app.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("predict.html", {"request": request, "user": user})


@app.get("/global-model", response_class=HTMLResponse)
def global_model(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("global.html", {"request": request, "user": user})


@app.get("/saved-models", response_class=HTMLResponse)
def saved_models_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("saved_models.html", {"request": request, "user": user})


@app.get("/about", response_class=HTMLResponse)
def about_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("about.html", {"request": request, "user": user})


# -------------------------------------------------
# CSV UPLOAD
# -------------------------------------------------
@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")

    path = os.path.join(UPLOAD_DIR, "dataset.csv")
    with open(path, "wb") as f:
        f.write(await file.read())

    # Get basic stats
    df = pd.read_csv(path)
    return {
        "message": "CSV uploaded successfully",
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns)
    }


# -------------------------------------------------
# START TRAINING (with configurable parameters)
# -------------------------------------------------
@app.post("/api/start-training")
async def start_training(request: Request):
    global training_process, training_status, training_config

    csv_path = os.path.join(UPLOAD_DIR, "dataset.csv")
    if not os.path.exists(csv_path):
        return JSONResponse(
            status_code=400,
            content={"error": "Upload CSV file first"}
        )

    # Check if training is already running
    if training_process and training_process.poll() is None:
        return {"message": "Training already running"}

    # Get config from request body
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Reset status
    training_status = {"running": True, "completed": False, "error": None}

    # Clear old metrics
    metrics_path = os.path.join(SAVE_DIR, "metrics.csv")
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    # Build config from request or defaults
    rounds = body.get("rounds", 50)
    num_clients = body.get("num_clients", 5)
    training_config["rounds"] = rounds
    training_config["num_clients"] = num_clients

    config = {
        "csv_path": csv_path,
        "rounds": rounds,
        "num_clients": body.get("num_clients", 5),
        "clients_per_round": body.get("clients_per_round", 4),
        "local_epochs": body.get("local_epochs", 10),
        "local_batch": body.get("local_batch", 16),
        "test_size": body.get("test_size", 0.2),
        "delta": body.get("delta", 1),
        "server_alpha": body.get("server_alpha", 0.2),
        "temporal_lambda": body.get("temporal_lambda", 0.05),
        "use_fedprox": body.get("use_fedprox", False),
        "fedprox_mu": body.get("fedprox_mu", 0.01),
        "use_topk": body.get("use_topk", True),
        "topk_frac": body.get("topk_frac", 0.5),
        "use_dp": body.get("use_dp", True),
        "dp_sigma": body.get("dp_sigma", 0.01),
        "dp_clip": body.get("dp_clip", 3.0),
        "robust": body.get("robust", "none"),
        "trim_k": body.get("trim_k", 1),
        "save_dir": SAVE_DIR,
        "log_file": LOG_FILE,
        "verbose": True
    }

    # Clear old log
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    # Run training as subprocess
    import subprocess
    train_script = os.path.join(BASE_DIR, "train_subprocess.py")
    training_process = subprocess.Popen(
        [sys.executable, train_script, json.dumps(config)],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    # Monitor subprocess
    def monitor_process():
        global training_status
        training_process.wait()
        if training_process.returncode == 0:
            training_status = {"running": False, "completed": True, "error": None}
            try:
                generate_metrics_plot(SAVE_DIR)
            except Exception:
                pass
        else:
            training_status = {"running": False, "completed": False, "error": "Training failed"}

    monitor_thread = threading.Thread(target=monitor_process, daemon=True)
    monitor_thread.start()

    return {"message": "Training started", "config": config}


# -------------------------------------------------
# STREAM TRAINING LOGS
# -------------------------------------------------
@app.get("/api/logs")
def get_logs():
    if not os.path.exists(LOG_FILE):
        return {"logs": "Waiting for logs..."}
    with open(LOG_FILE) as f:
        return {"logs": f.read()}


# -------------------------------------------------
# TRAINING STATUS
# -------------------------------------------------
@app.get("/api/training-status")
def get_training_status():
    global training_status, training_process

    is_running = training_process is not None and training_process.poll() is None

    return {
        "running": is_running,
        "completed": training_status.get("completed", False),
        "error": training_status.get("error")
    }


# -------------------------------------------------
# STATS FOR MAIN DASHBOARD
# -------------------------------------------------
@app.get("/api/stats")
def get_stats():
    metrics_path = os.path.join(SAVE_DIR, "metrics.csv")

    stats = {
        "training_rounds": 0,
        "institutions": training_config.get("num_clients", 0),
        "accuracy": None,
        "saved_models": 0
    }

    if os.path.exists(metrics_path):
        try:
            df = pd.read_csv(metrics_path)
            if len(df) > 0:
                stats["training_rounds"] = len(df)
                latest_acc = df["accuracy"].iloc[-1]
                stats["accuracy"] = round(latest_acc * 100, 1)
        except Exception:
            pass

    # Count saved models
    if os.path.exists(SAVED_MODELS_DIR):
        stats["saved_models"] = len([d for d in os.listdir(SAVED_MODELS_DIR)
                                      if os.path.isdir(os.path.join(SAVED_MODELS_DIR, d))])

    return stats


# -------------------------------------------------
# METRICS (TABLE + CHART)
# -------------------------------------------------
@app.get("/api/metrics")
def get_metrics():
    try:
        df = load_metrics(SAVE_DIR)
        return df.to_dict(orient="records")
    except Exception:
        return []


# -------------------------------------------------
# TRAINING PROGRESS
# -------------------------------------------------
@app.get("/api/progress")
def training_progress():
    metrics_path = os.path.join(SAVE_DIR, "metrics.csv")
    total_rounds = training_config.get("rounds", 50)

    if not os.path.exists(metrics_path):
        return {"progress": 0, "total_rounds": total_rounds}

    try:
        df = pd.read_csv(metrics_path)
        completed = len(df)
        progress = int((completed / total_rounds) * 100)
        return {"progress": min(progress, 100), "total_rounds": total_rounds, "completed": completed}
    except Exception:
        return {"progress": 0, "total_rounds": total_rounds}


# -------------------------------------------------
# MODEL FEATURES
# -------------------------------------------------
@app.get("/api/model-features")
def get_model_features():
    features_path = os.path.join(SAVE_DIR, "feature_columns.npy")
    if not os.path.exists(features_path):
        return {"features": []}
    import numpy as np
    features = list(np.load(features_path, allow_pickle=True))
    return {"features": features}

# -------------------------------------------------
# SINGLE PREDICTION
# -------------------------------------------------
@app.post("/api/predict")
async def predict(request: Request):
    try:
        data = await request.json()

        # Check if a specific model is specified
        model_dir = data.pop("model_dir", None) or SAVE_DIR

        result = predict_aflcp(model_dir, data)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Train or load a model first")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------
# BATCH CSV PREDICTION
# -------------------------------------------------
@app.post("/api/predict-csv")
async def predict_csv(request: Request, file: UploadFile = File(...)):
    global last_batch_predictions

    try:
        # Read CSV
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # Load predictor
        model_path = os.path.join(SAVE_DIR, "global_model.h5")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=400, detail="No trained model found. Please train or load a model first.")

        predictor = AFLCPPredictor(SAVE_DIR)

        # Detect and remove target column if present
        label_candidates = ['num', 'target', 'diagnosis', 'outcome', 'Outcome', 'class', 'y', 'label']
        target_col = next((c for c in label_candidates if c in df.columns), None)

        # Prepare prediction data
        pred_df = df.copy()
        actual_labels = None
        if target_col:
            actual_labels = df[target_col].values
            pred_df = pred_df.drop(columns=[target_col])

        # Make predictions for each row
        results = []
        for idx, row in pred_df.iterrows():
            input_dict = row.to_dict()
            try:
                pred_result = predictor.predict(input_dict)
                prediction = pred_result.get("prediction", 0)
                probability = pred_result.get("probability_of_disease", pred_result.get("confidence", 0))

                result_row = {
                    "patient_id": idx + 1,
                    **{k: (int(v) if isinstance(v, (np.integer,)) else
                          float(v) if isinstance(v, (np.floating, float)) else v)
                       for k, v in row.to_dict().items()},
                    "prediction": int(prediction),
                    "probability": float(probability),
                    "risk_level": "High Risk" if probability >= 0.5 else "Moderate Risk" if probability >= 0.3 else "Low Risk"
                }

                if actual_labels is not None:
                    result_row["actual"] = int(actual_labels[idx])

                results.append(result_row)
            except Exception as e:
                results.append({
                    "patient_id": idx + 1,
                    **row.to_dict(),
                    "prediction": -1,
                    "probability": 0,
                    "risk_level": "Error",
                    "error": str(e)
                })

        # Store for download
        last_batch_predictions = results

        # Split results
        heart_disease = [r for r in results if r.get("prediction") == 1]
        no_heart_disease = [r for r in results if r.get("prediction") == 0]

        return {
            "total": len(results),
            "heart_disease_count": len(heart_disease),
            "no_heart_disease_count": len(no_heart_disease),
            "heart_disease": heart_disease,
            "no_heart_disease": no_heart_disease,
            "all_results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------
# DOWNLOAD HEART DISEASE PATIENTS CSV
# -------------------------------------------------
@app.get("/api/download-hd-patients")
def download_hd_patients():
    global last_batch_predictions

    if not last_batch_predictions:
        raise HTTPException(status_code=404, detail="No predictions available. Run batch prediction first.")

    hd_patients = [r for r in last_batch_predictions if r.get("prediction") == 1]

    if not hd_patients:
        raise HTTPException(status_code=404, detail="No heart disease patients found in predictions.")

    # Create CSV
    output = io.StringIO()
    if hd_patients:
        writer = csv.DictWriter(output, fieldnames=hd_patients[0].keys())
        writer.writeheader()
        for patient in hd_patients:
            clean_row = {}
            for k, v in patient.items():
                if isinstance(v, (np.integer,)):
                    clean_row[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean_row[k] = float(v)
                else:
                    clean_row[k] = v
            writer.writerow(clean_row)

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=heart_disease_patients.csv"}
    )


# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------
@app.post("/api/save-model")
async def save_model(request: Request):
    body = await request.json()
    model_name = body.get("name", "").strip()
    description = body.get("description", "")

    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required")

    # Sanitize name
    safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in model_name)
    save_path = os.path.join(SAVED_MODELS_DIR, safe_name)

    if os.path.exists(save_path):
        raise HTTPException(status_code=409, detail=f"Model '{safe_name}' already exists")

    # Check that model exists
    model_file = os.path.join(SAVE_DIR, "global_model.h5")
    if not os.path.exists(model_file):
        raise HTTPException(status_code=400, detail="No trained model found. Train a model first.")

    os.makedirs(save_path, exist_ok=True)

    # Copy model files
    files_to_copy = [
        "global_model.h5", "scaler.pkl", "feature_columns.npy",
        "n_classes.npy", "metrics.csv"
    ]
    for fname in files_to_copy:
        src = os.path.join(SAVE_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(save_path, fname))

    # Load and include training results
    training_results = {}
    metrics_file = os.path.join(SAVE_DIR, "metrics.csv")
    if os.path.exists(metrics_file):
        mdf = pd.read_csv(metrics_file)
        if not mdf.empty:
            training_results = {
                "total_rounds": len(mdf),
                "final_accuracy": float(mdf.iloc[-1]["accuracy"]),
                "final_f1": float(mdf.iloc[-1]["f1"]),
                "final_auc": float(mdf.iloc[-1]["auc"]) if not pd.isna(mdf.iloc[-1]["auc"]) else None,
                "best_accuracy": float(mdf["accuracy"].max()),
                "best_f1": float(mdf["f1"].max()),
            }

    # Save metadata
    metadata = {
        "model_name": model_name,
        "description": description,
        "saved_at": datetime.now().isoformat(),
        "training_results": training_results,
    }
    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4, default=str)

    return {"message": f"Model saved as '{safe_name}'", "path": save_path}


# -------------------------------------------------
# LIST SAVED MODELS
# -------------------------------------------------
@app.get("/api/saved-models-list")
def list_saved_models():
    models = []
    if os.path.exists(SAVED_MODELS_DIR):
        for name in sorted(os.listdir(SAVED_MODELS_DIR)):
            model_dir = os.path.join(SAVED_MODELS_DIR, name)
            if not os.path.isdir(model_dir):
                continue

            metadata_path = os.path.join(model_dir, "metadata.json")
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata = json.load(f)

            has_model = os.path.exists(os.path.join(model_dir, "global_model.h5"))

            models.append({
                "name": name,
                "display_name": metadata.get("model_name", name),
                "description": metadata.get("description", ""),
                "saved_at": metadata.get("saved_at", ""),
                "training_results": metadata.get("training_results", {}),
                "has_model": has_model
            })

    return {"models": models}


# -------------------------------------------------
# LOAD SAVED MODEL
# -------------------------------------------------
@app.post("/api/load-model")
async def load_model(request: Request):
    body = await request.json()
    model_name = body.get("name", "")

    model_dir = os.path.join(SAVED_MODELS_DIR, model_name)
    if not os.path.exists(model_dir):
        raise HTTPException(status_code=404, detail="Saved model not found")

    # Copy model files to working directory
    files_to_copy = [
        "global_model.h5", "scaler.pkl", "feature_columns.npy",
        "n_classes.npy", "metrics.csv"
    ]
    for fname in files_to_copy:
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(SAVE_DIR, fname))

    return {"message": f"Model '{model_name}' loaded successfully"}


# -------------------------------------------------
# DELETE SAVED MODEL
# -------------------------------------------------
@app.delete("/api/saved-models/{name}")
def delete_saved_model(name: str):
    model_dir = os.path.join(SAVED_MODELS_DIR, name)
    if not os.path.exists(model_dir):
        raise HTTPException(status_code=404, detail="Model not found")

    shutil.rmtree(model_dir)
    return {"message": f"Model '{name}' deleted"}


# -------------------------------------------------
# DOWNLOADS
# -------------------------------------------------
@app.get("/download/model")
def download_model():
    path = os.path.join(SAVE_DIR, "global_model.h5")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(path, filename="global_model.h5")


@app.get("/download/metrics")
def download_metrics():
    path = os.path.join(SAVE_DIR, "metrics.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Metrics not found")
    return FileResponse(path, filename="metrics.csv")


# -------------------------------------------------
# METRICS PLOT
# -------------------------------------------------
@app.get("/api/metrics-plot")
def metrics_plot():
    plot_path = os.path.join(SAVE_DIR, "metrics_plot.png")
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail="Plot not ready")
    return FileResponse(plot_path)


def generate_metrics_plot(save_dir):
    metrics_path = os.path.join(save_dir, "metrics.csv")
    plot_path = os.path.join(save_dir, "metrics_plot.png")

    if not os.path.exists(metrics_path):
        return

    df = pd.read_csv(metrics_path)
    if len(df) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy
    axes[0].plot(df["round"], df["accuracy"], marker="o", color="#1193d4", linewidth=2)
    axes[0].fill_between(df["round"], df["accuracy"], alpha=0.1, color="#1193d4")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Model Accuracy Over Rounds")
    axes[0].grid(True, alpha=0.3)

    # F1 Score
    axes[1].plot(df["round"], df["f1"], marker="s", color="#10b981", linewidth=2)
    axes[1].fill_between(df["round"], df["f1"], alpha=0.1, color="#10b981")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("F1 Score Over Rounds")
    axes[1].grid(True, alpha=0.3)

    # AUC
    if "auc" in df.columns:
        auc_clean = df["auc"].dropna()
        if len(auc_clean) > 0:
            axes[2].plot(df["round"][:len(auc_clean)], auc_clean, marker="^", color="#f59e0b", linewidth=2)
            axes[2].fill_between(df["round"][:len(auc_clean)], auc_clean, alpha=0.1, color="#f59e0b")
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("AUC")
    axes[2].set_title("AUC-ROC Over Rounds")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close()
