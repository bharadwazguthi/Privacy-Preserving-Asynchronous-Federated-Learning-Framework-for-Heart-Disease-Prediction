# backend/main.py
"""
AFLCP Backend (FastAPI)
- Serves UI pages
- Uploads CSV
- Starts AFLCP training (background)
- Streams logs
- Serves metrics
- Runs predictions using trained model
"""

import os
import sys
import threading
import contextlib
import io
import matplotlib.pyplot as plt
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates

from aflcp_core import train_aflcp, predict_aflcp, load_metrics

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, "ui", "templates")

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
SAVE_DIR = os.path.join(BASE_DIR, "aflcp_weights")
LOG_FILE = os.path.join(BASE_DIR, "training.log")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------------------------------
# APP INIT
# -------------------------------------------------
app = FastAPI(title="AFLCP Backend")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

training_thread = None

# -------------------------------------------------
# UI ROUTES
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/home", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.get("/global-model", response_class=HTMLResponse)
def global_model(request: Request):
    return templates.TemplateResponse("global.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
def about_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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

    return {"message": "CSV uploaded successfully"}

# -------------------------------------------------
# START TRAINING
# -------------------------------------------------
@app.post("/api/start-training")
def start_training():
    global training_thread

    csv_path = os.path.join(UPLOAD_DIR, "dataset.csv")
    if not os.path.exists(csv_path):
        return JSONResponse(
            status_code=400,
            content={"error": "Upload CSV file first"}
        )

    if training_thread and training_thread.is_alive():
        return {"message": "Training already running"}

    def run_training():
        # Use file-based logging without redirecting global stdout/stderr
        with open(LOG_FILE, "w", buffering=1) as log:
            log.write("🚀 AFLCP Training Started\n")
            log.flush()
            
            # Capture stdout during training
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = log
                sys.stderr = log

                metrics = train_aflcp({
                    "csv_path": csv_path,
                    "rounds": 30,
                    "num_clients": 5,
                    "clients_per_round": 2,
                    "local_epochs": 3,
                    "use_fedprox": True,
                    "use_topk": True,
                    "save_dir": SAVE_DIR,
                    "verbose": True
                })

                generate_metrics_plot(SAVE_DIR)
                print("✅ AFLCP Training Completed", flush=True)
            finally:
                # Always restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()

    return {"message": "Training started"}

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
# STATS FOR MAIN DASHBOARD
# -------------------------------------------------
@app.get("/api/stats")
def get_stats():
    """Returns statistics for the main dashboard"""
    metrics_path = os.path.join(SAVE_DIR, "metrics.csv")
    
    # Default values
    stats = {
        "models_trained": 0,
        "institutions": 5,  # Number of clients configured
        "accuracy": None
    }
    
    if os.path.exists(metrics_path):
        try:
            df = pd.read_csv(metrics_path)
            if len(df) > 0:
                stats["models_trained"] = len(df)
                # Get latest accuracy as percentage
                latest_acc = df["accuracy"].iloc[-1]
                stats["accuracy"] = round(latest_acc * 100, 1)
        except Exception:
            pass
    
    return stats

# -------------------------------------------------
# METRICS (TABLE + CHART)
# -------------------------------------------------
@app.get("/api/metrics")
def get_metrics():
    try:
        df = load_metrics(SAVE_DIR)
        return df.to_dict(orient="records")
    except:
        return []

# -------------------------------------------------
# PREDICTION API
# -------------------------------------------------
@app.post("/api/predict")
async def predict(data: dict):
    try:
        result = predict_aflcp(SAVE_DIR, data)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Train model first")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/api/progress")
def training_progress():
    metrics_path = os.path.join(SAVE_DIR, "metrics.csv")
    total_rounds = 30

    if not os.path.exists(metrics_path):
        return {"progress": 0}

    df = pd.read_csv(metrics_path)
    completed = len(df)

    progress = int((completed / total_rounds) * 100)
    return {"progress": min(progress, 100)}
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

    plt.figure(figsize=(8, 5))
    plt.plot(df["round"], df["accuracy"], marker="o", label="Accuracy")
    plt.plot(df["round"], df["f1"], marker="s", label="F1-score")
    plt.xlabel("Round")
    plt.ylabel("Score")
    plt.title("AFLCP Training Performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
