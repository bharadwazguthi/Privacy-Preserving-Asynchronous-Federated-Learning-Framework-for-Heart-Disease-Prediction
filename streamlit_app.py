import streamlit as st
import subprocess, threading, time, os, glob, json
import pandas as pd, numpy as np
import pickle
import tensorflow as tf
from io import StringIO
from pathlib import Path

# -----------------------
# Helpers
# -----------------------
ROOT = Path(".").resolve()
SAVE_DIR_DEFAULT = str(ROOT / "aflcp_weights")

def find_latest_artifact(save_dir=SAVE_DIR_DEFAULT):
    p = Path(save_dir)
    models = sorted(p.glob("global_model*.h5"), key=os.path.getmtime) if p.exists() else []
    metrics = p / "metrics.csv"
    rounds = sorted(p.glob("aflcp_round_*.npz")) if p.exists() else []
    return {
        "model": str(models[-1]) if models else None,
        "metrics": str(metrics) if metrics.exists() else None,
        "rounds": [str(x) for x in rounds],
        "save_dir": str(p)
    }

def read_metrics_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def load_feature_columns(save_dir):
    p = Path(save_dir) / "feature_columns.npy"
    if p.exists():
        try:
            return np.load(p).tolist()
        except Exception:
            return None
    return None

def load_scaler(save_dir):
    p = Path(save_dir) / "scaler.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

def preprocess_input_from_dict(input_dict, save_dir):
    # mimic training preprocess: coerce numeric, get_dummies, align to feature_columns, scale
    feat_cols = load_feature_columns(save_dir)
    scaler = load_scaler(save_dir)
    if feat_cols is None or scaler is None:
        raise FileNotFoundError("feature_columns.npy or scaler.pkl missing in save_dir")
    df_in = pd.DataFrame([input_dict])
    df_coerced = pd.DataFrame()
    for c in df_in.columns:
        coerced = pd.to_numeric(df_in[c], errors='coerce')
        if coerced.notna().all():
            df_coerced[c] = coerced
        else:
            df_coerced[c] = df_in[c].astype(str).fillna("missing")
    df_dummies = pd.get_dummies(df_coerced, drop_first=True)
    aligned = pd.DataFrame(columns=feat_cols)
    aligned = pd.concat([aligned, pd.DataFrame([{}])], ignore_index=True).fillna(0.0)
    for c in df_dummies.columns:
        if c in aligned.columns:
            aligned.at[0,c] = df_dummies.at[0,c]
    X = scaler.transform(aligned.values)
    return X

def predict_single_ui(input_dict, save_dir):
    # load model & run prediction
    artifacts = find_latest_artifact(save_dir)
    if artifacts["model"] is None:
        st.error("No saved model found in " + save_dir)
        return None
    model = tf.keras.models.load_model(artifacts["model"])
    X = preprocess_input_from_dict(input_dict, save_dir)
    probs = model.predict(X)[0]
    n_classes_path = Path(save_dir) / "n_classes.npy"
    n_classes = int(np.load(n_classes_path)[0]) if n_classes_path.exists() else (1 if probs.shape==() else len(probs))
    if n_classes == 2:
        prob_pos = float(probs) if np.isscalar(probs) or probs.shape==() else float(probs[0])
        pred = 1 if prob_pos >= 0.5 else 0
        return {"probability_of_disease": prob_pos, "prediction": int(pred)}
    else:
        probs_list = probs.tolist()
        return {"probabilities": probs_list, "predicted_class": int(np.argmax(probs_list))}

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="AFLCP Control Panel", layout="wide")
st.title("AFLCP — single-command frontend")

col1, col2 = st.columns([2,1])
with col1:
    st.header("Run training (single-process)")

    csv_path = st.text_input("CSV dataset path", value=str(ROOT / "AFLCP" / "Datasets" / "heart2.csv"))
    rounds = st.number_input("Global rounds", min_value=1, value=20, step=1)
    num_clients = st.number_input("Num clients (shards)", min_value=1, value=5, step=1)
    clients_per_round = st.number_input("Clients per round", min_value=1, value=2, step=1)
    local_epochs = st.number_input("Local epochs", min_value=1, value=3, step=1)
    delta = st.number_input("Δ (deep/shallow schedule)", min_value=1, value=2, step=1)
    server_alpha = st.slider("Server alpha (mixing)", 0.0, 1.0, 0.6)
    temporal_lambda = st.number_input("Temporal lambda", min_value=0.0, value=0.05, step=0.01)

    st.subheader("Upgrades (optional)")
    use_fedprox = st.checkbox("Enable FedProx")
    fedprox_mu = st.number_input("FedProx μ", min_value=0.0, value=0.01, step=0.01)
    use_topk = st.checkbox("Enable Top-K")
    topk_frac = st.number_input("Top-K fraction", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
    use_dp = st.checkbox("Enable DP noise")
    dp_sigma = st.number_input("DP sigma", min_value=0.0, value=0.8, step=0.1)
    dp_clip = st.number_input("DP clip norm", min_value=0.0, value=1.0, step=0.1)
    robust = st.selectbox("Robust aggregation", options=["none","median","trimmed"])

    save_dir = st.text_input("Save artifacts to", value=SAVE_DIR_DEFAULT)

    start_btn = st.button("Start training (runs run_aflcp_sim.py)")

    st.markdown("---")
    st.write("Notes:")
    st.write("- Training runs in a subprocess; console output is shown live below.")
    st.write("- Make sure `run_aflcp_sim.py` is in the current working directory and Python env has deps.")

with col2:
    st.header("Artifacts")
    artifacts = find_latest_artifact(save_dir)
    st.write("Save dir:", artifacts["save_dir"])
    st.write("Global model:", artifacts["model"])
    st.write("Metrics CSV:", artifacts["metrics"])
    st.write("Round snapshots:", len(artifacts["rounds"]))
    if artifacts["metrics"]:
        dfm = read_metrics_csv(artifacts["metrics"])
        if dfm is not None:
            st.dataframe(dfm.tail(10))
            st.line_chart(dfm.set_index("round")[["accuracy","f1"]])

# -----------------------
# Run training process and capture output
# -----------------------
if start_btn:
    # build command
    cmd = ["python", "run_aflcp_sim.py",
           "--csv", csv_path,
           "--rounds", str(int(rounds)),
           "--num_clients", str(int(num_clients)),
           "--clients_per_round", str(int(clients_per_round)),
           "--local_epochs", str(int(local_epochs)),
           "--delta", str(int(delta)),
           "--server_alpha", str(float(server_alpha)),
           "--temporal_lambda", str(float(temporal_lambda)),
           "--save_dir", save_dir,
           "--robust", robust]
    if use_fedprox:
        cmd += ["--use_fedprox", "--fedprox_mu", str(float(fedprox_mu))]
    if use_topk:
        cmd += ["--use_topk", "--topk_frac", str(float(topk_frac))]
    if use_dp:
        cmd += ["--use_dp", "--dp_sigma", str(float(dp_sigma)), "--dp_clip", str(float(dp_clip))]

    st.write("Running:", " ".join(cmd))

    # run in background thread and stream stdout
    log_box = st.empty()
    def run_subproc():
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = ""
        for line in p.stdout:
            out += line
            log_box.text(out)
        p.wait()
        log_box.text(out + "\n[PROCESS EXITED. Check artifacts in save_dir]")

    thread = threading.Thread(target=run_subproc, daemon=True)
    thread.start()

# -----------------------
# Prediction UI
# -----------------------
st.markdown("---")
st.header("Predict single patient")
save_dir_input = st.text_input("Artifacts directory for prediction", value=SAVE_DIR_DEFAULT)
feat_cols = load_feature_columns(save_dir_input)
if feat_cols is None:
    st.warning("No feature_columns.npy found in save dir. After you run training the app will detect features.")
    # allow raw CSV upload for quick prediction
    uploaded = st.file_uploader("Upload CSV row (single-row) for prediction", type=["csv"])
    if uploaded is not None:
        df_new = pd.read_csv(uploaded)
        st.write("First row:")
        st.write(df_new.head(1))
        if st.button("Predict from uploaded row"):
            row = df_new.iloc[0].to_dict()
            try:
                res = predict_single_ui(row, save_dir_input)
                st.json(res)
            except Exception as e:
                st.error("Prediction failed: " + str(e))
else:
    st.write(f"Detected {len(feat_cols)} model input features (after one-hot). You can provide raw fields; the app will one-hot and align.")
    st.info("Provide raw columns present in your training CSV (the app will coerce types and align dummies).")
    raw_json = st.text_area("Paste a JSON object with raw patient fields (e.g. {\"age\":63, \"sex\":\"Male\", ...})")
    if st.button("Predict from JSON"):
        try:
            input_dict = json.loads(raw_json)
            res = predict_single_ui(input_dict, save_dir_input)
            st.json(res)
        except Exception as e:
            st.error("Invalid JSON or prediction failed: " + str(e))

st.markdown("---")
st.write("App ready.")