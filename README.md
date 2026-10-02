# AFLCP — Privacy-Preserving Asynchronous Federated Learning for Heart Disease Prediction

This project implements an asynchronous federated learning framework designed for cardiovascular disease prediction. It simulates a multi-hospital training environment where patient data never leaves individual institutions, while still producing a high-quality shared diagnostic model.

The system is built as a full-stack web application: a FastAPI backend handles the federated training loop and inference pipeline, while an interactive browser-based UI provides real-time training visualization, patient risk assessment, and model management.

---

## Table of Contents

- [Architecture](#architecture)
- [Setup](#setup)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [How Training Works](#how-training-works)
- [Configuration Parameters](#configuration-parameters)
- [Default Credentials](#default-credentials)
- [Datasets](#datasets)
- [References](#references)

---

## Architecture

```
                        +-------------------+
                        |   Browser (UI)    |
                        |  Tailwind + JS    |
                        +--------+----------+
                                 |
                            HTTP / REST
                                 |
                        +--------v----------+
                        |  FastAPI Server   |
                        |  (backend/main.py)|
                        +--------+----------+
                                 |
              +------------------+------------------+
              |                  |                   |
     +--------v------+  +-------v--------+  +-------v--------+
     |   SQLite DB   |  |  aflcp_core.py |  |  Jinja2 HTML   |
     |  (auth/sessions)| |  (FL engine)   |  |  (templates)   |
     +---------------+  +-------+--------+  +----------------+
                                 |
                    Spawns as subprocess
                    for background training
```

The training loop runs as a separate Python process to avoid blocking the web server. TensorFlow's C++ runtime has thread-safety constraints that make in-process background training unreliable under ASGI servers, so we isolate it via `subprocess.Popen`.

---

## Setup

**Prerequisites:** Python 3.10+ (tested on 3.13)

```bash
# Clone the repository
git clone https://github.com/bharadwazguthi/Privacy-Preserving-Asynchronous-Federated-Learning-Framework-for-Heart-Disease-Prediction.git
cd Privacy-Preserving-Asynchronous-Federated-Learning-Framework-for-Heart-Disease-Prediction

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

```bash
source venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000` in a browser. The database is created automatically on first startup — no manual setup required.

---

## Project Structure

```
.
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI routes, API endpoints, metrics plotting
│   ├── database.py           # User auth, session management (SQLite)
│   └── aflcp_core.py         # Federated learning engine, model definition,
│                              # preprocessing, training loop, prediction
├── data/
│   ├── heart.csv             # UCI Cleveland dataset (303 records, 14 features)
│   └── heart2.csv            # Extended cardiovascular dataset (1025 records)
├── ui/
│   ├── static/
│   │   └── favicon.svg
│   └── templates/
│       ├── login.html        # Login and registration
│       ├── main.html         # Dashboard home / overview
│       ├── dashboard.html    # Training controls, live logs, metrics charts
│       ├── predict.html      # Single patient and batch CSV prediction
│       ├── saved_models.html # Model checkpoint management
│       ├── global.html       # Global model performance view
│       └── about.html        # System architecture description
├── requirements.txt
└── .gitignore
```

Directories created at runtime (not committed):

- `aflcp_weights/` — Active model artifacts (`.h5`, scaler, metrics CSV)
- `saved_models/` — User-saved model snapshots
- `uploads/` — Temporary storage for uploaded CSV files
- `backend/aflcp.db` — SQLite database (auto-initialized on startup)

---

## How Training Works

Training is an in-process simulation of a federated learning system with multiple hospital clients:

1. A CSV dataset is uploaded through the dashboard.
2. The data is split into `N` non-overlapping client shards using Dirichlet-based non-IID partitioning (`alpha=0.5`) to simulate demographic variation across hospitals.
3. Each communication round, `C` clients are randomly selected. Each client trains a local copy of the global model on its shard for `E` epochs.
4. Client updates are timestamped with simulated network delays (uniform random, 0 to 0.8s). Updates are processed in arrival order, weighted by a temporal decay factor to down-weight stale gradients.
5. The server aggregates updates into the global model. Depending on configuration, aggregation can use simple weighted averaging, coordinate-wise median, or trimmed mean.

**Privacy and communication mechanisms:**

| Technique | What it does |
| :--- | :--- |
| FedProx | Adds a proximal regularization term to local training to limit divergence from the global model on heterogeneous data |
| Top-K Sparsification | Transmits only the top K% of weight deltas per round (with residual error feedback to preserve convergence) |
| Differential Privacy | Clips the L2 norm of updates and adds calibrated Gaussian noise before transmission |
| Deep/Shallow Exchange | Alternates between transmitting shallow-layer and deep-layer weights each round, cutting per-round payload roughly in half |
| Robust Aggregation | Median or trimmed-mean aggregation to filter out poisoned or adversarial client updates |

---

## Configuration Parameters

These can be set through the dashboard UI when starting a training run.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `rounds` | 30 | Number of communication rounds |
| `num_clients` | 5 | Total simulated hospital clients |
| `clients_per_round` | 2 | Clients selected per round |
| `local_epochs` | 3 | Training epochs per client per round |
| `local_batch` | 16 | Client-side batch size |
| `server_alpha` | 0.6 | Server-side learning rate for aggregation |
| `temporal_lambda` | 0.05 | Decay rate for staleness weighting |
| `use_fedprox` | false | Enable FedProx proximal term |
| `fedprox_mu` | 0.01 | FedProx regularization strength |
| `use_topk` | false | Enable Top-K gradient sparsification |
| `topk_frac` | 0.02 | Fraction of parameters to transmit |
| `use_dp` | false | Enable differential privacy |
| `dp_sigma` | 0.8 | Noise multiplier for DP |
| `dp_clip` | 1.0 | L2 clipping norm for DP |
| `robust` | none | Aggregation method: `none`, `median`, or `trimmed` |

---

## Default Credentials

The database is seeded with two accounts on first startup:

| Username | Password | Role |
| :--- | :--- | :--- |
| `admin` | `admin123` | Administrator |
| `doctor` | `doctor123` | Researcher |

New accounts can be created through the registration form on the login page.

---

## Datasets

The `data/` directory contains two heart disease datasets:

- **heart.csv** — The UCI Cleveland Heart Disease dataset. 303 patient records, 13 clinical features (age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, ECG results, max heart rate, exercise-induced angina, ST depression, ST slope, number of major vessels, thalassemia), and a binary target column.

- **heart2.csv** — A larger combined cardiovascular dataset with 1025 records and the same feature schema. This is the default training dataset.

Both datasets are sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease).

---

## References

1. B. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS 2017.
2. T. Li et al., "Federated Optimization in Heterogeneous Networks" (FedProx), MLSys 2020.
3. J. Xu et al., "Asynchronous Federated Learning on Heterogeneous Devices" (FedFa), 2019.
4. K. Wei et al., "Federated Learning with Differential Privacy," 2020.
5. M. Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning," USENIX Security 2020.

---

## License

This project was developed as part of a final-year research project on privacy-preserving machine learning in healthcare.
