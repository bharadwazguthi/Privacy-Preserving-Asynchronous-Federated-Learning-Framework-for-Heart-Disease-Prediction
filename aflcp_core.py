"""
AFLCP Backend - Consolidated single-file backend for Federated Learning
Combines training and prediction into reusable functions.
"""
import os
import time
import math
import random
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================
class AFLCPConfig:
    """Configuration for AFLCP training"""
    def __init__(self, **kwargs):
        # Data
        self.csv_path = kwargs.get('csv_path', 'data/heart2.csv')
        self.test_size = kwargs.get('test_size', 0.2)
        self.seed = kwargs.get('seed', 42)
        
        # FL parameters
        self.rounds = kwargs.get('rounds', 30)
        self.num_clients = kwargs.get('num_clients', 5)
        self.clients_per_round = kwargs.get('clients_per_round', 2)
        self.local_epochs = kwargs.get('local_epochs', 3)
        self.local_batch = kwargs.get('local_batch', 16)
        self.delta = kwargs.get('delta', 2)  # deep/shallow schedule
        self.server_alpha = kwargs.get('server_alpha', 0.6)
        self.temporal_lambda = kwargs.get('temporal_lambda', 0.05)
        
        # Advanced features
        self.use_fedprox = kwargs.get('use_fedprox', False)
        self.fedprox_mu = kwargs.get('fedprox_mu', 0.01)
        self.use_topk = kwargs.get('use_topk', False)
        self.topk_frac = kwargs.get('topk_frac', 0.02)
        self.use_dp = kwargs.get('use_dp', False)
        self.dp_sigma = kwargs.get('dp_sigma', 0.8)
        self.dp_clip = kwargs.get('dp_clip', 1.0)
        self.robust = kwargs.get('robust', 'none')  # none/median/trimmed
        self.trim_k = kwargs.get('trim_k', 1)
        
        # Output
        self.save_dir = kwargs.get('save_dir', 'aflcp_weights')
        self.verbose = kwargs.get('verbose', True)


# ============================================================================
# DATA PREPROCESSING
# ============================================================================
class DataPreprocessor:
    """Handles data loading and preprocessing"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.n_classes = None
        self.label_col = None
        
    def load_and_preprocess(self, csv_path):
        """Load CSV and preprocess features"""
        print(f"Loading {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Detect label column
        label_candidates = ['num', 'target', 'diagnosis', 'outcome', 'Outcome', 'class', 'y', 'label']
        self.label_col = next((c for c in label_candidates if c in df.columns), None)
        if self.label_col is None:
            self.label_col = df.columns[-1]
            print(f"Using last column as label: {self.label_col}")
        
        # Drop id/dataset columns
        drop_cols = [c for c in ['id', 'dataset'] if c in df.columns]
        df = df.drop(columns=drop_cols, errors='ignore')
        
        # Separate features
        feature_raw = [c for c in df.columns if c != self.label_col]
        
        # Coerce numeric and handle categoricals
        Xdf = pd.DataFrame(index=df.index)
        for c in feature_raw:
            coerced = pd.to_numeric(df[c], errors='coerce')
            if coerced.notna().all():
                Xdf[c] = coerced
            else:
                Xdf[c] = df[c].astype(str).fillna("missing")
        
        # One-hot encode
        Xdf = pd.get_dummies(Xdf, drop_first=True)
        print(f"Features after one-hot: {Xdf.shape}")
        
        # Combine and drop NaN
        data = pd.concat([Xdf, df[self.label_col]], axis=1).dropna().reset_index(drop=True)
        if len(data) < len(df):
            print(f"Dropped {len(df)-len(data)} rows due to NaN")
        
        X_all = data.drop(columns=[self.label_col]).values
        y_raw = data[self.label_col].values
        
        # Convert labels to integers
        try:
            y_num = pd.to_numeric(y_raw, errors='coerce')
            if y_num.isna().any():
                y = pd.factorize(y_raw)[0]
            else:
                y = y_num.astype(int).values
        except (ValueError, TypeError):
            y = pd.factorize(y_raw)[0]
        
        self.n_classes = len(np.unique(y))
        print(f"Detected classes: {self.n_classes}")
        
        # Scale features
        X_all = self.scaler.fit_transform(X_all)
        self.feature_columns = list(Xdf.columns)
        
        return X_all, y
    
    def save_artifacts(self, save_dir):
        """Save preprocessing artifacts"""
        os.makedirs(save_dir, exist_ok=True)
        pickle.dump(self.scaler, open(os.path.join(save_dir, "scaler.pkl"), "wb"))
        np.save(os.path.join(save_dir, "feature_columns.npy"), np.array(self.feature_columns))
        np.save(os.path.join(save_dir, "n_classes.npy"), np.array([self.n_classes]))
        print(f"Saved preprocessing artifacts to {save_dir}")
    
    @staticmethod
    def load_artifacts(save_dir):
        """Load preprocessing artifacts"""
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        features_path = os.path.join(save_dir, "feature_columns.npy")
        n_classes_path = os.path.join(save_dir, "n_classes.npy")
        
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        feature_columns = np.load(features_path).tolist()
        n_classes = int(np.load(n_classes_path)[0])
        
        return scaler, feature_columns, n_classes
    
    def preprocess_single(self, input_dict, scaler=None, feature_columns=None):
        """Preprocess a single input dictionary for prediction"""
        if scaler is None:
            scaler = self.scaler
        if feature_columns is None:
            feature_columns = self.feature_columns
        
        df_in = pd.DataFrame([input_dict])
        
        # Coerce numeric
        df_coerced = pd.DataFrame()
        for c in df_in.columns:
            coerced = pd.to_numeric(df_in[c], errors='coerce')
            if coerced.notna().all():
                df_coerced[c] = coerced
            else:
                df_coerced[c] = df_in[c].astype(str).fillna("missing")
        
        # One-hot encode
        df_dummies = pd.get_dummies(df_coerced, drop_first=True)
        
        # Align to training features
        aligned = pd.DataFrame(0.0, index=[0], columns=feature_columns)
        for c in df_dummies.columns:
            if c in aligned.columns:
                aligned.at[0, c] = df_dummies.at[0, c]
        
        # Scale
        X = scaler.transform(aligned.values)
        return X


# ============================================================================
# MODEL CREATION
# ============================================================================
def create_model(input_dim, n_classes):
    """Create a simple feedforward neural network"""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu')
    ])
    
    if n_classes == 2:
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model


# ============================================================================
# FEDERATED LEARNING UTILITIES
# ============================================================================
class FLUtils:
    """Utility functions for federated learning"""
    
    @staticmethod
    def split_shards(X, y, n):
        """Split data into n client shards"""
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        splits = np.array_split(idx, n)
        return [X[s] for s in splits], [y[s] for s in splits]
    
    @staticmethod
    def set_weights(model, weights):
        """Set model weights from list"""
        for var, w in zip(model.weights, weights):
            var.assign(w)
    
    @staticmethod
    def get_weights(model):
        """Get model weights as list of numpy arrays"""
        return [w.numpy() for w in model.weights]
    
    @staticmethod
    def l2_norm(weights_list):
        """Calculate L2 norm of weights"""
        return math.sqrt(sum(float(np.sum((x.astype(np.float64))**2)) for x in weights_list))
    
    @staticmethod
    def clip_weights(weights_list, clip_norm):
        """Clip weights to have max L2 norm"""
        norm = FLUtils.l2_norm(weights_list)
        if norm <= clip_norm:
            return weights_list
        factor = clip_norm / (norm + 1e-12)
        return [x * factor for x in weights_list]
    
    @staticmethod
    def topk_sparsify(delta_list, residuals, k_frac):
        """Apply Top-K sparsification with error feedback"""
        sent = [None] * len(delta_list)
        new_res = [None] * len(delta_list)
        
        for i, (d, r) in enumerate(zip(delta_list, residuals)):
            combined = d + r
            flat = combined.flatten()
            sz = flat.size
            k = max(1, int(k_frac * sz))
            
            if k >= sz:
                sent[i] = combined
                new_res[i] = np.zeros_like(combined)
            else:
                idx = np.argpartition(np.abs(flat), -k)[-k:]
                mask = np.zeros(sz, dtype=bool)
                mask[idx] = True
                sent_flat = np.zeros(sz, dtype=flat.dtype)
                sent_flat[mask] = flat[mask]
                sent[i] = sent_flat.reshape(combined.shape)
                res_flat = flat - sent_flat
                new_res[i] = res_flat.reshape(combined.shape)
        
        return sent, new_res
    
    @staticmethod
    def add_dp_noise(weights_list, sigma, clip_norm):
        """Add differential privacy noise"""
        non_null = [x for x in weights_list if x is not None]
        if not non_null:
            return weights_list
        
        clipped = FLUtils.clip_weights(non_null, clip_norm)
        noisy = []
        for t in clipped:
            noise = np.random.normal(0.0, sigma * clip_norm, size=t.shape).astype(t.dtype)
            noisy.append(t + noise)
        
        # Map back
        out = []
        it = 0
        for x in weights_list:
            if x is None:
                out.append(None)
            else:
                out.append(noisy[it])
                it += 1
        return out
    
    @staticmethod
    def robust_aggregate(values_list, method="mean", trim_k=1):
        """Robust aggregation (median or trimmed mean)"""
        if not values_list:
            return None
        
        stacked = np.stack(values_list, axis=0)
        
        if method == "median":
            return np.median(stacked, axis=0)
        elif method == "trimmed":
            n = stacked.shape[0]
            k = min(trim_k, (n - 1) // 2)
            if k == 0:
                return np.mean(stacked, axis=0)
            sorted_vals = np.sort(stacked, axis=0)
            return np.mean(sorted_vals[k:n-k, ...], axis=0)
        else:
            return np.mean(stacked, axis=0)


# ============================================================================
# FEDPROX TRAINING
# ============================================================================
def fedprox_train(client_X, client_y, model, global_weights, config):
    """Train with FedProx regularization"""
    FLUtils.set_weights(model, global_weights)
    
    optimizer = tf.keras.optimizers.Adam(1e-3)
    n_classes = len(np.unique(client_y))
    
    if n_classes == 2:
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    
    dataset = tf.data.Dataset.from_tensor_slices((client_X, client_y))
    dataset = dataset.shuffle(2048).batch(config.local_batch)
    
    for epoch in range(config.local_epochs):
        for xb, yb in dataset:
            with tf.GradientTape() as tape:
                preds = model(xb, training=True)
                loss = loss_fn(yb, preds)
                
                # FedProx proximal term
                prox = tf.constant(0.0, dtype=loss.dtype)
                for var, gw in zip(model.trainable_variables, global_weights):
                    prox += tf.reduce_sum(tf.square(var - tf.convert_to_tensor(gw, dtype=var.dtype)))
                prox = (config.fedprox_mu / 2.0) * prox
                total_loss = loss + prox
            
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))


# ============================================================================
# MAIN TRAINER CLASS
# ============================================================================
class AFLCPTrainer:
    """Main AFLCP training orchestrator"""
    
    def __init__(self, config):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.global_model = None
        self.global_weights = None
        self.clients_X = None
        self.clients_y = None
        self.X_test = None
        self.y_test = None
        self.residuals = {}
        self.metrics_history = {"round": [], "accuracy": [], "f1": [], "auc": []}
        
        # Set seeds
        random.seed(config.seed)
        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)
        
    def prepare_data(self):
        """Load and split data"""
        X_all, y = self.preprocessor.load_and_preprocess(self.config.csv_path)
        
        # Train/test split
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X_all, y, test_size=self.config.test_size,
            random_state=self.config.seed, stratify=y
        )
        
        # Split into client shards
        self.clients_X, self.clients_y = FLUtils.split_shards(
            X_train, y_train, self.config.num_clients
        )
        
        for i, cx in enumerate(self.clients_X):
            print(f"Client {i} has {len(cx)} examples")
        
        # Initialize model
        self.global_model = create_model(X_all.shape[1], self.preprocessor.n_classes)
        self.global_weights = FLUtils.get_weights(self.global_model)
        
        # Initialize residuals for Top-K
        self.residuals = {
            cid: [np.zeros_like(w) for w in self.global_weights]
            for cid in range(self.config.num_clients)
        }
        
        # Define layer split for deep/shallow
        total_tensors = len(self.global_weights)
        split_idx = total_tensors // 2
        self.shallow_idx = list(range(0, split_idx))
        self.deep_idx = list(range(split_idx, total_tensors))
        
        # Save preprocessing artifacts
        self.preprocessor.save_artifacts(self.config.save_dir)
    
    def train_client(self, client_id, round_num):
        """Train a single client"""
        if len(self.clients_X[client_id]) == 0:
            return None
        
        # Create local model
        local_model = create_model(
            self.X_test.shape[1],
            self.preprocessor.n_classes
        )
        
        # Snapshot global weights
        gw_snapshot = [w.copy() for w in self.global_weights]
        FLUtils.set_weights(local_model, gw_snapshot)
        
        # Train
        if self.config.use_fedprox:
            fedprox_train(
                self.clients_X[client_id],
                self.clients_y[client_id],
                local_model,
                gw_snapshot,
                self.config
            )
        else:
            local_model.fit(
                self.clients_X[client_id],
                self.clients_y[client_id],
                epochs=self.config.local_epochs,
                batch_size=self.config.local_batch,
                verbose=0
            )
        
        local_weights = [w.astype(np.float32) for w in local_model.get_weights()]
        
        # Determine deep/shallow exchange
        if (round_num % self.config.delta) == 0:
            send_idx = self.deep_idx
            exchange = "deep"
        else:
            send_idx = self.shallow_idx
            exchange = "shallow"
        
        # Compute deltas
        delta = [(lw - gw) for lw, gw in zip(local_weights, gw_snapshot)]
        
        # Apply Top-K if enabled
        mask = [0] * len(delta)
        sent = [None] * len(delta)
        
        if self.config.use_topk:
            per_delta = [
                delta[i] if i in send_idx else np.zeros_like(delta[i])
                for i in range(len(delta))
            ]
            sent_sparse, new_res = FLUtils.topk_sparsify(
                per_delta, self.residuals[client_id], self.config.topk_frac
            )
            for i in range(len(delta)):
                if i in send_idx:
                    sent[i] = sent_sparse[i]
                    mask[i] = 1
                    self.residuals[client_id][i] = new_res[i]
                else:
                    sent[i] = None
                    mask[i] = 0
        else:
            for i in range(len(delta)):
                if i in send_idx:
                    sent[i] = delta[i]
                    mask[i] = 1
                else:
                    sent[i] = None
                    mask[i] = 0
        
        # Add DP noise
        if self.config.use_dp:
            sent = FLUtils.add_dp_noise(sent, self.config.dp_sigma, self.config.dp_clip)
        
        # Convert to absolute weights
        client_tensors = [
            (gw_snapshot[i] + sent[i]) if sent[i] is not None else None
            for i in range(len(delta))
        ]
        
        return {
            "client_id": client_id,
            "weights": client_tensors,
            "mask": mask,
            "num_examples": len(self.clients_X[client_id]),
            "exchange": exchange
        }
    
    def aggregate(self, arrivals, server_time):
        """Aggregate client updates"""
        if self.config.robust == "none":
            # Sequential aggregation with temporal weighting
            for arr in arrivals:
                staleness = max(0.0, server_time - arr["timestamp"])
                temporal_w = math.exp(-self.config.temporal_lambda * staleness)
                
                if self.config.verbose:
                    print(f"  Client {arr['client_id']}: staleness={staleness:.3f}, "
                          f"temporal_w={temporal_w:.4f}, exchange={arr['exchange']}")
                
                for i, (gw, cw, m) in enumerate(zip(self.global_weights, arr["weights"], arr["mask"])):
                    if int(m) == 0 or cw is None:
                        continue
                    cw_arr = np.array(cw, dtype=gw.dtype)
                    self.global_weights[i] = (
                        (1 - self.config.server_alpha) * gw +
                        self.config.server_alpha * (temporal_w * cw_arr)
                    )
        else:
            # Robust aggregation
            per_tensor_vals = [[] for _ in range(len(self.global_weights))]
            
            for arr in arrivals:
                staleness = max(0.0, server_time - arr["timestamp"])
                temporal_w = math.exp(-self.config.temporal_lambda * staleness)
                
                for i, (cw, m) in enumerate(zip(arr["weights"], arr["mask"])):
                    if int(m) == 0 or cw is None:
                        continue
                    per_tensor_vals[i].append(temporal_w * np.array(cw, dtype=np.float32))
            
            for i, vals in enumerate(per_tensor_vals):
                if len(vals) == 0:
                    continue
                agg = FLUtils.robust_aggregate(
                    vals,
                    method="median" if self.config.robust == "median" else "trimmed",
                    trim_k=self.config.trim_k
                )
                self.global_weights[i] = (
                    (1 - self.config.server_alpha) * self.global_weights[i] +
                    self.config.server_alpha * agg
                )
    
    def evaluate(self):
        """Evaluate global model"""
        FLUtils.set_weights(self.global_model, self.global_weights)
        preds_prob = self.global_model.predict(self.X_test, batch_size=64, verbose=0)
        
        if self.preprocessor.n_classes == 2:
            preds = (preds_prob > 0.5).astype(int).reshape(-1)
        else:
            preds = np.argmax(preds_prob, axis=1)
        
        acc = accuracy_score(self.y_test, preds)
        f1 = f1_score(self.y_test, preds, average='weighted')
        
        try:
            if self.preprocessor.n_classes == 2:
                auc = roc_auc_score(self.y_test, preds_prob)
            else:
                auc = roc_auc_score(self.y_test, preds_prob, multi_class='ovr')
        except (ValueError, TypeError):
            auc = None
        
        return acc, f1, auc
    
    def train(self):
        """Main training loop"""
        os.makedirs(self.config.save_dir, exist_ok=True)
        server_time = 0.0
        
        print(f"\n{'='*60}")
        print("Starting AFLCP Training")
        print(f"{'='*60}\n")
        
        for round_num in range(1, self.config.rounds + 1):
            round_start = time.time()
            
            # Select clients
            chosen = random.sample(
                range(self.config.num_clients),
                k=min(self.config.clients_per_round, self.config.num_clients)
            )
            
            # Train clients
            arrivals = []
            for cid in chosen:
                delay = random.uniform(0.0, 0.8)
                result = self.train_client(cid, round_num)
                
                if result is not None:
                    result["timestamp"] = server_time + delay
                    arrivals.append(result)
            
            # Sort by arrival time
            arrivals.sort(key=lambda x: x["timestamp"])
            
            # Update server time
            if arrivals:
                server_time = max(arr["timestamp"] for arr in arrivals)
            
            # Aggregate
            self.aggregate(arrivals, server_time)
            
            # Evaluate
            acc, f1, auc = self.evaluate()
            self.metrics_history["round"].append(round_num)
            self.metrics_history["accuracy"].append(acc)
            self.metrics_history["f1"].append(f1)
            self.metrics_history["auc"].append(auc if auc is not None else float('nan'))
            
            round_time = time.time() - round_start
            
            print(f"Round {round_num:02d} | "
                  f"Clients {[a['client_id'] for a in arrivals]} | "
                  f"Acc {acc:.4f} | F1 {f1:.4f} | "
                  f"AUC {auc:.4f if auc else 'N/A'} | "
                  f"Time {round_time:.2f}s")
            
            # Save round weights
            np.savez(
                os.path.join(self.config.save_dir, f"aflcp_round_{round_num}.npz"),
                *self.global_weights
            )
        
        # Save final model
        FLUtils.set_weights(self.global_model, self.global_weights)
        self.global_model.save(os.path.join(self.config.save_dir, "global_model.h5"))
        
        # Save metrics
        pd.DataFrame(self.metrics_history).to_csv(
            os.path.join(self.config.save_dir, "metrics.csv"),
            index=False
        )
        
        print(f"\n{'='*60}")
        print(f"Training complete! Artifacts saved to {self.config.save_dir}")
        print(f"{'='*60}\n")
        
        return self.metrics_history


# ============================================================================
# PREDICTION CLASS
# ============================================================================
class AFLCPPredictor:
    """Handles prediction with trained model"""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.n_classes = None
        self.preprocessor = DataPreprocessor(AFLCPConfig())
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model and preprocessing artifacts"""
        model_path = os.path.join(self.save_dir, "global_model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = tf.keras.models.load_model(model_path)
        self.scaler, self.feature_columns, self.n_classes = DataPreprocessor.load_artifacts(self.save_dir)
        print(f"Loaded model from {self.save_dir}")
    
    def predict(self, input_dict, prob_threshold=0.5):
        """Make prediction on single input"""
        X = self.preprocessor.preprocess_single(input_dict, self.scaler, self.feature_columns)
        probs = self.model.predict(X, verbose=0)[0]
        
        if self.n_classes == 2:
            prob_pos = float(probs) if np.isscalar(probs) or probs.shape == () else float(probs[0])
            pred = 1 if prob_pos >= prob_threshold else 0
            return {
                "probability_of_disease": prob_pos,
                "prediction": int(pred),
                "confidence": prob_pos if pred == 1 else (1 - prob_pos)
            }
        else:
            probs_list = probs.tolist()
            pred_class = int(np.argmax(probs_list))
            return {
                "probabilities": probs_list,
                "predicted_class": pred_class,
                "confidence": probs_list[pred_class]
            }
    
    def predict_batch(self, input_list):
        """Make predictions on multiple inputs"""
        return [self.predict(inp) for inp in input_list]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================
def train_aflcp(config_dict):
    """Train AFLCP model with given configuration
    
    Args:
        config_dict: Dictionary with training parameters
        
    Returns:
        metrics_history: Dictionary with training metrics
    """
    config = AFLCPConfig(**config_dict)
    trainer = AFLCPTrainer(config)
    trainer.prepare_data()
    return trainer.train()


def predict_aflcp(save_dir, input_dict):
    """Make prediction using trained model
    
    Args:
        save_dir: Directory containing trained model
        input_dict: Dictionary with patient features
        
    Returns:
        prediction: Dictionary with prediction results
    """
    predictor = AFLCPPredictor(save_dir)
    return predictor.predict(input_dict)


def load_metrics(save_dir):
    """Load training metrics from CSV
    
    Args:
        save_dir: Directory containing metrics.csv
        
    Returns:
        DataFrame with metrics
    """
    metrics_path = os.path.join(save_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        return pd.read_csv(metrics_path)