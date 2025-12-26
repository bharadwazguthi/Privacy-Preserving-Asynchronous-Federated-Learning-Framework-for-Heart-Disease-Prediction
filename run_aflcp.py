"""
Single-process AFLCP simulator with FedProx, Top-K+residuals, DP, and robust aggregation.
Run with: python run_aflcp_sim.py --csv path/to/your.csv [options]

Outputs:
 - aflcp_round_{r}.npz  (per-round raw weights)
 - global_model.h5
 - scaler.pkl
 - feature_columns.npy
 - metrics.csv
"""
import os, time, math, random, argparse, pickle
from copy import deepcopy
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ----------------- Args -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="/Users/bharadwazguthi/Downloads/Major Proj/data/heart2.csv")
parser.add_argument("--rounds", type=int, default=30)
parser.add_argument("--num_clients", type=int, default=5)
parser.add_argument("--clients_per_round", type=int, default=2)
parser.add_argument("--local_epochs", type=int, default=3)
parser.add_argument("--local_batch", type=int, default=16)
parser.add_argument("--delta", type=int, default=2, help="deep/shallow schedule Δ")
parser.add_argument("--server_alpha", type=float, default=0.6)
parser.add_argument("--temporal_lambda", type=float, default=0.05)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_dir", type=str, default="/Users/bharadwazguthi/Downloads/Major Proj/aflcp_weights")

# Upgrades
parser.add_argument("--use_fedprox", action="store_true")
parser.add_argument("--fedprox_mu", type=float, default=0.01)
parser.add_argument("--use_topk", action="store_true")
parser.add_argument("--topk_frac", type=float, default=0.02)
parser.add_argument("--use_dp", action="store_true")
parser.add_argument("--dp_sigma", type=float, default=0.8)
parser.add_argument("--dp_clip", type=float, default=1.0)
parser.add_argument("--robust", type=str, default="none", choices=["none","median","trimmed"])
parser.add_argument("--trim_k", type=int, default=1)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

# reproducibility
random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)

os.makedirs(args.save_dir, exist_ok=True)

# ----------------- Data loader & preprocessing -----------------
print("Loading", args.csv)
df = pd.read_csv(args.csv)
# detect label
label_candidates = ['num','target','diagnosis','outcome','Outcome','class','y','label']
label_col = next((c for c in label_candidates if c in df.columns), None)
if label_col is None:
    label_col = df.columns[-1]
    print("Using last column as label:", label_col)

# drop id/dataset if present
drop_cols = [c for c in ['id','dataset'] if c in df.columns]
df = df.drop(columns=drop_cols, errors='ignore')

feature_raw = [c for c in df.columns if c != label_col]
# coerce numeric, treat non-numeric as category
Xdf = pd.DataFrame(index=df.index)
for c in feature_raw:
    coerced = pd.to_numeric(df[c], errors='coerce')
    if coerced.notna().all():
        Xdf[c] = coerced
    else:
        Xdf[c] = df[c].astype(str).fillna("missing")
# one-hot encode all object columns
Xdf = pd.get_dummies(Xdf, drop_first=True)
print("Features after one-hot:", Xdf.shape)
# combine with label and drop NA
data = pd.concat([Xdf, df[label_col]], axis=1).dropna().reset_index(drop=True)
if len(data) < len(df):
    print(f"Dropped {len(df)-len(data)} rows due to NaN after preprocessing")
X_all = data.drop(columns=[label_col]).values
y_raw = data[label_col].values
# labels -> ints
try:
    y_num = pd.to_numeric(y_raw, errors='coerce')
    if y_num.isna().any():
        y = pd.factorize(y_raw)[0]
    else:
        y = y_num.astype(int).values
except:
    y = pd.factorize(y_raw)[0]
n_classes = len(np.unique(y))
print("Detected classes:", n_classes)
scaler = StandardScaler()
X_all = scaler.fit_transform(X_all)
feature_cols = list(Xdf.columns)

# save preprocessing artifacts
pickle.dump(scaler, open(os.path.join(args.save_dir,"scaler.pkl"), "wb"))
np.save(os.path.join(args.save_dir,"feature_columns.npy"), np.array(feature_cols))
np.save(os.path.join(args.save_dir,"n_classes.npy"), np.array([n_classes]))

# train/test split
X_trainval, X_test, y_trainval, y_test = train_test_split(X_all, y, test_size=0.2, random_state=args.seed, stratify=y)

# split into client shards
def split_shards(X, y, n):
    idx = np.arange(len(X)); np.random.shuffle(idx)
    splits = np.array_split(idx, n)
    return [X[s] for s in splits], [y[s] for s in splits]

clients_X, clients_y = split_shards(X_trainval, y_trainval, args.num_clients)
for i, cx in enumerate(clients_X):
    print(f"Client {i} has {len(cx)} examples")

# ----------------- Model factory -----------------
def create_model(input_dim, n_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(shape=(input_dim,)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    if n_classes == 2:
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

global_model = create_model(X_all.shape[1], n_classes)
global_weights = [w.numpy() for w in global_model.weights]

# define shallow/deep split
total_tensors = len(global_weights)
split_idx = total_tensors // 2
shallow_idx = list(range(0, split_idx))
deep_idx = list(range(split_idx, total_tensors))

# ----------------- helper functions -----------------
def set_weights_list(model, weights):
    for var, w in zip(model.weights, weights):
        var.assign(w)

def get_weights_list(model):
    return [w.numpy() for w in model.weights]

def l2_norm_list(lst):
    return math.sqrt(sum(float(np.sum((x.astype(np.float64))**2)) for x in lst))

def clip_list(lst, clip_norm):
    norm = l2_norm_list(lst)
    if norm <= clip_norm:
        return lst
    factor = clip_norm / (norm + 1e-12)
    return [x * factor for x in lst]

def topk_sparsify_per_tensor(delta_list, residuals, k_frac):
    sent = [None]*len(delta_list)
    new_res = [None]*len(delta_list)
    for i,(d,r) in enumerate(zip(delta_list,residuals)):
        combined = d + r
        flat = combined.flatten()
        sz = flat.size
        k = max(1, int(k_frac * sz))
        if k >= sz:
            sent[i] = combined
            new_res[i] = np.zeros_like(combined)
        else:
            idx = np.argpartition(np.abs(flat), -k)[-k:]
            mask = np.zeros(sz, dtype=bool); mask[idx]=True
            sent_flat = np.zeros(sz, dtype=flat.dtype)
            sent_flat[mask] = flat[mask]
            sent[i] = sent_flat.reshape(combined.shape)
            res_flat = flat - sent_flat
            new_res[i] = res_flat.reshape(combined.shape)
    return sent, new_res

def add_dp_noise(sent_list, sigma, clip_norm):
    # sent_list: list of numpy arrays (some may be None)
    non_null = [x for x in sent_list if x is not None]
    if not non_null:
        return sent_list
    clipped = clip_list(non_null, clip_norm)
    noisy = []
    for t in clipped:
        noise = np.random.normal(0.0, sigma*clip_norm, size=t.shape).astype(t.dtype)
        noisy.append(t + noise)
    # map back
    out = []
    it = 0
    for x in sent_list:
        if x is None:
            out.append(None)
        else:
            out.append(noisy[it]); it+=1
    return out

def robust_aggregate_per_tensor(list_of_client_vals, method="median", trim_k=1):
    # list_of_client_vals: list of arrays (n_clients, shape...) or empty
    if not list_of_client_vals:
        return None
    stacked = np.stack(list_of_client_vals, axis=0)
    if method=="median":
        return np.median(stacked, axis=0)
    elif method=="trimmed":
        n = stacked.shape[0]; k = min(trim_k, (n-1)//2)
        if k==0:
            return np.mean(stacked, axis=0)
        sorted_vals = np.sort(stacked, axis=0)
        return np.mean(sorted_vals[k:n-k,...], axis=0)
    else:
        return np.mean(stacked, axis=0)

# ----------------- Simulation state (residuals per client, for top-k) -----------------
residuals = {cid: [np.zeros_like(w) for w in global_weights] for cid in range(args.num_clients)}

# ----------------- Training helper (client) -----------------
import tensorflow as tf
optimizer = tf.keras.optimizers.Adam(1e-3)
if n_classes==2:
    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=False)
else:
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

def fedprox_train(client_X, client_y, model, global_w, epochs, batch, mu):
    # global_w list of numpy arrays
    set_weights_list(model, global_w)
    ds = tf.data.Dataset.from_tensor_slices((client_X, client_y)).shuffle(2048).batch(batch)
    for epoch in range(epochs):
        for xb, yb in ds:
            with tf.GradientTape() as tape:
                preds = model(xb, training=True)
                loss = loss_obj(yb, preds)
                prox = tf.constant(0.0, dtype=loss.dtype)
                for var, gw in zip(model.trainable_variables, global_w):
                    prox += tf.reduce_sum(tf.square(var - tf.convert_to_tensor(gw, dtype=var.dtype)))
                prox = (mu / 2.0) * prox
                total = loss + prox
            grads = tape.gradient(total, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

# ----------------- Evaluation helper -----------------
def evaluate_global(weights):
    set_weights_list(global_model, weights)
    preds_prob = global_model.predict(X_test, batch_size=64)
    if n_classes==2:
        preds = (preds_prob > 0.5).astype(int).reshape(-1)
    else:
        preds = np.argmax(preds_prob, axis=1)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    try:
        auc = roc_auc_score(y_test, preds_prob if n_classes==2 else np.argmax(preds_prob,axis=1))
    except:
        auc = None
    return acc, f1, auc

# ----------------- Main AFLCP simulation loop -----------------
metrics = {"round":[],"accuracy":[],"f1":[],"auc":[]}
server_time = time.time()
print("Starting single-process AFLCP simulation")
for r in range(1, args.rounds+1):
    # pick clients for this round
    chosen = random.sample(range(args.num_clients), k=min(args.clients_per_round, args.num_clients))
    arrivals = []
    for cid in chosen:
        # simulate computation delay
        delay = random.uniform(0.0, 0.8)
        # snapshot global weights
        gw_snapshot = [w.copy() for w in global_weights]
        # local training
        if len(clients_X[cid])==0:
            # nothing to do
            arrival = {"client_id":cid, "weights":[None]*len(global_weights), "timestamp": server_time + delay, "mask":[0]*len(global_weights)}
            arrivals.append(arrival); continue
        # create a fresh model for local training
        local_model = create_model(X_all.shape[1], n_classes)
        # set to global snapshot
        set_weights_list(local_model, gw_snapshot)
        # train
        if args.use_fedprox:
            fedprox_train(clients_X[cid], clients_y[cid], local_model, gw_snapshot, args.local_epochs, args.local_batch, args.fedprox_mu)
        else:
            local_model.fit(clients_X[cid], clients_y[cid], epochs=args.local_epochs, batch_size=args.local_batch, verbose=0)
        local_w = [w.astype(np.float32) for w in local_model.get_weights()]
        # select deep or shallow
        if (r % args.delta)==0:
            send_idx = deep_idx; exch = "deep"
        else:
            send_idx = shallow_idx; exch = "shallow"
        # compute delta
        delta = [ (lw - gw) for lw, gw in zip(local_w, gw_snapshot) ]
        # if topk enabled: sparsify deltas only for send_idx
        mask = [0]*len(delta)
        sent = [None]*len(delta)
        if args.use_topk:
            per_delta = [ delta[i] if i in send_idx else np.zeros_like(delta[i]) for i in range(len(delta))]
            sent_sparse, new_res = topk_sparsify_per_tensor(per_delta, residuals[cid], args.topk_frac)
            for i in range(len(delta)):
                if i in send_idx:
                    sent[i] = sent_sparse[i]
                    mask[i]=1
                    residuals[cid][i] = new_res[i]
                else:
                    sent[i]=None; mask[i]=0
        else:
            for i in range(len(delta)):
                if i in send_idx:
                    sent[i] = delta[i]  # send full delta
                    mask[i]=1
                else:
                    sent[i]=None; mask[i]=0
        # optionally add DP noise to sent list
        if args.use_dp:
            sent = add_dp_noise(sent, args.dp_sigma, args.dp_clip)
        # client prepares "weights to send" as full tensors (server expects tensors to mix)
        # convert sent deltas back to "client tensors" in absolute form: client_tensor = gw_snapshot + delta_sent
        client_tensors = [ (gw_snapshot[i] + sent[i]) if sent[i] is not None else None for i in range(len(delta)) ]
        arrival = {"client_id":cid, "weights": client_tensors, "timestamp": server_time + delay, "mask": mask, "num_examples": len(clients_X[cid]), "exchange":exch}
        arrivals.append(arrival)
    # sort arrivals by timestamp (simulated arrival order)
    arrivals.sort(key=lambda x: x["timestamp"])
    # server aggregation
    server_time = max(server_time, time.time())
    if args.robust == "none":
        # sequential processing
        for arr in arrivals:
            staleness = max(0.0, server_time - arr["timestamp"])
            temporal_w = math.exp(-args.temporal_lambda * staleness)
            if args.verbose:
                print(f"[Round {r}] Applying client {arr['client_id']} staleness={staleness:.3f} temporal_w={temporal_w:.4f} exchange={arr.get('exchange')}")
            for i,(gw, cw, m) in enumerate(zip(global_weights, arr["weights"], arr["mask"])):
                if int(m)==0 or cw is None:
                    continue
                cw_arr = np.array(cw, dtype=gw.dtype)
                global_weights[i] = (1 - args.server_alpha) * gw + args.server_alpha * (temporal_w * cw_arr)
    else:
        # robust per-tensor across arrivals
        server_time = time.time()
        per_tensor_vals = [[] for _ in range(len(global_weights))]
        for arr in arrivals:
            staleness = max(0.0, server_time - arr["timestamp"])
            temporal_w = math.exp(-args.temporal_lambda * staleness)
            for i,(cw,m) in enumerate(zip(arr["weights"], arr["mask"])):
                if int(m)==0 or cw is None:
                    continue
                per_tensor_vals[i].append(temporal_w * np.array(cw, dtype=np.float32))
        for i, vals in enumerate(per_tensor_vals):
            if len(vals)==0: continue
            agg = robust_aggregate_per_tensor(vals, method=("median" if args.robust=="median" else "trimmed"), trim_k=args.trim_k)
            global_weights[i] = (1 - args.server_alpha) * global_weights[i] + args.server_alpha * agg
        print(f"[Round {r}] Robust aggregation ({args.robust}) processed {len(arrivals)} arrivals")
    # evaluate
    acc, f1, auc = evaluate_global(global_weights)
    metrics["round"].append(r); metrics["accuracy"].append(acc); metrics["f1"].append(f1); metrics["auc"].append(auc if auc is not None else float('nan'))
    print(f"Round {r:02d} | clients { [a['client_id'] for a in arrivals] } | acc {acc:.4f} | f1 {f1:.4f} | auc {auc}")
    # save round weights
    np.savez(os.path.join(args.save_dir, f"aflcp_round_{r}.npz"), *global_weights)

# final save model
set_weights_list(global_model, global_weights)
global_model.save(os.path.join(args.save_dir, "global_model.h5"))
pd.DataFrame(metrics).to_csv(os.path.join(args.save_dir,"metrics.csv"), index=False)
print("Training complete. Artifacts saved to", args.save_dir)

# ----------------- prediction helper -----------------
def preprocess_input_dict(d):
    # d: raw dict (original raw columns)
    df_in = pd.DataFrame([d])
    df_coerced = pd.DataFrame()
    for col in df_in.columns:
        coerced = pd.to_numeric(df_in[col], errors='coerce')
        if coerced.notna().all():
            df_coerced[col] = coerced
        else:
            df_coerced[col] = df_in[col].astype(str).fillna("missing")
    df_dummies = pd.get_dummies(df_coerced, drop_first=True)
    # align to feature_cols
    aligned = pd.DataFrame(columns=feature_cols)
    aligned = aligned.append(pd.Series(), ignore_index=True).fillna(0.0)
    for c in df_dummies.columns:
        if c in aligned.columns:
            aligned.at[0,c] = df_dummies.at[0,c]
    Xs = scaler.transform(aligned.values)
    return Xs

def predict_single(input_dict, prob_threshold=0.5):
    Xs = preprocess_input_dict(input_dict)
    model = tf.keras.models.load_model(os.path.join(args.save_dir,"global_model.h5"))
    probs = model.predict(Xs)[0]
    if n_classes==2:
        prob_pos = float(probs) if np.isscalar(probs) or probs.shape==() else float(probs[0])
        pred = 1 if prob_pos >= prob_threshold else 0
        return {"probability_of_disease": prob_pos, "prediction": int(pred)}
    else:
        probs_list = probs.tolist()
        pred_class = int(np.argmax(probs_list))
        return {"probabilities": probs_list, "predicted_class": pred_class}

print("\nExample call to predict_single():")
print("example = { 'age':63, 'sex':'Male', ... }  -> predict_single(example)")