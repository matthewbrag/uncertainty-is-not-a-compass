"""
The first MLP (128 hidden units) overfit badly (negative R2, worse than
predicting the mean) - that's evidence THAT config was too much capacity for
the signal, not evidence nonlinearity doesn't exist. Sweeps smaller/more
regularized MLPs plus gradient boosting (a different nonlinear model class
entirely) on one representative layer, using a validation split carved out
of TRAINING data only, so hyperparameter selection never touches the real
held-out test set. Reports the winning config's honest test performance
against the Ridge baseline from before.
"""
import json

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score

LAYER = "22"  # best Ridge R2 from the first pass
IN_PATH = "overnight_merged.jsonl"
TEST_FRAC = 0.1
VAL_FRAC = 0.15  # carved out of the training problems only
SEED = 0

print("Loading data...", flush=True)
records = []
with open(IN_PATH) as f:
    for line in f:
        records.append(json.loads(line))
print(f"{len(records)} points loaded", flush=True)

problem_keys = sorted(set((r["shard"], r["gsm8k_train_index"]) for r in records))
rng = np.random.default_rng(SEED)
rng.shuffle(problem_keys)
n_test = int(len(problem_keys) * TEST_FRAC)
test_problems = set(problem_keys[:n_test])
trainval_problems = problem_keys[n_test:]

rng2 = np.random.default_rng(SEED + 1)
trainval_shuffled = list(trainval_problems)
rng2.shuffle(trainval_shuffled)
n_val = int(len(trainval_shuffled) * VAL_FRAC)
val_problems = set(trainval_shuffled[:n_val])
train_problems = set(trainval_shuffled[n_val:])

train_records = [r for r in records if (r["shard"], r["gsm8k_train_index"]) in train_problems]
val_records = [r for r in records if (r["shard"], r["gsm8k_train_index"]) in val_problems]
test_records = [r for r in records if (r["shard"], r["gsm8k_train_index"]) in test_problems]
print(f"train={len(train_records)} val={len(val_records)} test={len(test_records)} points "
      f"(layer {LAYER})", flush=True)


def build_X(recs):
    return np.array([r["hidden_states"][LAYER] for r in recs], dtype=np.float32)


y_train = np.array([r["outcome_entropy"] for r in train_records])
y_val = np.array([r["outcome_entropy"] for r in val_records])
y_test = np.array([r["outcome_entropy"] for r in test_records])
y_test_binary = (y_test > 0).astype(int)

X_train = build_X(train_records)
X_val = build_X(val_records)
X_test = build_X(test_records)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# baseline: ridge on this layer, same split structure, for direct comparison
ridge = RidgeCV(alphas=np.logspace(-2, 4, 20))
ridge.fit(X_train_s, y_train)
ridge_val_r2 = r2_score(y_val, ridge.predict(X_val_s))
ridge_test_pred = ridge.predict(X_test_s)
ridge_test_r2 = r2_score(y_test, ridge_test_pred)
ridge_test_r = np.corrcoef(ridge_test_pred, y_test)[0, 1]
print(f"\nRidge baseline: val_R2={ridge_val_r2:+.4f}  test_R2={ridge_test_r2:+.4f}  test_r={ridge_test_r:+.4f}",
      flush=True)

print("\n=== MLP sweep (smaller + more regularized) ===", flush=True)
mlp_configs = [
    {"hidden_layer_sizes": (8,), "alpha": 1e-1},
    {"hidden_layer_sizes": (8,), "alpha": 1.0},
    {"hidden_layer_sizes": (16,), "alpha": 1e-1},
    {"hidden_layer_sizes": (16,), "alpha": 1.0},
    {"hidden_layer_sizes": (32,), "alpha": 1e-1},
    {"hidden_layer_sizes": (32,), "alpha": 1.0},
    {"hidden_layer_sizes": (16, 8), "alpha": 1.0},
]
best_mlp = None
best_mlp_val_r2 = -np.inf
for cfg in mlp_configs:
    mlp = MLPRegressor(max_iter=500, early_stopping=True, n_iter_no_change=15,
                        random_state=SEED, **cfg)
    mlp.fit(X_train_s, y_train)
    val_r2 = r2_score(y_val, mlp.predict(X_val_s))
    print(f"  {cfg} -> val_R2={val_r2:+.4f}", flush=True)
    if val_r2 > best_mlp_val_r2:
        best_mlp_val_r2 = val_r2
        best_mlp = mlp
        best_mlp_cfg = cfg

print(f"\nBest MLP config: {best_mlp_cfg} (val_R2={best_mlp_val_r2:+.4f})", flush=True)
mlp_test_pred = best_mlp.predict(X_test_s)
mlp_test_r2 = r2_score(y_test, mlp_test_pred)
mlp_test_r = np.corrcoef(mlp_test_pred, y_test)[0, 1] if np.std(mlp_test_pred) > 0 else 0.0
print(f"Best MLP on held-out test: test_R2={mlp_test_r2:+.4f}  test_r={mlp_test_r:+.4f}", flush=True)

print("\n=== Gradient boosting sweep (different nonlinear model class) ===", flush=True)
gb_configs = [
    {"max_depth": 2, "max_iter": 100, "learning_rate": 0.05, "l2_regularization": 1.0},
    {"max_depth": 3, "max_iter": 100, "learning_rate": 0.05, "l2_regularization": 1.0},
    {"max_depth": 3, "max_iter": 200, "learning_rate": 0.03, "l2_regularization": 5.0},
    {"max_depth": 4, "max_iter": 100, "learning_rate": 0.05, "l2_regularization": 5.0},
]
best_gb = None
best_gb_val_r2 = -np.inf
for cfg in gb_configs:
    gb = HistGradientBoostingRegressor(random_state=SEED, **cfg)
    gb.fit(X_train, y_train)  # tree models don't need scaling
    val_r2 = r2_score(y_val, gb.predict(X_val))
    print(f"  {cfg} -> val_R2={val_r2:+.4f}", flush=True)
    if val_r2 > best_gb_val_r2:
        best_gb_val_r2 = val_r2
        best_gb = gb
        best_gb_cfg = cfg

print(f"\nBest GB config: {best_gb_cfg} (val_R2={best_gb_val_r2:+.4f})", flush=True)
gb_test_pred = best_gb.predict(X_test)
gb_test_r2 = r2_score(y_test, gb_test_pred)
gb_test_r = np.corrcoef(gb_test_pred, y_test)[0, 1] if np.std(gb_test_pred) > 0 else 0.0
print(f"Best GB on held-out test: test_R2={gb_test_r2:+.4f}  test_r={gb_test_r:+.4f}", flush=True)

print("\n=== Summary (layer " + LAYER + ", same held-out test set) ===", flush=True)
print(f"  Ridge (linear):        test_R2={ridge_test_r2:+.4f}  test_r={ridge_test_r:+.4f}", flush=True)
print(f"  MLP (tuned):           test_R2={mlp_test_r2:+.4f}  test_r={mlp_test_r:+.4f}", flush=True)
print(f"  Gradient boosting:     test_R2={gb_test_r2:+.4f}  test_r={gb_test_r:+.4f}", flush=True)
print("\nDone.", flush=True)
