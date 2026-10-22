"""
Follow-up to tune_entropy_head.py: tests whether an extreme bottleneck (as
few as 1-4 hidden units) does about as well as the wider nets did. If so,
that's evidence the real signal lives in a low-dimensional subspace of the
1024-dim hidden state - which would also explain why Ridge (effectively a
rank-1-ish linear projection) already captures it, and the wider/deeper nets
in the previous sweep weren't finding anything Ridge couldn't.

Same data, same problem-level train/val/test split (same seeds) as
tune_entropy_head.py, so results are directly comparable.
"""
import json

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

LAYER = "22"
IN_PATH = "overnight_merged.jsonl"
TEST_FRAC = 0.1
VAL_FRAC = 0.15
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

X_train = build_X(train_records)
X_val = build_X(val_records)
X_test = build_X(test_records)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

ridge = RidgeCV(alphas=np.logspace(-2, 4, 20))
ridge.fit(X_train_s, y_train)
ridge_test_pred = ridge.predict(X_test_s)
ridge_test_r2 = r2_score(y_test, ridge_test_pred)
ridge_test_r = np.corrcoef(ridge_test_pred, y_test)[0, 1]
print(f"\nRidge baseline (for reference): test_R2={ridge_test_r2:+.4f}  test_r={ridge_test_r:+.4f}", flush=True)

print("\n=== Bottleneck MLP sweep (forcing compression) ===", flush=True)
configs = []
for size in [1, 2, 3, 4, 6, 8]:
    for alpha in [0.3, 1.0, 3.0]:
        configs.append({"hidden_layer_sizes": (size,), "alpha": alpha})
# also a couple of two-layer bottlenecks: wide-then-narrow, forcing compression at the second layer
for sizes in [(16, 2), (16, 4), (32, 4)]:
    for alpha in [1.0, 3.0]:
        configs.append({"hidden_layer_sizes": sizes, "alpha": alpha})

best_val_r2 = -np.inf
best_cfg = None
best_model = None
all_results = []
for cfg in configs:
    mlp = MLPRegressor(max_iter=500, early_stopping=True, n_iter_no_change=15,
                        random_state=SEED, **cfg)
    mlp.fit(X_train_s, y_train)
    val_r2 = r2_score(y_val, mlp.predict(X_val_s))
    all_results.append((cfg, val_r2))
    print(f"  {cfg} -> val_R2={val_r2:+.4f}", flush=True)
    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        best_cfg = cfg
        best_model = mlp

print(f"\nBest bottleneck config: {best_cfg} (val_R2={best_val_r2:+.4f})", flush=True)
test_pred = best_model.predict(X_test_s)
test_r2 = r2_score(y_test, test_pred)
test_r = np.corrcoef(test_pred, y_test)[0, 1] if np.std(test_pred) > 0 else 0.0
print(f"Best bottleneck on held-out test: test_R2={test_r2:+.4f}  test_r={test_r:+.4f}", flush=True)

print(f"\n=== Comparison (layer {LAYER}, same held-out test set) ===", flush=True)
print(f"  Ridge (linear, 1024-dim):        test_R2={ridge_test_r2:+.4f}  test_r={ridge_test_r:+.4f}", flush=True)
print(f"  Best bottleneck MLP:             test_R2={test_r2:+.4f}  test_r={test_r:+.4f}  "
      f"(hidden_layer_sizes={best_cfg['hidden_layer_sizes']})", flush=True)
print("\nDone.", flush=True)
