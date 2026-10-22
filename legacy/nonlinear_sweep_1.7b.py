"""
Does a nonlinear model beat the linear Ridge probe for predicting outcome_p
on 1.7B? Same question as the 0.6B bottleneck_sweep.py, but targeting
outcome_p (the thing that actually matters for search usefulness, per the
signed-entropy finding) instead of outcome_entropy, and on the bigger model.

Proper 3-way split this time (train/val/test by problem, not just
train/test) to avoid the exact multiple-comparisons trap the 0.6B sweep
demonstrated live: pick the best config on val, report ONLY that config's
test performance as the real number.
"""
import json

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

LAYER = "22"
IN_PATH = "overnight_qwen3-1.7b_shard0.jsonl"
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
n = len(problem_keys)
n_test = int(n * 0.15)
n_val = int(n * 0.15)
test_problems = set(problem_keys[:n_test])
val_problems = set(problem_keys[n_test:n_test + n_val])
train_problems = set(problem_keys[n_test + n_val:])

def subset(probset):
    return [r for r in records if (r["shard"], r["gsm8k_train_index"]) in probset]

train_records = subset(train_problems)
val_records = subset(val_problems)
test_records = subset(test_problems)
print(f"train={len(train_records)}  val={len(val_records)}  test={len(test_records)}\n", flush=True)

def build_Xy(recs):
    X = np.array([r["hidden_states"][LAYER] for r in recs], dtype=np.float32)
    y = np.array([r["outcome_p"] for r in recs])
    return X, y

X_train, y_train = build_Xy(train_records)
X_val, y_val = build_Xy(val_records)
X_test, y_test = build_Xy(test_records)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

configs = {}

ridge = RidgeCV(alphas=np.logspace(-2, 8, 30))
ridge.fit(X_train_s, y_train)
configs["ridge"] = ridge

mlp_small = MLPRegressor(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=2000, random_state=SEED, early_stopping=True)
mlp_small.fit(X_train_s, y_train)
configs["mlp_32"] = mlp_small

mlp_bottleneck = MLPRegressor(hidden_layer_sizes=(64, 8), alpha=1e-2, max_iter=2000, random_state=SEED, early_stopping=True)
mlp_bottleneck.fit(X_train_s, y_train)
configs["mlp_64_8"] = mlp_bottleneck

gbr = HistGradientBoostingRegressor(max_depth=4, random_state=SEED)
gbr.fit(X_train_s, y_train)
configs["gradient_boosting"] = gbr

print(f"{'config':20s} {'val_R2':>8s} {'val_r':>8s}")
val_scores = {}
for name, model in configs.items():
    pred = model.predict(X_val_s)
    r2 = r2_score(y_val, pred)
    r = np.corrcoef(pred, y_val)[0, 1]
    val_scores[name] = r2
    print(f"{name:20s} {r2:+8.4f} {r:+8.4f}", flush=True)

best_name = max(val_scores, key=val_scores.get)
print(f"\nbest on val: {best_name} (val_R2={val_scores[best_name]:+.4f})")

best_model = configs[best_name]
test_pred = best_model.predict(X_test_s)
test_r2 = r2_score(y_test, test_pred)
test_r = np.corrcoef(test_pred, y_test)[0, 1]
print(f"\n=== HONEST TEST-SET RESULT (only reporting the val-winner's test score) ===")
print(f"{best_name}: test_R2={test_r2:+.4f}  test_r={test_r:+.4f}")

ridge_test_pred = configs["ridge"].predict(X_test_s)
ridge_test_r2 = r2_score(y_test, ridge_test_pred)
ridge_test_r = np.corrcoef(ridge_test_pred, y_test)[0, 1]
print(f"ridge (for reference, regardless of val outcome): test_R2={ridge_test_r2:+.4f}  test_r={ridge_test_r:+.4f}")
