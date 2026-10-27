"""
Re-runs the model-architecture comparison (Ridge / MLP / gradient boosting /
extreme-bottleneck MLP) against the CORRECTED diversity_entropy target,
instead of the old binary outcome_entropy target every prior run of this
comparison used. Same methodology (held-out problem-level split, pick the
val-winner, report its honest test score) for a clean apples-to-apples
comparison against the old numbers.

Usage: python experiments/2_estimators/diversity_architecture_compare.py <in_path> <layer> <label>
"""
import json
import math
import sys
from collections import Counter

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

IN_PATH = sys.argv[1]
LAYER = sys.argv[2]
LABEL = sys.argv[3] if len(sys.argv) > 3 else IN_PATH


def diversity_entropy(answers):
    vals = [a if a is not None else "unparsed" for a in answers]
    counts = Counter(vals)
    n = len(vals)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h


records = [json.loads(l) for l in open(IN_PATH)]
print(f"[{LABEL}] {len(records)} points loaded (layer {LAYER})", flush=True)

keyfield = "gsm8k_train_index" if "gsm8k_train_index" in records[0] else "math_train_index"
keys = sorted(set(r[keyfield] for r in records))
rng = np.random.default_rng(0)
rng.shuffle(keys)
n_test = int(len(keys) * 0.1)
n_val = int(len(keys) * 0.15)
test_keys = set(keys[:n_test])
val_keys = set(keys[n_test:n_test + n_val])
train_recs = [r for r in records if r[keyfield] not in test_keys and r[keyfield] not in val_keys]
val_recs = [r for r in records if r[keyfield] in val_keys]
test_recs = [r for r in records if r[keyfield] in test_keys]
print(f"[{LABEL}] train={len(train_recs)} val={len(val_recs)} test={len(test_recs)}", flush=True)

X_train = np.array([r["hidden_states"][LAYER] for r in train_recs], dtype=np.float32)
X_val = np.array([r["hidden_states"][LAYER] for r in val_recs], dtype=np.float32)
X_test = np.array([r["hidden_states"][LAYER] for r in test_recs], dtype=np.float32)
y_train = np.array([diversity_entropy(r["answers"]) for r in train_recs])
y_val = np.array([diversity_entropy(r["answers"]) for r in val_recs])
y_test = np.array([diversity_entropy(r["answers"]) for r in test_recs])

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

results = {}

ridge = RidgeCV(alphas=np.logspace(-2, 8, 30))
ridge.fit(X_train_s, y_train)
results["ridge"] = ridge

mlp32 = MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, early_stopping=True,
                      n_iter_no_change=15, alpha=1.0, random_state=0)
mlp32.fit(X_train_s, y_train)
results["mlp_32"] = mlp32

mlp64_8 = MLPRegressor(hidden_layer_sizes=(64, 8), max_iter=500, early_stopping=True,
                        n_iter_no_change=15, alpha=1.0, random_state=0)
mlp64_8.fit(X_train_s, y_train)
results["mlp_64_8"] = mlp64_8

gb = GradientBoostingRegressor(max_depth=4, random_state=0)
gb.fit(X_train_s, y_train)
results["gradient_boosting"] = gb

# extreme bottleneck sweep (1-4 hidden units): does a near-1D projection
# already capture most of the signal Ridge finds?
for size in [1, 2, 4]:
    for alpha in [0.3, 1.0, 3.0]:
        name = f"bottleneck_{size}_a{alpha}"
        m = MLPRegressor(hidden_layer_sizes=(size,), max_iter=500, early_stopping=True,
                          n_iter_no_change=15, alpha=alpha, random_state=0)
        m.fit(X_train_s, y_train)
        results[name] = m

print(f"\n[{LABEL}] {'config':22s} {'val_R2':>8s} {'val_r':>8s}", flush=True)
best_val_r2 = -np.inf
best_name = None
for name, model in results.items():
    pred = model.predict(X_val_s)
    v_r2 = r2_score(y_val, pred)
    v_r = np.corrcoef(pred, y_val)[0, 1] if np.std(pred) > 0 else 0.0
    print(f"[{LABEL}] {name:22s} {v_r2:+.4f}  {v_r:+.4f}", flush=True)
    if v_r2 > best_val_r2:
        best_val_r2 = v_r2
        best_name = name

print(f"\n[{LABEL}] best on val: {best_name} (val_R2={best_val_r2:+.4f})", flush=True)
best_model = results[best_name]
test_pred = best_model.predict(X_test_s)
test_r2 = r2_score(y_test, test_pred)
test_r = np.corrcoef(test_pred, y_test)[0, 1] if np.std(test_pred) > 0 else 0.0

ridge_test_pred = ridge.predict(X_test_s)
ridge_test_r2 = r2_score(y_test, ridge_test_pred)
ridge_test_r = np.corrcoef(ridge_test_pred, y_test)[0, 1]

bottleneck1_best = max(
    (n for n in results if n.startswith("bottleneck_1_")),
    key=lambda n: r2_score(y_val, results[n].predict(X_val_s)),
)
b1_test_pred = results[bottleneck1_best].predict(X_test_s)
b1_test_r2 = r2_score(y_test, b1_test_pred)
b1_test_r = np.corrcoef(b1_test_pred, y_test)[0, 1] if np.std(b1_test_pred) > 0 else 0.0

print(f"\n[{LABEL}] === HONEST TEST-SET RESULTS ===", flush=True)
print(f"[{LABEL}] val-winner ({best_name}): test_R2={test_r2:+.4f}  test_r={test_r:+.4f}", flush=True)
print(f"[{LABEL}] ridge (for reference):    test_R2={ridge_test_r2:+.4f}  test_r={ridge_test_r:+.4f}", flush=True)
print(f"[{LABEL}] best single-unit bottleneck ({bottleneck1_best}): test_R2={b1_test_r2:+.4f}  test_r={b1_test_r:+.4f}", flush=True)
# every config's test score too (the article table reports Ridge, boosting and
# the best 1-4 unit bottleneck side by side, not just the val-winner)
print(f"\n[{LABEL}] test R2 for every config:", flush=True)
for name, m in results.items():
    pred = m.predict(X_test_s)
    print(f"[{LABEL}]   {name:22s} test_R2={r2_score(y_test, pred):+.4f}  test_r={np.corrcoef(pred, y_test)[0, 1]:+.4f}", flush=True)
print(f"[{LABEL}] Done.", flush=True)
