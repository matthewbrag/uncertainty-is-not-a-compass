"""
Trains probes to predict outcome_entropy from the full hidden-state vectors
collected overnight, testing the hypothesis that raw scalar entropy (tested
all night, consistently null) was throwing away signal that's actually
present in the full representation.

Splits by PROBLEM (not by individual point) so candidate points from the same
trace never leak across train/test - several points per problem share
context and would otherwise inflate held-out performance artificially.

For each of the 7 collected layers (and a concatenated-all-layers variant):
  - Ridge regression (linear probe) - baseline, is there ANY linear signal
  - Small MLP (nonlinear) - is there signal a linear probe can't see
Reports both regression (R^2, Pearson r) and binary classification framing
(is outcome_entropy > 0 at all - arguably the more practically useful target
given ~70% of points are exactly zero) via ROC-AUC.
"""
import json

import numpy as np
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score

LAYERS = [6, 10, 14, 18, 22, 26, 28]
IN_PATH = "overnight_merged.jsonl"
TEST_FRAC = 0.1
SEED = 0

print("Loading data...")
records = []
with open(IN_PATH) as f:
    for line in f:
        records.append(json.loads(line))
print(f"{len(records)} points loaded")

# split by problem, not by point
problem_keys = sorted(set((r["shard"], r["gsm8k_train_index"]) for r in records))
rng = np.random.default_rng(SEED)
rng.shuffle(problem_keys)
n_test_problems = int(len(problem_keys) * TEST_FRAC)
test_problems = set(problem_keys[:n_test_problems])
train_problems = set(problem_keys[n_test_problems:])
print(f"{len(train_problems)} train problems, {len(test_problems)} test problems")

train_records = [r for r in records if (r["shard"], r["gsm8k_train_index"]) in train_problems]
test_records = [r for r in records if (r["shard"], r["gsm8k_train_index"]) in test_problems]
print(f"{len(train_records)} train points, {len(test_records)} test points")

y_train = np.array([r["outcome_entropy"] for r in train_records])
y_test = np.array([r["outcome_entropy"] for r in test_records])
y_train_binary = (y_train > 0).astype(int)
y_test_binary = (y_test > 0).astype(int)
print(f"train nonzero rate: {y_train_binary.mean():.3f}  test nonzero rate: {y_test_binary.mean():.3f}")


def build_X(records, layer_key):
    return np.array([r["hidden_states"][layer_key] for r in records], dtype=np.float32)


def evaluate_layer(layer_key, label):
    X_train = build_X(train_records, layer_key)
    X_test = build_X(test_records, layer_key)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # Ridge (linear probe)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 20))
    ridge.fit(X_train_s, y_train)
    pred = ridge.predict(X_test_s)
    r2 = r2_score(y_test, pred)
    pearson_r = np.corrcoef(pred, y_test)[0, 1] if np.std(pred) > 0 else 0.0
    results["ridge_r2"] = r2
    results["ridge_pearson_r"] = pearson_r

    # small MLP (nonlinear probe)
    mlp = MLPRegressor(hidden_layer_sizes=(128,), alpha=1e-3, max_iter=300,
                        early_stopping=True, n_iter_no_change=10, random_state=SEED)
    mlp.fit(X_train_s, y_train)
    pred_mlp = mlp.predict(X_test_s)
    r2_mlp = r2_score(y_test, pred_mlp)
    pearson_r_mlp = np.corrcoef(pred_mlp, y_test)[0, 1] if np.std(pred_mlp) > 0 else 0.0
    results["mlp_r2"] = r2_mlp
    results["mlp_pearson_r"] = pearson_r_mlp

    # binary framing: is outcome_entropy > 0 at all (logistic on same features)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train_s, y_train_binary)
    proba = clf.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test_binary, proba) if len(set(y_test_binary)) > 1 else float("nan")
    results["logistic_auc"] = auc

    print(f"{label:20} ridge_R2={r2:+.4f} ridge_r={pearson_r:+.4f}  "
          f"mlp_R2={r2_mlp:+.4f} mlp_r={pearson_r_mlp:+.4f}  nonzero_AUC={auc:.4f}")
    return results


print("\n=== Per-layer probe results (held-out test, split by problem) ===")
all_results = {}
for L in LAYERS:
    all_results[L] = evaluate_layer(str(L), f"layer {L}")

print("\n=== Concatenated all-layers probe ===")


def build_X_concat(records):
    return np.array(
        [np.concatenate([r["hidden_states"][str(L)] for L in LAYERS]) for r in records],
        dtype=np.float32,
    )


X_train_concat = build_X_concat(train_records)
X_test_concat = build_X_concat(test_records)
scaler = StandardScaler()
X_train_concat_s = scaler.fit_transform(X_train_concat)
X_test_concat_s = scaler.transform(X_test_concat)

ridge = RidgeCV(alphas=np.logspace(-2, 4, 20))
ridge.fit(X_train_concat_s, y_train)
pred = ridge.predict(X_test_concat_s)
r2 = r2_score(y_test, pred)
pearson_r = np.corrcoef(pred, y_test)[0, 1] if np.std(pred) > 0 else 0.0

mlp = MLPRegressor(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=300,
                    early_stopping=True, n_iter_no_change=10, random_state=SEED)
mlp.fit(X_train_concat_s, y_train)
pred_mlp = mlp.predict(X_test_concat_s)
r2_mlp = r2_score(y_test, pred_mlp)
pearson_r_mlp = np.corrcoef(pred_mlp, y_test)[0, 1] if np.std(pred_mlp) > 0 else 0.0

clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(X_train_concat_s, y_train_binary)
proba = clf.predict_proba(X_test_concat_s)[:, 1]
auc = roc_auc_score(y_test_binary, proba)

print(f"{'all-layers':20} ridge_R2={r2:+.4f} ridge_r={pearson_r:+.4f}  "
      f"mlp_R2={r2_mlp:+.4f} mlp_r={pearson_r_mlp:+.4f}  nonzero_AUC={auc:.4f}")

print("\nDone.")
