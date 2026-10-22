"""
Tests whether probability-weighting the outcome-entropy target (using each
forced branch's actual top-5 probability, already recorded, instead of
treating all 5 branches as equally-weighted) produces a cleaner training
target - entirely from existing data, zero new generation.

Same layer (22), same problem-level train/val/test split (same seeds) as
the earlier sweeps, so results are directly comparable to the unweighted
Ridge baseline (test_R2=+0.0688, test_r=+0.2632).
"""
import json
import math

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

LAYER = "22"
IN_PATH = "overnight_merged.jsonl"
TEST_FRAC = 0.1
SEED = 0


def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


print("Loading data...", flush=True)
records = []
with open(IN_PATH) as f:
    for line in f:
        records.append(json.loads(line))
print(f"{len(records)} points loaded", flush=True)

n_mismatched = 0
for r in records:
    ref = r["reference_answer"]
    probs = r["forced_token_probs"]
    answers = r["answers"]
    if len(probs) != len(answers):
        n_mismatched += 1
        r["outcome_entropy_weighted"] = r["outcome_entropy"]
        continue
    total_prob = sum(probs)
    if total_prob <= 0:
        r["outcome_entropy_weighted"] = r["outcome_entropy"]
        continue
    weighted_correct = 0.0
    for p, a in zip(probs, answers):
        correct = a is not None and ref is not None and abs(a - ref) < 1e-4
        weighted_correct += p * (1.0 if correct else 0.0)
    p_weighted = weighted_correct / total_prob
    r["outcome_entropy_weighted"] = binary_entropy(p_weighted)

print(f"records with mismatched probs/answers lengths (fell back to unweighted): {n_mismatched}", flush=True)

# sanity: how much do the two targets actually differ?
orig = np.array([r["outcome_entropy"] for r in records])
weighted = np.array([r["outcome_entropy_weighted"] for r in records])
diff = np.abs(orig - weighted)
print(f"mean |orig - weighted|: {diff.mean():.4f}  (0 would mean no effect at all)", flush=True)
print(f"correlation between the two targets: {np.corrcoef(orig, weighted)[0,1]:.4f}", flush=True)

# same split as before
problem_keys = sorted(set((r["shard"], r["gsm8k_train_index"]) for r in records))
rng = np.random.default_rng(SEED)
rng.shuffle(problem_keys)
n_test = int(len(problem_keys) * TEST_FRAC)
test_problems = set(problem_keys[:n_test])
train_problems = set(problem_keys[n_test:])

train_records = [r for r in records if (r["shard"], r["gsm8k_train_index"]) in train_problems]
test_records = [r for r in records if (r["shard"], r["gsm8k_train_index"]) in test_problems]


def build_X(recs):
    return np.array([r["hidden_states"][LAYER] for r in recs], dtype=np.float32)


X_train = build_X(train_records)
X_test = build_X(test_records)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("\n=== Unweighted target (original, for reference) ===", flush=True)
y_train = np.array([r["outcome_entropy"] for r in train_records])
y_test = np.array([r["outcome_entropy"] for r in test_records])
ridge = RidgeCV(alphas=np.logspace(-2, 6, 25))
ridge.fit(X_train_s, y_train)
pred = ridge.predict(X_test_s)
print(f"test_R2={r2_score(y_test, pred):+.4f}  test_r={np.corrcoef(pred, y_test)[0,1]:+.4f}", flush=True)

print("\n=== Probability-weighted target ===", flush=True)
y_train_w = np.array([r["outcome_entropy_weighted"] for r in train_records])
y_test_w = np.array([r["outcome_entropy_weighted"] for r in test_records])
ridge_w = RidgeCV(alphas=np.logspace(-2, 6, 25))
ridge_w.fit(X_train_s, y_train_w)
pred_w = ridge_w.predict(X_test_s)
print(f"test_R2={r2_score(y_test_w, pred_w):+.4f}  test_r={np.corrcoef(pred_w, y_test_w)[0,1]:+.4f}", flush=True)

print("\nDone.", flush=True)
