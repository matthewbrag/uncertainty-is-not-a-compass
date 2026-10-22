"""
Does the Ridge-on-full-hidden-state signal (the thing that survived all
night, after embedding similarity / raw entropy / attention all failed)
replicate on Qwen3-1.7B, or was it a 0.6B-specific quirk?

Same methodology as the original 0.6B sweeps: problem-level held-out split
(never split by individual candidate point, to avoid leakage between
correlated points from the same trace), same layer set, same seed.
"""
import json

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

IN_PATH = "overnight_qwen3-1.7b_shard0.jsonl"
LAYERS = ["6", "10", "14", "18", "22", "26", "28"]
TEST_FRAC = 0.1
SEED = 0

print("Loading data...", flush=True)
records = []
with open(IN_PATH) as f:
    for line in f:
        records.append(json.loads(line))
print(f"{len(records)} points loaded\n", flush=True)

problem_keys = sorted(set((r["shard"], r["gsm8k_train_index"]) for r in records))
rng = np.random.default_rng(SEED)
rng.shuffle(problem_keys)
n_test = int(len(problem_keys) * TEST_FRAC)
test_problems = set(problem_keys[:n_test])
train_problems = set(problem_keys[n_test:])

train_records = [r for r in records if (r["shard"], r["gsm8k_train_index"]) in train_problems]
test_records = [r for r in records if (r["shard"], r["gsm8k_train_index"]) in test_problems]
print(f"train_problems={len(train_problems)} ({len(train_records)} points)  "
      f"test_problems={len(test_problems)} ({len(test_records)} points)\n", flush=True)

y_train = np.array([r["outcome_entropy"] for r in train_records])
y_test = np.array([r["outcome_entropy"] for r in test_records])

print(f"{'layer':>6}  {'test_R2':>8}  {'test_r':>8}")
best = None
for L in LAYERS:
    X_train = np.array([r["hidden_states"][L] for r in train_records], dtype=np.float32)
    X_test = np.array([r["hidden_states"][L] for r in test_records], dtype=np.float32)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    ridge = RidgeCV(alphas=np.logspace(-2, 8, 30))
    ridge.fit(X_train_s, y_train)
    pred = ridge.predict(X_test_s)
    r2 = r2_score(y_test, pred)
    r = np.corrcoef(pred, y_test)[0, 1]
    print(f"{L:>6}  {r2:+8.4f}  {r:+8.4f}", flush=True)
    if best is None or r2 > best[1]:
        best = (L, r2, r)

print(f"\nbest layer: {best[0]}  test_R2={best[1]:+.4f}  test_r={best[2]:+.4f}")
print(f"\nfor reference, 0.6b at layer 22 (unweighted target): test_R2=+0.0695  test_r=+0.2646")
