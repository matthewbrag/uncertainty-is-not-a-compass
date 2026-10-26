"""
Exports a probe on the diversity_entropy target (Shannon entropy over the
distinct final answers of the 5 forced branches, unparseable answers sharing
one bucket) -- the ground truth the article actually uses. Same Ridge setup
as export_probe_generic.py / export_probe_p_target.py, which only covered the
binary outcome_entropy and outcome_p targets. Also reports a held-out
(split by problem) correlation so the Part 1 table can be refreshed.

Usage: python export_probe_diversity.py <model_key> <in_path> <layer> <out_path>
"""
import json
import math
import sys
from collections import Counter

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

MODEL_KEY = sys.argv[1]
IN_PATH = sys.argv[2]
LAYER = sys.argv[3]
OUT_PATH = sys.argv[4]


def diversity_entropy(answers):
    vals = [a if a is not None else "unparsed" for a in answers]
    counts = Counter(vals)
    n = len(vals)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h


print("Loading data...", flush=True)
X_list, y_list, keys = [], [], []
with open(IN_PATH) as f:
    for line in f:
        r = json.loads(line)
        X_list.append(r["hidden_states"][LAYER])
        y_list.append(diversity_entropy(r["answers"]))
        keys.append(r["math_train_index"])
X = np.array(X_list, dtype=np.float32)
y = np.array(y_list)
keys = np.array(keys)
print(f"{len(y)} points loaded, mean diversity_entropy={y.mean():.3f}", flush=True)

# held-out check, split by problem so traces never straddle train/test
uniq = np.unique(keys)
rng = np.random.default_rng(0)
rng.shuffle(uniq)
test_keys = set(uniq[: len(uniq) // 5].tolist())
is_test = np.array([k in test_keys for k in keys])
sc = StandardScaler().fit(X[~is_test])
held = RidgeCV(alphas=np.logspace(-2, 8, 30)).fit(sc.transform(X[~is_test]), y[~is_test])
pred = held.predict(sc.transform(X[is_test]))
print(f"held-out (by problem) r={np.corrcoef(pred, y[is_test])[0, 1]:.4f}  "
      f"R2={r2_score(y[is_test], pred):.4f}  (n_test={is_test.sum()})", flush=True)

# final artifact trains on everything
scaler = StandardScaler()
X_s = scaler.fit_transform(X)
ridge = RidgeCV(alphas=np.logspace(-2, 8, 30))
ridge.fit(X_s, y)
print(f"chosen alpha: {ridge.alpha_}", flush=True)

artifact = {
    "model": MODEL_KEY,
    "layer": int(LAYER),
    "target": "diversity_entropy",
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "coef": ridge.coef_.tolist(),
    "intercept": float(ridge.intercept_),
    "alpha": float(ridge.alpha_),
    "n_train_points": len(y),
}
with open(OUT_PATH, "w") as f:
    json.dump(artifact, f)
print(f"\nExported probe to {OUT_PATH}")
