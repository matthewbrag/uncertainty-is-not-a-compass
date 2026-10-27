"""
Exports a probe on the diversity_entropy target (Shannon entropy over the
distinct final answers of the 5 forced branches, unparseable answers sharing
one bucket) -- the ground truth the article actually uses. Same Ridge setup
as export_probe_generic.py / export_probe_p_target.py, which only covered the
binary outcome_entropy and outcome_p targets. Also reports a held-out
(split by problem) correlation so the Part 1 table can be refreshed.

Usage: python experiments/2_estimators/export_probe_diversity.py <model_key> <in_path> <layer|best> <out_path>
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
X_by_layer, y_list, keys, final_entropy = {}, [], [], []
with open(IN_PATH) as f:
    for line in f:
        r = json.loads(line)
        for L, v in r["hidden_states"].items():
            X_by_layer.setdefault(L, []).append(v)
        y_list.append(diversity_entropy(r["answers"]))
        keys.append(r.get("math_train_index", r.get("gsm8k_train_index")))
        final_entropy.append(r["layer_entropies"]["28"])  # the model's own next-token entropy
X_by_layer = {L: np.array(v, dtype=np.float32) for L, v in X_by_layer.items()}
y = np.array(y_list)
keys = np.array(keys)
print(f"{len(y)} points loaded, mean diversity_entropy={y.mean():.3f}", flush=True)
print(f"all 5 branches agree: {np.mean(y < 1e-9):.1%}   all 5 differ: {np.mean(y > 2.3):.1%}", flush=True)

# held-out check at every recorded layer, split by problem so traces never
# straddle train/test -- this is what the Part 1 table reports
uniq = np.unique(keys)
rng = np.random.default_rng(0)
rng.shuffle(uniq)
test_keys = set(uniq[: len(uniq) // 5].tolist())
is_test = np.array([k in test_keys for k in keys])
held_out_r = {}
for L in sorted(X_by_layer, key=int):
    XL = X_by_layer[L]
    sc = StandardScaler().fit(XL[~is_test])
    held = RidgeCV(alphas=np.logspace(-2, 8, 30)).fit(sc.transform(XL[~is_test]), y[~is_test])
    pred = held.predict(sc.transform(XL[is_test]))
    r_heldout = np.corrcoef(pred, y[is_test])[0, 1]
    held_out_r[L] = r_heldout
    print(f"layer {L:>2}: held-out (by problem) r={r_heldout:.4f}  "
          f"R2={r2_score(y[is_test], pred):.4f}  (n_test={is_test.sum()})", flush=True)
print(f"simple entropy (final layer, no training), same held-out problems: "
      f"r={np.corrcoef(np.array(final_entropy)[is_test], y[is_test])[0, 1]:.4f}", flush=True)
if LAYER == "best":
    # pick by held-out r on TRAIN-split problems only (tournament uses the
    # test split), same criterion the original layer choices were made on
    LAYER = max(held_out_r, key=held_out_r.get)
    print(f"best layer by held-out r: {LAYER}", flush=True)
X = X_by_layer[LAYER]

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
