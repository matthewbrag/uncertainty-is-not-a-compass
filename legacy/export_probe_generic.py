"""
Generic version of export_entropy_probe.py - trains on ALL points (no
held-out split, since generalization was already validated separately) and
exports a model-specific probe artifact.

Usage: python export_probe_generic.py <model_key> <in_path> <layer> <out_path>
"""
import json
import sys

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

MODEL_KEY = sys.argv[1]
IN_PATH = sys.argv[2]
LAYER = sys.argv[3]
OUT_PATH = sys.argv[4]

print("Loading data...", flush=True)
records = []
with open(IN_PATH) as f:
    for line in f:
        records.append(json.loads(line))
print(f"{len(records)} points loaded", flush=True)

X = np.array([r["hidden_states"][LAYER] for r in records], dtype=np.float32)
y = np.array([r["outcome_entropy"] for r in records])

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

ridge = RidgeCV(alphas=np.logspace(-2, 8, 30))
ridge.fit(X_s, y)
print(f"chosen alpha: {ridge.alpha_}", flush=True)

train_pred = ridge.predict(X_s)
print(f"in-sample R2 (not held-out, just a sanity check): {r2_score(y, train_pred):.4f}", flush=True)

artifact = {
    "model": MODEL_KEY,
    "layer": int(LAYER),
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "coef": ridge.coef_.tolist(),
    "intercept": float(ridge.intercept_),
    "alpha": float(ridge.alpha_),
    "n_train_points": len(records),
}
with open(OUT_PATH, "w") as f:
    json.dump(artifact, f)
print(f"\nExported probe to {OUT_PATH}")
