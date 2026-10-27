"""
Retrains the winning Ridge probe (layer 22, the best-performing single layer
from the sweeps) on the FULL 35k-point dataset - no held-out split this time,
since we already have honest generalization numbers from the sweeps and this
is the deployed artifact, not another evaluation.

Exports just the raw weights (scaler mean/scale + ridge coef/intercept) as
JSON, so the live app can apply the probe with a plain dot product and never
needs sklearn (or this training script) at serving time.
"""
import json

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

LAYER = "22"
IN_PATH = "overnight_merged.jsonl"
OUT_PATH = "entropy_probe_layer22.json"

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
from sklearn.metrics import r2_score
print(f"in-sample R2 (not held-out, just a sanity check): {r2_score(y, train_pred):.4f}", flush=True)

artifact = {
    "layer": 22,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "coef": ridge.coef_.tolist(),
    "intercept": float(ridge.intercept_),
    "alpha": float(ridge.alpha_),
    "n_train_points": len(records),
}
with open(OUT_PATH, "w") as f:
    json.dump(artifact, f)
print(f"\nExported probe to {OUT_PATH}", flush=True)
