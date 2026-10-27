"""
Analysis for fork_detection_benchmark.py: how well does each heuristic find
real forks (positions where committing the token shifts the final-answer
distribution)?

  - within-trace Spearman rho(heuristic, fork strength) on the random
    positions, averaged over traces (each trace ranks its own positions)
  - mean fork strength of each heuristic's top-3 picks vs random positions
  - share of picks that are "real forks": fork TV above the 95th percentile
    of a null where both sides are resampled from the same distribution

Usage: python analyze_fork_detection.py fork_det_w0.jsonl ...
"""
import json
import random
import sys
from collections import Counter

import numpy as np

R = [json.loads(l) for p in sys.argv[1:] for l in open(p)]
HEUR = [h for h in ["probe", "local", "sem_local", "product", "sem_product", "rind", "rind_x_outcome",
                    "product_x_lookdiv", "local_x_lookdiv", "product_x_nextdiv", "earliness"] if h in R[0]["top_positions"]]
DERIVED = {"product_x_lookdiv": ("product", "lookdiv"), "local_x_lookdiv": ("local", "lookdiv"),
           "product_x_nextdiv": ("product", "nextdiv")}


def hv(p, h):
    """A heuristic's value at a stored position (gated scores are base x gate)."""
    if h in DERIVED:
        b, g = DERIVED[h]
        return p[b] * (p.get(g) or 0.0)
    return p[h]


def dist(ans):
    c = Counter("none" if a is None else round(a, 4) for a in ans)
    return {k: v / len(ans) for k, v in c.items()}


def tv(p, q):
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))


# null: both sides drawn from the pooled answers at that position
rng = random.Random(0)
null = []
for r in R:
    for t in r["random_positions"]:
        d = r["positions"][str(t)]
        pool = d["ans_before"] + d["ans_after"]
        for _ in range(3):
            a = [rng.choice(pool) for _ in range(len(d["ans_before"]))]
            b = [rng.choice(pool) for _ in range(len(d["ans_after"]))]
            null.append(tv(dist(a), dist(b)))
thr = float(np.quantile(null, 0.95))
rand = [r["positions"][str(t)]["fork_tv"] for r in R for t in r["random_positions"]]
print(f"{len(R)} wrong traces, {len(rand)} random positions")
print(f"noise floor: null TV mean {np.mean(null):.3f}, 95th pct {thr:.3f} -> 'real fork' = TV > {thr:.3f}")
print(f"random positions: mean fork TV {np.mean(rand):.3f}, real forks {np.mean([x > thr for x in rand]):.1%}\n")


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


print(f"{'heuristic':12s} {'within-trace rho':>17s} {'top-3 mean TV':>14s} {'top-3 real forks':>17s} {'lift vs random':>15s}")
for h in HEUR:
    rhos = []
    for r in R:
        pts = [r["positions"][str(t)] for t in r["random_positions"]]
        if len(pts) >= 6:
            rho = spearman([hv(p, h) for p in pts], [p["fork_tv"] for p in pts])
            if not np.isnan(rho):
                rhos.append(rho)
    picks = [r["positions"][str(t)]["fork_tv"] for r in R for t in r["top_positions"][h]]
    real = np.mean([x > thr for x in picks])
    print(f"{h:12s} {np.mean(rhos):+17.3f} {np.mean(picks):14.3f} {real:17.1%} {real / max(1e-9, np.mean([x > thr for x in rand])):14.2f}x")

def partial_rank(x, y, z):
    """Spearman of x and y after regressing both ranks on z's rank (position control)."""
    rx, ry, rz = (np.argsort(np.argsort(v)).astype(float) for v in (x, y, z))
    if np.std(rz) == 0:
        return np.nan
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    ex, ey = res(rx), res(ry)
    if np.std(ex) == 0 or np.std(ey) == 0:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


print("\nposition-controlled within-trace rho (heuristic vs fork TV, partialling out position):")
for h in [x for x in HEUR if x != "earliness"]:
    vals = []
    for r in R:
        pts = [(t, r["positions"][str(t)]) for t in r["random_positions"]]
        if len(pts) >= 6:
            v = partial_rank([hv(p, h) for _, p in pts], [p["fork_tv"] for _, p in pts], [t for t, _ in pts])
            if not np.isnan(v):
                vals.append(v)
    pos_corr = np.mean([spearman([hv(p, h) for p in [r["positions"][str(t)] for t in r["random_positions"]]],
                                 r["random_positions"]) for r in R if len(r["random_positions"]) >= 6])
    print(f"  {h:12s} partial rho {np.mean(vals):+.3f}   (its own rho with position: {pos_corr:+.3f})")

print("\nexample top-3 picks per heuristic (first 3 traces): token | fork TV")
for r in R[:3]:
    print(f"  trace #{r['math_test_index']}")
    for h in HEUR:
        s = ", ".join(f"{r['positions'][str(t)]['token']!r}@{t}:{r['positions'][str(t)]['fork_tv']:.2f}" for t in r["top_positions"][h])
        print(f"    {h:12s} {s}")
