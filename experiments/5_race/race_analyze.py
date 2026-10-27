"""
Analysis for race.py: replay policies over each problem's shared pool and
compare tokens to the first correct answer (perfect verifier assumed).

  resample      : fresh samples in order
  hybrid_<fork> : a fresh sample; if it is wrong (and among the first 4),
                  its 4 fork branches; then the next fresh sample ...
  hybrid_gated_pruned : gated, but only branches whose lookahead diverged
Gated policies are charged their lookahead tokens before their branches. The
probe/entropy forward pass (a prefill over the sample) is not charged.
Budgets stop at 8000: past ~16 samples the resample pool runs out.

Usage: python experiments/5_race/race_analyze.py race_w0.jsonl race_w1.jsonl race_w2.jsonl
"""
import json
import math
import sys

import numpy as np

R = [json.loads(l) for p in sys.argv[1:] for l in open(p)]
BUDGETS = [500, 1000, 2000, 3000, 4000, 6000, 8000]
CAP = 8000


def mcnemar(a, b):
    m = a + b
    return 1.0 if m == 0 else min(1.0, 2 * sum(math.comb(m, i) for i in range(min(a, b) + 1)) / 2 ** m)


def seq(r, policy):
    out = []
    for s in r["samples"]:
        out.append((s["length"], s["correct"]))
        if policy == "resample" or s["correct"]:
            continue
        name = policy.removeprefix("hybrid_").removesuffix("_free").removesuffix("_pruned")
        fork = s["forks"].get(name)
        if not fork:
            continue
        if fork["gate_tokens"] and not policy.endswith("_free"):
            out.append((fork["gate_tokens"], False))
        for b in fork["branches"]:
            if "_pruned" in policy and b["diverges"] is False:
                continue
            out.append((b["length"], b["correct"]))
    return out


def first_correct(sq):
    t = 0
    for L, ok in sq:
        t += L
        if ok:
            return t
    return None


# *_free: lookahead tokens not charged (upper bound on what a cheaper gate could do)
POLICIES = ["resample", "hybrid_gated", "hybrid_gated_pruned", "hybrid_gated_pruned_free", "hybrid_local", "hybrid_random"]
first = {p: [first_correct(seq(r, p)) for r in R] for p in POLICIES}
solved = lambda p, b: [f is not None and f <= b for f in first[p]]

print(f"{len(R)} problems (first trace was wrong). mean fresh-sample length "
      f"{np.mean([s['length'] for r in R for s in r['samples']]):.0f}, first-sample accuracy "
      f"{np.mean([r['samples'][0]['correct'] for r in R]):.1%}\n")
print(f"{'policy':22s} " + " ".join(f"{'@' + str(b):>7s}" for b in BUDGETS) + f" {'mean tok (cap ' + str(CAP) + ')':>20s}")
for p in POLICIES:
    capped = np.mean([min(f, CAP) if f is not None else CAP for f in first[p]])
    print(f"{p:22s} " + " ".join(f"{np.mean(solved(p, b)):7.1%}" for b in BUDGETS) + f" {capped:20.0f}")

print("\npaired vs resample (wins-losses, McNemar p) at each budget:")
for p in POLICIES[1:]:
    cols = []
    for b in BUDGETS:
        x, y = solved(p, b), solved("resample", b)
        w = sum(a and not c for a, c in zip(x, y)); l = sum(c and not a for a, c in zip(x, y))
        cols.append(f"@{b} +{w}-{l} p={mcnemar(w, l):.2f}")
    print(f"  {p:22s} " + "  ".join(cols))

print("\nfork branches: share correct (per wrong sample with a fork)")
for name in ("gated", "local", "random"):
    br = [b for r in R for s in r["samples"] for b in s["forks"].get(name, {}).get("branches", [])]
    any_ok = [any(b["correct"] for b in s["forks"][name]["branches"]) for r in R for s in r["samples"] if name in s["forks"]]
    print(f"  {name:7s} branches {len(br):5d}  correct {np.mean([b['correct'] for b in br]):6.1%}"
          f"  fork has any correct {np.mean(any_ok):6.1%}")
print(f"  (a fresh sample is correct {np.mean([s['correct'] for r in R for s in r['samples']]):.1%} of the time)")

# average tokens to the first correct answer, on problems every policy eventually solves
both = [i for i in range(len(R)) if all(first[p][i] is not None for p in POLICIES)]
rng = np.random.default_rng(0)
print(f"\nmean tokens to first correct, on the {len(both)} problems every policy solves (paired vs resample, 95% bootstrap CI)")
for p in POLICIES:
    x = np.array([first[p][i] for i in both], float)
    d = x - np.array([first["resample"][i] for i in both], float)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]
    diff = "" if p == "resample" else f"{d.mean():+7.0f} [{np.percentile(boot, 2.5):+.0f}, {np.percentile(boot, 97.5):+.0f}]"
    print(f"  {p:24s} mean {x.mean():6.0f}  median {np.median(x):6.0f}  {diff}")

# forced-token probabilities (runs that saved them: race.py from the probability-saving version on)
if any("prob" in b for r in R for s in r["samples"] for f in s["forks"].values() for b in f["branches"]):
    edges = [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.01]
    names = ["<1e-10", "1e-10..1e-8", "1e-8..1e-6", "1e-6..1e-4", "1e-4..1e-2", ">=1e-2"]
    print("\nforced tokens by the model's probability for them (share of each fork type's forced tokens)")
    print(f"  {'fork':7s} " + " ".join(f"{n:>12s}" for n in names) + f" {'<0.001%':>9s}")
    for name in ("random", "local", "gated"):
        ps = np.array([b["prob"] for r in R for s in r["samples"] for b in s["forks"].get(name, {}).get("branches", [])])
        h = np.histogram(ps, bins=edges)[0] / len(ps)
        print(f"  {name:7s} " + " ".join(f"{v:12.0%}" for v in h) + f" {np.mean(ps < 1e-5):9.1%}")
    br = [b for r in R for s in r["samples"] for f in s["forks"].values() for b in f["branches"]]
    for lo, hi, label in ((0.1, 2, "above 10%"), (0, 1e-5, "below 0.001%")):
        sel = [b["correct"] for b in br if lo <= b["prob"] < hi]
        print(f"  branches forced at a token {label}: {len(sel)}, correct {np.mean(sel):.1%}")
