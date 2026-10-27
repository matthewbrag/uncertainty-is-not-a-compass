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

Usage: python race_analyze.py race_w0.jsonl race_w1.jsonl race_w2.jsonl
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
