"""
Tokens-to-first-correct, every strategy vs fresh resampling, on the same
integer-answer problems (the gate add-on's 250). Each strategy explores in its
own order and stops at the first correct answer (perfect verifier assumed).
Fresh samples (the 25 restarts) are averaged over 200 random orderings.
Gate scoring cost (lookahead decoding) is NOT counted here; see the notes.

Usage: python experiments/3_tournament/ttfc_compare.py gate_w0.jsonl gate_w1.jsonl gate_w2.jsonl
"""
import json
import random
import sys

import numpy as np

sys_argv, sys.argv = sys.argv, sys.argv[:1] + sys.argv[1:]
import analyze_gate as ag  # reuses arm() orderings; prints its own table on import

G = ag.G
BUDGETS = [500, 1000, 2000, 4000, 8000, 16000]


def first_correct(seq):
    t = 0
    for L, ok in seq:
        t += L
        if ok:
            return t
    return None


curves, spent = {}, {}
for a in ag.ARMS:
    firsts, tot = [], []
    for g in G:
        _, _, _, seq = ag.arm(g, a)
        firsts.append(first_correct(seq)); tot.append(sum(L for L, _ in seq))
    curves[a], spent[a] = firsts, tot

rng = random.Random(0)
fr_curve, fr_spent = [], []
for g in G:
    rs = ag.T[g["math_test_index"]]["restarts"]
    per = []
    for _ in range(200):
        o = rs[:]; rng.shuffle(o)
        per.append(first_correct([(x["length"], x["correct"]) for x in o]))
    fr_curve.append(per); fr_spent.append(sum(x["length"] for x in rs))


def solved_by(firsts, b):
    return np.mean([f is not None and f <= b for f in firsts])


print(f"\n\n{len(G)} problems. Share solved within budget (tokens after the wrong trace):")
print(f"{'strategy':28s} " + " ".join(f"{'@'+str(b):>7s}" for b in BUDGETS) + f" {'max solved':>11s} {'max tokens':>11s}")
row = lambda name, vals, mx, tok: print(f"{name:28s} " + " ".join(f"{v:7.1%}" for v in vals) + f" {mx:11.1%} {tok:11.0f}")
row("fresh samples (25)", [np.mean([solved_by(p, b) for p in fr_curve]) for b in BUDGETS],
    np.mean([any(x["correct"] for x in ag.T[g["math_test_index"]]["restarts"]) for g in G]), np.mean(fr_spent))
for a in ag.ARMS:
    row(a, [solved_by(curves[a], b) for b in BUDGETS], np.mean([f is not None for f in curves[a]]), np.mean(spent[a]))

# among problems both a strategy and fresh sampling eventually solve: median tokens to first correct
print("\nproblems solved by BOTH: median tokens to first correct (fresh = mean over orderings)")
for a in ag.ARMS:
    both = [(f, np.mean([x for x in p if x is not None])) for f, p in zip(curves[a], fr_curve)
            if f is not None and any(x is not None for x in p)]
    wins = sum(f < r for f, r in both)
    print(f"  {a:28s} n={len(both):3d}  strategy {np.median([f for f, _ in both]):6.0f}  fresh {np.median([r for _, r in both]):6.0f}"
          f"   strategy faster on {wins}/{len(both)}")
