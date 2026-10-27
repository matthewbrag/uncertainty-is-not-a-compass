"""
Analysis for gate_addon.py: gated arms vs the tournament's own arms on the
same saved traces (integer-answer problems), paired. Reports rescue (any
branch correct), majority vote over the arm's branches, tokens, and
tokens-to-first-correct at a few budgets (cuts visited in the arm's score
order: restart first, then by gated score).

Usage: python analyze_gate.py gate_w0.jsonl gate_w1.jsonl gate_w2.jsonl
"""
import glob
import json
import math
import sys
from collections import Counter

import numpy as np

T = {}
for p in sorted(glob.glob("advanced_vllm_w*.jsonl")):
    for l in open(p):
        r = json.loads(l)
        if float(r["reference_answer"]).is_integer():
            T[r["math_test_index"]] = r
G = [json.loads(l) for p in sys.argv[1:] for l in open(p)]
n = len(G)


def mcnemar(a, b):
    m = a + b
    return 1.0 if m == 0 else min(1.0, 2 * sum(math.comb(m, i) for i in range(min(a, b) + 1)) / 2 ** m)


def forced_table(g):
    return {**T[g["math_test_index"]]["forced"], **g["forced_new"]}


def arm(g, name):
    """(points, rescued, answers, ordered (length, correct) sequence) for any arm."""
    t = T[g["math_test_index"]]
    fz = forced_table(g)
    if name in g["results"]:
        pts = g["results"][name]["points"]
        gate = g["gate"]
        z, le = np.array(t["probe_z"]), np.array(t["local_entropy"])
        prod = np.maximum(z, 0) * le
        key = {"restart_product_x_nextdiv": lambda c: prod[c] * gate[str(c)][0],
               "restart_product_x_lookdiv": lambda c: prod[c] * gate[str(c)][1],
               "restart_local_x_lookdiv": lambda c: le[c] * gate[str(c)][1]}[name]
        order = [0] + sorted([p for p in pts if p], key=lambda c: -key(c))
    else:
        pts = t["results"][name]["points"]
        z, le = np.array(t["probe_z"]), np.array(t["local_entropy"])
        sc = {"restart_product": np.maximum(z, 0) * le, "restart_local_only": le}.get(name)
        order = [0] + (sorted([p for p in pts if p], key=lambda c: -sc[c]) if sc is not None else [p for p in pts if p])
    answers = [a for p in pts for a in fz[str(p)]["answers"]]
    seq = [(L, ok) for p in order for L, ok in zip(fz[str(p)]["lengths"], fz[str(p)]["correct"])]
    return pts, any(any(fz[str(p)]["correct"]) for p in pts), answers, seq


def maj_ok(answers, ref):
    c = Counter(round(a, 4) for a in answers if a is not None)
    return bool(c) and abs(c.most_common(1)[0][0] - ref) < 1e-4


ARMS = ["restart_product_x_nextdiv", "restart_product_x_lookdiv", "restart_local_x_lookdiv",
        "restart_product", "restart_local_only", "restart_uniform_interior"]
rows = {}
for a in ARMS:
    res = []
    for g in G:
        pts, ok, ans, seq = arm(g, a)
        ref = T[g["math_test_index"]]["reference_answer"]
        t, first = 0, None
        for L, c in seq:
            t += L
            if c and first is None:
                first = t
        res.append((ok, maj_ok(ans, ref), sum(L for L, _ in seq), first))
    rows[a] = res

print(f"{n} integer-answer problems, same saved traces for every arm\n")
B = [2000, 4000, 8000]
print(f"{'arm':28s} {'rescued':>8s} {'majority':>9s} {'tokens':>7s} " + " ".join(f"{'@'+str(b//1000)+'k':>6s}" for b in B))
for a, res in rows.items():
    print(f"{a:28s} {np.mean([r[0] for r in res]):8.1%} {np.mean([r[1] for r in res]):9.1%} {np.mean([r[2] for r in res]):7.0f} "
          + " ".join(f"{np.mean([r[3] is not None and r[3] <= b for r in res]):6.1%}" for b in B))

print("\npaired (rescued / majority / solved within 4k tokens):")
for a in ARMS[:3]:
    for b in ("restart_product", "restart_local_only", "restart_uniform_interior"):
        cols = []
        for k, f in (("rescue", lambda r: r[0]), ("maj", lambda r: r[1]), ("@4k", lambda r: r[3] is not None and r[3] <= 4000)):
            w = sum(f(x) and not f(y) for x, y in zip(rows[a], rows[b]))
            l = sum(f(y) and not f(x) for x, y in zip(rows[a], rows[b]))
            cols.append(f"{k} +{w}-{l} p={mcnemar(w, l):.2f}")
        print(f"  {a:26s} vs {b:24s} " + "  ".join(cols))
