"""
Analysis for fork_vote_tournament.py: for each fork arm, oracle ("any branch
correct") and majority-vote accuracy, versus best-of-N sampled restarts given
the SAME token budget on the same problem, with paired exact McNemar tests.

Usage: python analyze_fork_vote.py fork_vote_w0.jsonl fork_vote_w1.jsonl ...
"""
import json
import math
import sys
from collections import Counter

R = [json.loads(l) for p in sys.argv[1:] for l in open(p)]
n = len(R)


def correct(a, ref):
    return a is not None and abs(a - ref) < 1e-4


def majority(answers):
    c = Counter(round(a, 4) for a in answers if a is not None)
    return c.most_common(1)[0][0] if c else None


def mcnemar(a, b):
    m = a + b
    if m == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(min(a, b) + 1)) / 2 ** m)


def matched_restarts(r, budget):
    used, picked = 0, []
    for s in r["restarts"]:
        if used + s["length"] > budget:
            break
        used += s["length"]
        picked.append(s)
    return picked


print(f"{n} scored problems (trace wrong)\n")
print(f"{'arm':16s} {'tok':>6s} {'any':>6s} {'maj':>6s} | matched best-of-N: {'N':>5s} {'any':>6s} {'maj':>6s} | "
      f"maj arm vs restarts (paired)")
for arm in R[0]["arms"]:
    any_a = maj_a = any_r = maj_r = win = loss = 0
    toks = ns = 0
    for r in R:
        ref = r["reference_answer"]
        br = r["arms"][arm]
        budget = sum(b["length"] for b in br)
        rs = matched_restarts(r, budget)
        toks += budget
        ns += len(rs)
        a_any = any(b["correct"] for b in br)
        a_maj = correct(majority([b["answer"] for b in br]), ref)
        r_any = any(s["correct"] for s in rs)
        r_maj = correct(majority([s["answer"] for s in rs]), ref)
        any_a += a_any; maj_a += a_maj; any_r += r_any; maj_r += r_maj
        win += a_maj and not r_maj
        loss += r_maj and not a_maj
    print(f"{arm:16s} {toks / n:6.0f} {any_a / n:6.1%} {maj_a / n:6.1%} |                    {ns / n:5.1f} "
          f"{any_r / n:6.1%} {maj_r / n:6.1%} | +{win} -{loss} p={mcnemar(win, loss):.3f}")

print("\nall 32 restarts: any {:.1%}  maj {:.1%}".format(
    sum(any(s["correct"] for s in r["restarts"]) for r in R) / n,
    sum(correct(majority([s["answer"] for s in r["restarts"]]), r["reference_answer"]) for r in R) / n))
print("\nshare of branches that land back on the trace's own wrong answer:")
for arm in R[0]["arms"]:
    tot = sum(len(r["arms"][arm]) for r in R)
    back = sum(sum(b["answer"] is not None and r["trace_answer"] is not None and abs(b["answer"] - r["trace_answer"]) < 1e-4
                   for b in r["arms"][arm]) for r in R)
    print(f"  {arm:16s} {back / tot:.1%}")
tot = sum(len(r["restarts"]) for r in R)
back = sum(sum(s["answer"] is not None and r["trace_answer"] is not None and abs(s["answer"] - r["trace_answer"]) < 1e-4
               for s in r["restarts"]) for r in R)
print(f"  {'restarts':16s} {back / tot:.1%}")
