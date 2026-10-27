"""
Analysis for the noise ablation ("would a better probe help?") in
shared_trace_monotonic.py output: rescue rate as the probe's score is mixed with
noise, from 0% (pure probe) to 100% (pure noise), on problems with integer
reference answers. If a better estimate of the probe's target helped, rescue
should fall as noise rises.

Reports each noise level's rescue rate, paired McNemar tests against the pure
probe, the fitted linear trend across noise levels with a bootstrap CI, and how tightly
each level's cut points clump (the pure probe's picks bunch together; noise
spreads them out).

Usage: python experiments/3_tournament/analyze_noise_ablation.py data/shared_trace_fixed_level4_offset0.jsonl ...
"""
import json
import math
import random
import statistics
import sys

R = [json.loads(l) for p in sys.argv[1:] for l in open(p)]
R = [r for r in R if float(r["reference_answer"]).is_integer()]
LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
key = lambda a: f"noise_alpha{a}"
resc = {a: [bool(r["strategy_results"][key(a)]["rescued"]) for r in R] for a in LEVELS}


def mcnemar(a, b):
    m = a + b
    return 1.0 if m == 0 else min(1.0, 2 * sum(math.comb(m, i) for i in range(min(a, b) + 1)) / 2 ** m)


print(f"{len(R)} problems with integer reference answers (first sampled trace wrong)\n")
print(f"{'noise':>6s} {'rescued':>8s}   vs pure probe            median gap between cuts   cuts within 5 tokens")
for a in LEVELS:
    w = sum(x and not y for x, y in zip(resc[a], resc[0.0]))
    l = sum(y and not x for x, y in zip(resc[a], resc[0.0]))
    gaps = []
    for r in R:
        p = sorted(r["strategy_results"][key(a)]["points"])
        gaps += [y - x for x, y in zip(p, p[1:])]
    vs = "--" if a == 0.0 else f"+{w}/-{l}  p={mcnemar(w, l):.2f}"
    print(f"{int(a * 100):5d}% {sum(resc[a]) / len(R):8.1%}   {vs:24s} {statistics.median(gaps):10.0f} tokens"
          f"{sum(g <= 5 for g in gaps) / len(gaps):20.0%}")


def trend(idx):
    """Least-squares slope of rescue rate on noise level, in points per 0->100% noise."""
    ys = [100 * sum(resc[a][i] for i in idx) / len(idx) for a in LEVELS]
    mx, my = sum(LEVELS) / len(LEVELS), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(LEVELS, ys)) / sum((x - mx) ** 2 for x in LEVELS)


rng = random.Random(0)
boot = sorted(trend([rng.randrange(len(R)) for _ in R]) for _ in range(10000))
print(f"\nlinear trend from pure probe to pure noise: {trend(range(len(R))):+.1f} points, "
      f"95% bootstrap CI [{boot[249]:+.1f}, {boot[9749]:+.1f}] (resampling problems)")
