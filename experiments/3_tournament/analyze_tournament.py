"""
Analysis for advanced_tournament_vllm.py output: rescue rate per arm (raw and
ignoring answer-line cuts), tokens per problem and per rescue, paired exact
McNemar tests against the two controls (every arm runs on the same traces, so
pairing is valid), majority vote over each arm's completions, and what kind of
cut produced each rescue.

--integer-only keeps problems with integer reference answers: the article's
table. Decimal answers were exposed to a stop-rule bug ("#### 4.5" cut to "4.")
in these runs; integer answers can't be.

Usage: python experiments/3_tournament/analyze_tournament.py [--integer-only] data/advanced_vllm_w0.jsonl data/advanced_vllm_w1.jsonl ...
"""
import json
import math
import sys
from collections import Counter

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
INTEGER_ONLY = "--integer-only" in sys.argv
R = [json.loads(l) for path in sys.argv[1:] if not path.startswith("--") for l in open(path)]
if INTEGER_ONLY:
    R = [r for r in R if float(r["reference_answer"]).is_integer()]
n = len(R)
arms = list(R[0]["results"])
print(f"{n} scored problems (first sampled trace wrong){' with integer answers' if INTEGER_ONLY else ''}\n")


def mcnemar(a, b):
    """Exact two-sided McNemar p-value from discordant pair counts."""
    m = a + b
    if m == 0:
        return 1.0
    k = min(a, b)
    p = sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m
    return min(1.0, 2 * p)


def majority_correct(r, arm):
    """Is the most common parsed answer across the arm's completions correct?"""
    if arm == "sampled_restarts":
        answers = [x["answer"] for x in r["restarts"]]
    else:
        table = r["resampled"] if arm == "resample_uniform_cuts" else r["forced"]
        answers = [a for p in r["results"][arm]["points"] for a in table[str(p)]["answers"]]
    c = Counter(round(a, 4) for a in answers if a is not None)
    return bool(c) and abs(c.most_common(1)[0][0] - r["reference_answer"]) < 1e-4


def rescued(r, arm, strict=False):
    res = r["results"][arm]
    return res.get("rescued_excl_answer_line") if strict and "rescued_excl_answer_line" in res else res["rescued"]


print(f"{'arm':28s} {'rescued':>8s} {'excl ans':>9s} {'majority':>9s} {'tok/prob':>9s} {'tok/rescue':>11s}   vs uniform        vs restart+interior")
for arm in arms:
    rr = sum(rescued(r, arm) for r in R) / n
    rs = sum(bool(rescued(r, arm, True)) for r in R) / n
    toks = sum(r["results"][arm]["tokens"] for r in R) / n
    maj = sum(majority_correct(r, arm) for r in R) / n
    cols = []
    for ctrl in ("uniform", "restart_uniform_interior"):
        if arm == ctrl:
            cols.append(f"{'--':16s}")
            continue
        win = sum(rescued(r, arm) and not rescued(r, ctrl) for r in R)
        loss = sum(rescued(r, ctrl) and not rescued(r, arm) for r in R)
        cols.append(f"+{win:<3d}-{loss:<3d} p={mcnemar(win, loss):.3f}")
    print(f"{arm:28s} {rr:8.1%} {rs:9.1%} {maj:9.1%} {toks:9.0f} {toks / max(rr, 1e-9):11.0f}   {cols[0]}  {cols[1]}")

if "sampled_restarts" in arms:
    print(f"\nsampled_restarts using only the first 5 samples: "
          f"{sum(r['results']['sampled_restarts']['rescued_first5'] for r in R) / n:.1%}")


def cut_kind(r, cut):
    gen = r["generated_token_ids"]
    if cut == 0:
        return "restart (token 0)"
    if cut >= r["answer_start"]:
        return "answer line"
    t = tok.decode([gen[cut]])
    if any(c.isdigit() for c in t):
        return "digit"
    if not any(c.isalnum() for c in t):
        return "formatting"
    rel = cut / r["trace_len"]
    return "word, early" if rel < 0.33 else "word, mid" if rel < 0.67 else "word, late"


print("\nWhat kind of cut rescued? (share of rescuing cuts per arm; a problem can be rescued by several)")
kinds = ["restart (token 0)", "word, early", "word, mid", "word, late", "digit", "formatting", "answer line"]
print(f"{'arm':28s} " + " ".join(f"{k[:12]:>12s}" for k in kinds))
for arm in arms:
    if arm == "sampled_restarts":
        continue
    table = "resampled" if arm == "resample_uniform_cuts" else "forced"
    c = Counter()
    for r in R:
        for p in r["results"][arm]["points"]:
            if any(r[table][str(p)]["correct"]):
                c[cut_kind(r, p)] += 1
    tot = sum(c.values()) or 1
    print(f"{arm:28s} " + " ".join(f"{c[k] / tot:12.0%}" for k in kinds) + f"   (n={tot})")

print("\nRescues NOT available from the token-0 restart (problems the restart cut alone misses):")
no_restart = [r for r in R if "0" in r["forced"] and not any(r["forced"]["0"]["correct"])]
print(f"  {len(no_restart)} problems")
for arm in arms:
    if arm in ("sampled_restarts",):
        continue
    print(f"  {arm:28s} {sum(rescued(r, arm) for r in no_restart) / max(1, len(no_restart)):.1%}")
