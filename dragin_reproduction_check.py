"""
Faithful reproduction of DRAGIN's RIND score (Su et al., 2024, arXiv 2403.10081, Eq. 5):

    RIND(t_i) = H_i * a_max(i) * s_i

    H_i     = entropy of the model's predicted token distribution at position i   (Eq. 1)
    a_max(i)= max_{j>i} A_{j,i}, the largest attention weight any later position j
              pays back to position i, last transformer layer, averaged over heads (Eq. 3)
    s_i     = 0 if the token at position i is a stopword, else 1                   (Eq. 4)

Unlike the earlier v2_importance pilot (n=30, pure attention weight, no entropy term,
no stopword gate), this is the actual published formula, run at real sample size.

One forward pass per unique problem (prompt + full greedy trace, eager attention so
attentions are exposed), reused across every saved candidate cut position for that
problem - same traces already collected in overnight_merged.jsonl, no new generation.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python dragin_reproduction_check.py [n_points]
"""
import json
import math
import random
import sys
from collections import Counter, defaultdict

import torch

from app import get_model, DEVICE

MODEL_KEY = "qwen3-0.6b"
N_POINTS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

STOPWORDS = set("""
a an the and or but if then else when while of to in on at by for with about
against between into through during before after above below from up down out
off over under again further once here there all any both each few more most
other some such no nor not only own same so than too very s t can will just don
should now is are was were be been being have has had do does did doing i you
he she it we they them his her its our their this that these those am as until
""".split())


def diversity_entropy(answers):
    vals = [a if a is not None else "unparsed" for a in answers]
    counts = Counter(vals)
    n = len(vals)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h


def is_stopword(token_text):
    stripped = token_text.strip().lower()
    if stripped == "" or not any(ch.isalnum() for ch in stripped):
        return True  # whitespace / punctuation / BPE fragment -> treat as non-content
    return stripped in STOPWORDS


tokenizer, model = get_model(MODEL_KEY)
model.set_attn_implementation("eager")

records = []
with open("overnight_merged.jsonl") as f:
    for line in f:
        records.append(json.loads(line))
print(f"{len(records)} total points available", flush=True)

by_problem = defaultdict(list)
for r in records:
    by_problem[r["gsm8k_train_index"]].append(r)

random.seed(0)
problem_ids = list(by_problem.keys())
random.shuffle(problem_ids)

rind_scores, h_scores, outcomes, diversities = [], [], [], []
n_points_done = 0
n_problems_done = 0

with torch.no_grad():
    for pid in problem_ids:
        if n_points_done >= N_POINTS:
            break
        pts = by_problem[pid]
        base_rec = pts[0]
        full_ids = base_rec["prompt_token_ids"] + base_rec["generated_token_ids"]
        base = len(base_rec["prompt_token_ids"])
        n_gen = len(base_rec["generated_token_ids"])

        input_ids = torch.tensor([full_ids], device=DEVICE)
        out = model(input_ids=input_ids, output_attentions=True)
        logits = out.logits[0]  # [seq, vocab]
        last_attn = out.attentions[-1][0].mean(dim=0)  # [seq, seq], avg over heads

        # entropy trace: for decode step i (0-indexed into generated tokens),
        # distribution predicted at sequence position (base + i - 1)
        probs_all = torch.softmax(logits, dim=-1)

        for r in pts:
            cut = r["cut_token"]
            seq_pos = base + cut - 1
            if seq_pos < 0 or seq_pos >= len(full_ids) - 1:
                continue
            p = probs_all[seq_pos]
            h = -(p * torch.log2(p.clamp(min=1e-12))).sum().item()

            future_positions = range(seq_pos + 1, len(full_ids))
            if not future_positions:
                continue
            a_max = max(last_attn[j, seq_pos].item() for j in future_positions)

            token_text = tokenizer.decode([full_ids[seq_pos]])
            s = 0.0 if is_stopword(token_text) else 1.0

            rind = h * a_max * s
            rind_scores.append(rind)
            h_scores.append(h)
            outcomes.append(r["outcome_entropy"])
            diversities.append(diversity_entropy(r["answers"]))
            n_points_done += 1
            if n_points_done >= N_POINTS:
                break

        n_problems_done += 1
        if n_problems_done % 50 == 0:
            print(f"{n_problems_done} problems, {n_points_done}/{N_POINTS} points done", flush=True)

import numpy as np
rind_scores = np.array(rind_scores)
outcomes = np.array(outcomes)
diversities = np.array(diversities)

print(f"\nn={len(outcomes)} points from {n_problems_done} problems")
print(f"RIND vs outcome_entropy (old, binary):     r={np.corrcoef(rind_scores, outcomes)[0,1]:+.4f}")
print(f"RIND vs diversity_entropy (corrected):      r={np.corrcoef(rind_scores, diversities)[0,1]:+.4f}")
print(f"fraction of points gated to zero by s_i=0:  {(rind_scores == 0).mean():.1%}")
