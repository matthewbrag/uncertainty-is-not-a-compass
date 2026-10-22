"""
Quantifies the embedding-weighted entropy heuristic the same way every other
candidate metric in this project gets judged: correlate it against the real
ground-truth outcome_entropy from forced branching. Reuses already-collected
GSM8K candidate points (prefix + cut position + outcome_entropy already
recorded) - no new generation needed, just a forward pass per point to get
the real softmax distribution and the model's own embedding matrix.

H(t)     = -sum_i p_i log2(p_i)                        [raw entropy]
H_sem(t) = -sum_i p_i log2(sum_j S_ij p_j)              [embedding-weighted]
where S_ij = cosine similarity between token embeddings i, j.

Truncates to top-K tokens by probability before building the K x K
similarity matrix (full-vocab pairwise cosine sim is intractable).

Usage: CUDA_VISIBLE_DEVICES=<gpu> python embedding_entropy_check.py [n_samples]
"""
import json
import math
import random
import sys

import torch

from app import get_model, DEVICE

MODEL_KEY = "qwen3-0.6b"
TOP_K = 40
N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

tokenizer, model = get_model(MODEL_KEY)
embed_matrix = model.get_input_embeddings().weight.detach()  # [vocab, hidden]
embed_norm = torch.nn.functional.normalize(embed_matrix, dim=-1)

records = []
with open("overnight_merged.jsonl") as f:
    for line in f:
        records.append(json.loads(line))
print(f"{len(records)} total points available", flush=True)

random.seed(0)
sample = random.sample(records, min(N_SAMPLES, len(records)))


def raw_and_semantic_entropy(probs_topk, idx_topk):
    p = probs_topk / probs_topk.sum()  # renormalize over top-K
    h_raw = -(p * torch.log2(p)).sum().item()

    e = embed_norm[idx_topk]  # [K, hidden]
    S = e @ e.T  # [K, K] cosine similarity (unit vectors)
    weighted = S @ p  # sum_j S_ij p_j, for each i
    weighted = weighted.clamp(min=1e-8)
    h_sem = -(p * torch.log2(weighted)).sum().item()
    return h_raw, h_sem


h_raws, h_sems, outcomes = [], [], []
with torch.no_grad():
    for n, r in enumerate(sample):
        cut = r["cut_token"]
        base = len(r["prompt_token_ids"])
        pos_in_seq = base + cut - 1  # same convention used throughout this project
        full_ids = r["prompt_token_ids"] + r["generated_token_ids"][:cut + 1]
        if pos_in_seq >= len(full_ids) or pos_in_seq < 0:
            continue
        input_ids = torch.tensor([full_ids[:pos_in_seq + 1]], device=DEVICE)
        out = model(input_ids=input_ids)
        logits = out.logits[0, -1]
        probs = torch.softmax(logits, dim=-1)
        topv, topi = torch.topk(probs, TOP_K)
        h_raw, h_sem = raw_and_semantic_entropy(topv, topi)
        h_raws.append(h_raw)
        h_sems.append(h_sem)
        outcomes.append(r["outcome_entropy"])
        if (n + 1) % 200 == 0:
            print(f"{n+1}/{len(sample)} done", flush=True)

import numpy as np
h_raws = np.array(h_raws)
h_sems = np.array(h_sems)
outcomes = np.array(outcomes)

print(f"\nn={len(outcomes)}")
print(f"corr(raw entropy, outcome_entropy):       r={np.corrcoef(h_raws, outcomes)[0,1]:+.4f}")
print(f"corr(embedding entropy, outcome_entropy): r={np.corrcoef(h_sems, outcomes)[0,1]:+.4f}")
print(f"corr(raw entropy, embedding entropy):     r={np.corrcoef(h_raws, h_sems)[0,1]:+.4f}  (how much do they actually differ)")
