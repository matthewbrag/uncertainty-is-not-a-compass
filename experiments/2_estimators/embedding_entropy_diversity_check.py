"""
Same as embedding_entropy_check.py but validated against the corrected
diversity_entropy ground truth (Shannon entropy over distinct answer values
among the 5 forced branches) instead of the old binary outcome_entropy.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python experiments/2_estimators/embedding_entropy_diversity_check.py [n_samples] [dataset_path] [model_key]
"""
import json
import math
import random
import sys
from collections import Counter

import torch

from forkprobe.hf import get_model, DEVICE

TOP_K = 40
N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
DATASET_PATH = sys.argv[2] if len(sys.argv) > 2 else "data/overnight_merged.jsonl"
MODEL_KEY = sys.argv[3] if len(sys.argv) > 3 else "qwen3-0.6b"

tokenizer, model = get_model(MODEL_KEY)
embed_matrix = model.get_input_embeddings().weight.detach()  # [vocab, hidden]
embed_norm = torch.nn.functional.normalize(embed_matrix, dim=-1)


def diversity_entropy(answers):
    vals = [a if a is not None else "unparsed" for a in answers]
    counts = Counter(vals)
    n = len(vals)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h


records = []
with open(DATASET_PATH) as f:
    for line in f:
        records.append(json.loads(line))
print(f"{DATASET_PATH}: {len(records)} total points available", flush=True)

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


h_raws, h_sems, diversities = [], [], []
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
        diversities.append(diversity_entropy(r["answers"]))
        if (n + 1) % 200 == 0:
            print(f"{n+1}/{len(sample)} done", flush=True)

import numpy as np
h_raws = np.array(h_raws)
h_sems = np.array(h_sems)
diversities = np.array(diversities)

print(f"\nn={len(diversities)}")
print(f"corr(raw entropy, diversity_entropy):       r={np.corrcoef(h_raws, diversities)[0,1]:+.4f}")
print(f"corr(embedding entropy, diversity_entropy): r={np.corrcoef(h_sems, diversities)[0,1]:+.4f}")
print(f"corr(raw entropy, embedding entropy):       r={np.corrcoef(h_raws, h_sems)[0,1]:+.4f}  (how much do they actually differ)")
