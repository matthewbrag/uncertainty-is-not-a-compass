"""
Is greedy decoding hurting us? Solves the same MATH test problems with greedy
decoding and with the sampled settings used for traces (T=0.7, top_k=20,
top_p=0.8), same budgets and stop rules, and compares accuracy -- branch
continuations in every tournament are greedy, traces are sampled.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python greedy_vs_sampled.py <level> <n> <offset> <out_json>
"""
import json
import random
import re
import sys

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import LogitsProcessorList

from app import get_model, DEFAULT_TEMPLATE, extract_answer, ForceEosAfterAnswer, ForceEosOnRepetition, DEVICE

LEVEL, N, OFFSET, OUT = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
BATCH, MAX_NEW = 32, 1500
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)


def parse_numeric(a):
    a = a.strip().rstrip(".")
    try:
        return float(a)
    except ValueError:
        pass
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", a)
    return (-1.0 if m.group(1) else 1.0) * float(m.group(2)) / float(m.group(3)) if m else None


def extract_boxed(sol):
    idx = sol.rfind("\\boxed{")
    if idx == -1:
        return None
    start, depth, i = idx + 7, 1, idx + 7
    while i < len(sol) and depth > 0:
        depth += {"{": 1, "}": -1}.get(sol[i], 0)
        i += 1
    return sol[start:i - 1] if depth == 0 else None


subjects = ["algebra", "counting_and_probability", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
probs = []
for r in concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", s, split="test") for s in subjects]):
    if r["level"] == f"Level {LEVEL}":
        b = extract_boxed(r["solution"])
        ref = parse_numeric(b) if b is not None else None
        if ref is not None:
            probs.append((r["problem"], ref))
random.Random(7000).shuffle(probs)
probs = probs[OFFSET:OFFSET + N]

tok, model = get_model("qwen3-1.7b")
tok.padding_side = "left"


def run(batch, sample):
    ids = [tok.encode(tok.apply_chat_template([{"role": "user", "content": DEFAULT_TEMPLATE.replace("{question}", q)}],
                                              tokenize=False, add_generation_prompt=True, enable_thinking=False),
                      add_special_tokens=False) for q, _ in batch]
    L = max(map(len, ids))
    inp = torch.tensor([[tok.pad_token_id] * (L - len(x)) + x for x in ids], device=DEVICE)
    att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids], device=DEVICE)
    procs = LogitsProcessorList([ForceEosAfterAnswer(tok, L, tok.eos_token_id, False), ForceEosOnRepetition(L, tok.eos_token_id)])
    kw = dict(do_sample=True, **SAMPLE_KW) if sample else dict(do_sample=False)
    with torch.no_grad():
        out = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=MAX_NEW, eos_token_id=tok.eos_token_id,
                             pad_token_id=tok.pad_token_id, logits_processor=procs, **kw)
    res = []
    for j, (_, ref) in enumerate(batch):
        row = out[j, L:].tolist()
        capped = tok.eos_token_id not in row
        a = extract_answer(tok.decode(row, skip_special_tokens=True))
        res.append({"correct": a is not None and abs(a - ref) < 1e-4, "capped": capped})
    return res


results = []
for s in range(0, len(probs), BATCH):
    b = probs[s:s + BATCH]
    g, sm = run(b, False), run(b, True)
    results += [{"greedy": x, "sampled": y} for x, y in zip(g, sm)]
    print(f"{len(results)}/{len(probs)}", flush=True)
json.dump(results, open(OUT, "w"))
