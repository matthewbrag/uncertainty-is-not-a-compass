"""
Debug dump for the learned branching strategies: for a few greedy-wrong
MATH test problems, saves the full trace, the probe score at every token,
each strategy's picks, and every forced branch's text + answer -- the things
the tournament computes but doesn't keep. Same logic as
shared_trace_monotonic.py (fixed indexing), plus a stratified_probe preview.

Usage: python debug_strategy_trace.py <level> <probe_path> <n_wrong_problems> <out_json> [seed]
"""
import json
import random
import re
import sys

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import LogitsProcessorList

from app import (
    get_model, DEFAULT_TEMPLATE, extract_answer,
    ForceEosAfterAnswer, ForceEosOnRepetition, DEVICE,
)

LEVEL = int(sys.argv[1])
PROBE_PATH = sys.argv[2]
N_WRONG = int(sys.argv[3])
OUT = sys.argv[4]
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 4242

K_POINTS, N_BRANCHES = 5, 5
GEN_MAX, CONT_MAX = 1500, 1300
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)


def parse_numeric(a):
    a = a.strip().rstrip(".")
    try:
        return float(a)
    except ValueError:
        pass
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", a)
    if m:
        return (-1.0 if m.group(1) else 1.0) * float(m.group(2)) / float(m.group(3))
    return None


def extract_boxed(sol):
    idx = sol.rfind("\\boxed{")
    if idx == -1:
        return None
    start, depth, i = idx + len("\\boxed{"), 1, idx + len("\\boxed{")
    while i < len(sol) and depth > 0:
        depth += {"{": 1, "}": -1}.get(sol[i], 0)
        i += 1
    return sol[start:i - 1] if depth == 0 else None


subjects = ["algebra", "counting_and_probability", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]
test = concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", s, split="test") for s in subjects])
problems = []
for r in test:
    if r["level"] != f"Level {LEVEL}":
        continue
    b = extract_boxed(r["solution"])
    ref = parse_numeric(b) if b is not None else None
    if ref is not None:
        problems.append({"problem": r["problem"], "ref": ref, "solution": r["solution"]})

probe = json.load(open(PROBE_PATH))
LAYER = probe["layer"]
tok, model = get_model("qwen3-1.7b")
tok.padding_side = "left"
p_mean = torch.tensor(probe["scaler_mean"], device=DEVICE)
p_scale = torch.tensor(probe["scaler_scale"], device=DEVICE)
p_coef = torch.tensor(probe["coef"], device=DEVICE)


def prompt_ids_for(q):
    text = tok.apply_chat_template([{"role": "user", "content": DEFAULT_TEMPLATE.replace("{question}", q)}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok.encode(text, add_special_tokens=False)


def generate(id_lists, max_new, sample):
    """Batched generate for equal-length inputs (no padding needed): one
    trace, or the N_BRANCHES forced branches at one cut."""
    inp = torch.tensor(id_lists, device=DEVICE)
    procs = LogitsProcessorList([ForceEosAfterAnswer(tok, inp.shape[1], tok.eos_token_id, False),
                                 ForceEosOnRepetition(inp.shape[1], tok.eos_token_id)])
    kw = dict(do_sample=True, **SAMPLE_KW) if sample else dict(do_sample=False)
    with torch.no_grad():
        out = model.generate(inp, attention_mask=torch.ones_like(inp), max_new_tokens=max_new,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                             logits_processor=procs, **kw)
    rows = []
    for j in range(len(id_lists)):
        row = out[j, inp.shape[1]:].tolist()
        if tok.eos_token_id in row:
            row = row[: row.index(tok.eos_token_id) + 1]
        rows.append(row)
    return rows


def spaced_softmax_select(scores, n_pos, k, min_dist, temperature, rng):
    remaining, selected = list(range(n_pos)), []
    for _ in range(k):
        if not remaining:
            break
        s = np.array([scores[i] for i in remaining]) / temperature
        g = -np.log(-np.log(np.clip(rng.random(len(remaining)), 1e-12, 1 - 1e-12)))
        pick = remaining[int(np.argmax(s + g))]
        selected.append(pick)
        remaining = [i for i in remaining if abs(i - pick) > min_dist]
    return selected


rng_order = random.Random(SEED)
order = list(range(len(problems)))
rng_order.shuffle(order)
dumps = []
for idx in order:
    if len(dumps) >= N_WRONG:
        break
    pb = problems[idx]
    pids = prompt_ids_for(pb["problem"])
    gen = generate([pids], GEN_MAX, sample=True)[0]
    ans = extract_answer(tok.decode(gen, skip_special_tokens=True))
    if ans is not None and abs(ans - pb["ref"]) < 1e-4:
        continue  # greedy-correct: the tournament skips these too
    base, n_pos = len(pids), len(gen)
    with torch.no_grad():
        out = model(torch.tensor([pids + gen], device=DEVICE), output_hidden_states=True)
    hs_probe = out.hidden_states[LAYER][0, base - 1: base + n_pos - 1].float()
    scores = (((hs_probe - p_mean) / p_scale) @ p_coef + probe["intercept"]).tolist()
    z = np.array(scores)
    z = (z - z.mean()) / (z.std() + 1e-8)
    min_dist = max(1, n_pos // (K_POINTS * 2))
    seg = [(round(s * n_pos / K_POINTS), round((s + 1) * n_pos / K_POINTS)) for s in range(K_POINTS)]
    strategies = {
        "pure_probe": sorted(range(n_pos), key=lambda i: -z[i])[:K_POINTS],
        "hybrid_spaced_softmax": spaced_softmax_select(z, n_pos, K_POINTS, min_dist, 1.0, np.random.default_rng(idx)),
        "uniform": sorted(set(int(i * (n_pos - 1) / (K_POINTS - 1)) for i in range(K_POINTS))),
        "stratified_probe": [a + int(np.argmax(z[a:b])) for a, b in seg if b > a],
    }
    cuts = {}
    for cut in sorted(set(p for pts in strategies.values() for p in pts)):
        logits = model.lm_head(out.hidden_states[28][0, base + cut - 1])
        topv, topi = torch.topk(torch.softmax(logits.float(), -1), N_BRANCHES)
        branches = []
        conts = generate([pids + gen[:cut] + [t] for t in topi.tolist()], CONT_MAX, sample=False)
        for t, pr, cont in zip(topi.tolist(), topv.tolist(), conts):
            text = tok.decode([t] + cont, skip_special_tokens=True)
            a = extract_answer(text)
            branches.append({"token": tok.decode([t]), "prob": pr, "answer": a, "len": len(cont) + 1,
                             "correct": a is not None and abs(a - pb["ref"]) < 1e-4, "text": text})
        cuts[cut] = {"original_token": tok.decode([gen[cut]]), "probe_z": float(z[cut]),
                     "context_before": tok.decode(gen[max(0, cut - 40):cut]), "branches": branches}
    dumps.append({
        "test_index": idx, "problem": pb["problem"], "reference": pb["ref"], "reference_solution": pb["solution"],
        "trace": tok.decode(gen, skip_special_tokens=True), "trace_answer": ans, "n_tokens": n_pos,
        "token_texts": [tok.decode([t]) for t in gen], "probe_z": z.tolist(),
        "strategies": strategies, "cuts": {str(k): v for k, v in cuts.items()},
    })
    print(f"problem {idx}: {n_pos} tokens, {len(cuts)} unique cuts, "
          f"rescued: " + ", ".join(f"{s}={any(any(b['correct'] for b in cuts[p]['branches']) for p in pts)}"
                                   for s, pts in strategies.items()), flush=True)
json.dump(dumps, open(OUT, "w"))
print("wrote", OUT)
