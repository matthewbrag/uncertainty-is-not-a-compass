"""
Second-generation tournament, built from what the corrected tournament's
debugging showed (NOTES.md "CORRECTED run"):
  - uniform's edge is mostly its cut at token 0 (a 5-way restart), so restart
    is held equal across the probe-vs-uniform comparison
  - the probe's top picks are mostly digits, skewed late, and clumped, so the
    probe arms pick per-segment and skip digit tokens
  - final-answer-line cuts "rescue" by guessing a different answer digit, so
    every arm also reports a rescue rate that ignores cuts past the answer marker
  - forced-token + greedy is kept as the main branching mechanism (greedy
    accuracy 54.9% vs sampled 52.8% on 288 problems -- not a handicap), plus
    sampled arms for the natural "just branch here" version and best-of-N

Arms:
  uniform                    5 evenly spaced cuts incl. token 0 and EOS (unchanged control)
  restart_uniform_interior   cut 0 + 4 evenly spaced cuts before the answer marker
  restart_stratified_probe   cut 0 + argmax probe (non-digit) in each of 4 segments
  stratified_probe_nodigit   argmax probe (non-digit) in each of 5 segments, no restart
  resample_uniform_cuts      uniform's positions, 5 SAMPLED continuations each, no forcing
  sampled_restarts           25 sampled full solutions (best-of-25)

Workers pull chunks of problems from a shared counter file, so no GPU idles
while another finishes a hard shard. Saves trace token ids for post-hoc
classification of every rescue.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python advanced_tournament.py <probe_path> <queue_file> <out_path> [max_problems]
"""
import fcntl
import json
import math
import random
import re
import sys
import time

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import LogitsProcessorList

from forkprobe.hf import (
    get_model, DEFAULT_TEMPLATE, extract_answer,
    ForceEosAfterAnswer, ForceEosOnRepetition, DEVICE,
)

PROBE_PATH, QUEUE, OUT_PATH = sys.argv[1], sys.argv[2], sys.argv[3]
MAX_PROBLEMS = int(sys.argv[4]) if len(sys.argv) > 4 else None
LEVEL = 4
CHUNK = 8
N_BRANCHES = 5
N_RESTARTS = 25
GEN_MAX_NEW_TOKENS = 1500
CONT_MAX_NEW_TOKENS = 1300
MAX_BRANCH_BATCH = 15
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)
ANSWER_MARKERS = ["Final Answer", "\\boxed", "####"]


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
    start = i = idx + len("\\boxed{")
    depth = 1
    while i < len(sol) and depth > 0:
        depth += {"{": 1, "}": -1}.get(sol[i], 0)
        i += 1
    return sol[start:i - 1] if depth == 0 else None


subjects = ["algebra", "counting_and_probability", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]
math_test = []
for r in concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", s, split="test") for s in subjects]):
    if r["level"] != f"Level {LEVEL}":
        continue
    b = extract_boxed(r["solution"])
    ref = parse_numeric(b) if b is not None else None
    if ref is not None:
        math_test.append({"problem": r["problem"], "reference_answer": ref})
order = list(range(len(math_test)))
random.Random(7000).shuffle(order)  # same problem order as every earlier tournament
if MAX_PROBLEMS:
    order = order[:MAX_PROBLEMS]

probe = json.load(open(PROBE_PATH))
PROBE_LAYER = probe["layer"]
tokenizer, model = get_model("qwen3-1.7b")
tokenizer.padding_side = "left"
p_mean = torch.tensor(probe["scaler_mean"], dtype=torch.float32, device=DEVICE)
p_scale = torch.tensor(probe["scaler_scale"], dtype=torch.float32, device=DEVICE)
p_coef = torch.tensor(probe["coef"], dtype=torch.float32, device=DEVICE)


def claim_chunk():
    """Atomically take the next CHUNK problem positions from the shared counter."""
    with open(QUEUE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        s = f.read().strip()
        start = int(s) if s else 0
        f.seek(0)
        f.truncate()
        f.write(str(start + CHUNK))
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return start


def build_prompt_ids(question):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": DEFAULT_TEMPLATE.replace("{question}", question)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def generate_rows(id_lists, max_new_tokens, sample, seed=None):
    """Left-padded batched generation; returns the new tokens per row, cut at EOS."""
    if seed is not None:
        torch.manual_seed(seed)
    L = max(len(x) for x in id_lists)
    pad = tokenizer.pad_token_id
    input_ids = torch.tensor([[pad] * (L - len(x)) + x for x in id_lists], device=DEVICE)
    attention_mask = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in id_lists], device=DEVICE)
    procs = LogitsProcessorList([ForceEosAfterAnswer(tokenizer, L, tokenizer.eos_token_id, False),
                                 ForceEosOnRepetition(L, tokenizer.eos_token_id)])
    kw = dict(do_sample=True, **SAMPLE_KW) if sample else dict(do_sample=False)
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
                             eos_token_id=tokenizer.eos_token_id, pad_token_id=pad, logits_processor=procs, **kw)
    rows = []
    for j in range(len(id_lists)):
        row = out[j, L:].tolist()
        if tokenizer.eos_token_id in row:
            row = row[: row.index(tokenizer.eos_token_id) + 1]
        rows.append(row)
    return rows


def generate_chunked(id_lists, max_new_tokens, sample, seed=None):
    rows = []
    for k, start in enumerate(range(0, len(id_lists), MAX_BRANCH_BATCH)):
        rows.extend(generate_rows(id_lists[start:start + MAX_BRANCH_BATCH], max_new_tokens, sample,
                                  None if seed is None else seed + k))
    return rows


def score(text, ref):
    a = extract_answer(text)
    return a, (a is not None and ref is not None and abs(a - ref) < 1e-4)


def answer_start(gen_ids):
    """Token index where the final-answer line begins (n_pos if no marker).
    Binary search on decoded prefix length -- cheaper than decoding every prefix."""
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    pos = max(text.rfind(m) for m in ANSWER_MARKERS)
    if pos < 0:
        return len(gen_ids)
    lo, hi = 0, len(gen_ids)
    while lo < hi:  # smallest k whose decoded prefix reaches past the marker's first char
        mid = (lo + hi) // 2
        if len(tokenizer.decode(gen_ids[:mid + 1], skip_special_tokens=True)) > pos:
            hi = mid
        else:
            lo = mid + 1
    return lo


def segments(lo, hi, k):
    return [(lo + (hi - lo) * s // k, lo + (hi - lo) * (s + 1) // k) for s in range(k)]


def stratified(z, is_digit, lo, hi, k):
    # min gap stops two picks landing either side of a segment boundary
    # (seen in the smoke test: 27/29, 126/131)
    min_gap = max(1, (hi - lo) // (k * 4))
    picks = []
    for a, b in segments(lo, hi, k):
        if b <= a:
            continue
        seg = [i for i in range(a, b) if all(abs(i - p) >= min_gap for p in picks)] or list(range(a, b))
        cand = [i for i in seg if not is_digit[i]] or seg
        picks.append(max(cand, key=lambda i: z[i]))
    return picks


t_start = time.time()
n_done = n_skipped = 0
with open(OUT_PATH, "w") as f:
    while True:
        start = claim_chunk()
        if start >= len(order):
            break
        idxs = order[start:start + CHUNK]
        examples = [math_test[i] for i in idxs]
        prompts = [build_prompt_ids(ex["problem"]) for ex in examples]
        traces = generate_rows(prompts, GEN_MAX_NEW_TOKENS, sample=True)

        for idx, ex, pids, gen in zip(idxs, examples, prompts, traces):
            ref = ex["reference_answer"]
            _, greedy_ok = score(tokenizer.decode(gen, skip_special_tokens=True), ref)
            if greedy_ok or len(gen) < 12:
                n_skipped += 1
                continue
            base, n_pos = len(pids), len(gen)
            with torch.no_grad():
                out = model(input_ids=torch.tensor([pids + gen], device=DEVICE), output_hidden_states=True,
                            logits_to_keep=1, use_cache=False)
            hs = out.hidden_states[PROBE_LAYER][0, base - 1: base + n_pos - 1].float()
            z = (((hs - p_mean) / p_scale) @ p_coef + probe["intercept"]).cpu().numpy()
            z = (z - z.mean()) / (z.std() + 1e-8)
            tok_texts = [tokenizer.decode([t]) for t in gen]
            is_digit = [any(c.isdigit() for c in t) for t in tok_texts]
            ans_start = answer_start(gen)
            interior_hi = max(2, min(ans_start, n_pos - 1))  # never past the marker or onto EOS

            uniform = sorted(set(int(i * (n_pos - 1) / 4) for i in range(5)))
            arms = {
                "uniform": uniform,
                "restart_uniform_interior": sorted({0} | {interior_hi * (i + 1) // 5 for i in range(4)}),
                "restart_stratified_probe": sorted({0} | set(stratified(z, is_digit, 1, interior_hi, 4))),
                "stratified_probe_nodigit": sorted(set(stratified(z, is_digit, 1, interior_hi, 5))),
            }

            # forced-token branches (greedy), each unique cut computed once
            cuts = sorted(set(p for pts in arms.values() for p in pts))
            pairs, owner = [], []
            for cut in cuts:
                logits = model.lm_head(out.hidden_states[28][0, base + cut - 1]).float()
                topi = torch.topk(torch.softmax(logits, -1), N_BRANCHES).indices.tolist()
                for t in topi:
                    pairs.append(pids + gen[:cut] + [t])  # forced token REPLACES gen[cut]
                    owner.append((cut, t))
            rows = generate_chunked(pairs, CONT_MAX_NEW_TOKENS, sample=False)
            forced = {c: {"tokens": [], "answers": [], "correct": [], "lengths": []} for c in cuts}
            for (cut, t), row in zip(owner, rows):
                a, ok = score(tokenizer.decode([t] + row, skip_special_tokens=True), ref)
                d = forced[cut]
                d["tokens"].append(t); d["answers"].append(a); d["correct"].append(ok); d["lengths"].append(len(row) + 1)
            del out

            # resampled branches at uniform's cut positions (no forcing)
            rs_inputs = [pids + gen[:cut] for cut in uniform for _ in range(N_BRANCHES)]
            rs_rows = generate_chunked(rs_inputs, CONT_MAX_NEW_TOKENS, sample=True, seed=idx * 1000 + 1)
            resampled = {c: {"answers": [], "correct": [], "lengths": []} for c in uniform}
            for k, row in enumerate(rs_rows):
                cut = uniform[k // N_BRANCHES]
                a, ok = score(tokenizer.decode(row, skip_special_tokens=True), ref)
                resampled[cut]["answers"].append(a); resampled[cut]["correct"].append(ok)
                resampled[cut]["lengths"].append(len(row))

            # best-of-N sampled restarts
            rst_rows = generate_chunked([pids] * N_RESTARTS, GEN_MAX_NEW_TOKENS, sample=True, seed=idx * 1000 + 500)
            restarts = [dict(zip(("answer", "correct"), score(tokenizer.decode(r, skip_special_tokens=True), ref)),
                             length=len(r)) for r in rst_rows]

            def arm_result(pts, table):
                ok = {p: any(table[p]["correct"]) for p in pts}
                return {"points": pts, "rescued": any(ok.values()),
                        "rescued_excl_answer_line": any(v for p, v in ok.items() if p < ans_start),
                        "tokens": sum(sum(table[p]["lengths"]) for p in pts)}

            results = {name: arm_result(pts, forced) for name, pts in arms.items()}
            results["resample_uniform_cuts"] = arm_result(uniform, resampled)
            results["sampled_restarts"] = {"rescued": any(r["correct"] for r in restarts),
                                           "rescued_first5": any(r["correct"] for r in restarts[:5]),
                                           "tokens": sum(r["length"] for r in restarts)}
            f.write(json.dumps({
                "math_test_index": idx, "reference_answer": ref, "trace_len": n_pos,
                "answer_start": ans_start, "prompt_token_ids": pids, "generated_token_ids": gen,
                "probe_z": [round(float(v), 3) for v in z],
                "forced": {str(k): v for k, v in forced.items()},
                "resampled": {str(k): v for k, v in resampled.items()},
                "restarts": restarts, "results": results,
            }) + "\n")
            f.flush()
            n_done += 1
        el = time.time() - t_start
        print(f"[claimed up to {start + len(idxs)}/{len(order)}] scored={n_done} skipped={n_skipped} "
              f"elapsed={el / 60:.1f}min", flush=True)

print(f"\nDone. {n_done} scored, {n_skipped} skipped. Results in {OUT_PATH}")
