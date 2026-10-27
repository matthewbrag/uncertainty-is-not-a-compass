"""
What the user actually asked for: no predefined stopword list, no imposed
judgment of what's "filler" - just raw frequency counts of what words the
UNFILTERED probe_p / probe_signed_entropy strategies actually select, so we
can look through the top N ourselves and judge.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python filler_rate_check.py
"""
import json
import math
from collections import Counter

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import LogitsProcessorList

from forkprobe.hf import get_model, DEFAULT_TEMPLATE, extract_answer, ForceEosAfterAnswer, ForceEosOnRepetition, DEVICE

MODEL_KEY = "qwen3-1.7b"
LEVEL = 4
PTARGET_PROBE_PATH = "entropy_probe_1.7b_math_level4_layer14.json"
N_PROBLEMS = 60
K_POINTS = 5
GEN_MAX_NEW_TOKENS = 500
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)


def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


with open(PTARGET_PROBE_PATH) as f:
    _artifact = json.load(f)
PTARGET_LAYER = _artifact["layer"]

tokenizer, model = get_model(MODEL_KEY)
tokenizer.padding_side = "left"

_mean = torch.tensor(_artifact["scaler_mean"], dtype=torch.float32, device=DEVICE)
_scale = torch.tensor(_artifact["scaler_scale"], dtype=torch.float32, device=DEVICE)
_coef = torch.tensor(_artifact["coef"], dtype=torch.float32, device=DEVICE)
_intercept = _artifact["intercept"]


def predict_outcome_p(hs):
    x = (hs.float() - _mean) / _scale
    p = float(torch.dot(x, _coef).item() + _intercept)
    return min(max(p, 1e-6), 1 - 1e-6)


_subjects = ["algebra", "counting_and_probability", "intermediate_algebra",
             "number_theory", "prealgebra", "precalculus"]
_full_test = concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", s, split="test") for s in _subjects])


def extract_boxed(sol):
    idx = sol.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(sol) and depth > 0:
        if sol[i] == "{":
            depth += 1
        elif sol[i] == "}":
            depth -= 1
        i += 1
    return sol[start:i - 1] if depth == 0 else None


def parse_numeric(a):
    a = a.strip().rstrip(".")
    try:
        return float(a)
    except ValueError:
        return None


problems = []
for r in _full_test:
    if r["level"] != f"Level {LEVEL}":
        continue
    boxed = extract_boxed(r["solution"])
    ref = parse_numeric(boxed) if boxed else None
    if ref is not None:
        problems.append(r["problem"])
    if len(problems) >= N_PROBLEMS:
        break


def build_prompt_ids(q):
    content = DEFAULT_TEMPLATE.replace("{question}", q)
    text = tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                          tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer.encode(text, add_special_tokens=False)


probe_p_counts = Counter()
signed_counts = Counter()
n_traces = 0

for q in problems:
    prompt_ids = build_prompt_ids(q)
    input_ids = torch.tensor([prompt_ids], device=DEVICE)
    prompt_len = input_ids.shape[1]
    proc_a = ForceEosAfterAnswer(tokenizer, prompt_len, tokenizer.eos_token_id, False)
    proc_r = ForceEosOnRepetition(prompt_len, tokenizer.eos_token_id)
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=True, **SAMPLE_KW,
                              eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                              logits_processor=LogitsProcessorList([proc_a, proc_r]))
    row = out[0, prompt_len:].tolist()
    if tokenizer.eos_token_id in row:
        row = row[: row.index(tokenizer.eos_token_id) + 1]
    gen_ids = row
    n_pos = len(gen_ids)
    if n_pos < K_POINTS + 2:
        continue

    full_ids = prompt_ids + gen_ids
    with torch.no_grad():
        hout = model(input_ids=torch.tensor([full_ids], device=DEVICE), output_hidden_states=True)
    base = len(prompt_ids)

    ptarget_trace = [predict_outcome_p(hout.hidden_states[PTARGET_LAYER][0, base + i - 1]) for i in range(n_pos)]
    probe_p_points = sorted(range(n_pos), key=lambda i: -ptarget_trace[i])[:K_POINTS]

    mean_p = sum(ptarget_trace) / n_pos
    above = [i for i in range(n_pos) if ptarget_trace[i] > mean_p]
    above.sort(key=lambda i: -binary_entropy(ptarget_trace[i]))
    signed_points = above[:K_POINTS] if len(above) >= K_POINTS else above

    n_traces += 1
    for i in probe_p_points:
        word = tokenizer.decode([gen_ids[i]]).strip().lower()
        probe_p_counts[word] += 1
    for i in signed_points:
        word = tokenizer.decode([gen_ids[i]]).strip().lower()
        signed_counts[word] += 1

print(f"n_traces={n_traces}  total_probe_p_points={sum(probe_p_counts.values())}  "
      f"total_signed_points={sum(signed_counts.values())}", flush=True)

print("\n=== probe_p: top 40 most frequently selected words ===")
for word, count in probe_p_counts.most_common(40):
    print(f"  {count:4d}  {word!r}")

print("\n=== probe_signed_entropy: top 40 most frequently selected words ===")
for word, count in signed_counts.most_common(40):
    print(f"  {count:4d}  {word!r}")
