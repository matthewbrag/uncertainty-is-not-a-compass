"""
Companion to measure_natural_length.py: measures the TRUE natural length of
FORCED BRANCH continuations specifically (not the initial trace), so
CONT_MAX_NEW_TOKENS can be calibrated from real data instead of guessed/
scaled proportionally from the initial-trace number.

Generates traces at the newly-calibrated initial-trace budget (passed in),
picks 3 candidate points per trace (25%/50%/75% through), forces one branch
per point with a generous ceiling, and reports the natural length
distribution of those forced branches specifically.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python measure_natural_branch_length.py <model_key> <level> <trace_budget> <n_problems> [offset] [out_path]
"""
import json
import re
import sys
import time

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import LogitsProcessorList

from app import (
    get_model, DEFAULT_TEMPLATE, extract_answer, ForceEosOnRepetition,
    DEVICE, detect_repetition,
)

MODEL_KEY = sys.argv[1]
LEVEL = int(sys.argv[2])
TRACE_BUDGET = int(sys.argv[3])
N_PROBLEMS = int(sys.argv[4]) if len(sys.argv) > 4 else 100
SHARD_OFFSET = int(sys.argv[5]) if len(sys.argv) > 5 else 0
OUT_PATH = sys.argv[6] if len(sys.argv) > 6 else f"natural_branch_length_level{LEVEL}_{MODEL_KEY}_offset{SHARD_OFFSET}.jsonl"

CEILING = 3000
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)
GEN_BATCH_SIZE = 16


def parse_numeric(a):
    a = a.strip().rstrip(".")
    try:
        return float(a)
    except ValueError:
        pass
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", a)
    if m:
        sign = -1.0 if m.group(1) else 1.0
        return sign * float(m.group(2)) / float(m.group(3))
    return None


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


_subjects = ["algebra", "counting_and_probability", "intermediate_algebra",
             "number_theory", "prealgebra", "precalculus"]
_full_test = concatenate_datasets(
    [load_dataset("EleutherAI/hendrycks_math", s, split="test") for s in _subjects]
)
math_test = []
for r in _full_test:
    if r["level"] != f"Level {LEVEL}":
        continue
    boxed = extract_boxed(r["solution"])
    ref = parse_numeric(boxed) if boxed is not None else None
    if ref is not None:
        math_test.append({"problem": r["problem"], "reference_answer": ref})
print(f"{len(math_test)} level-{LEVEL} numeric-parseable TEST problems available", flush=True)

tokenizer, model = get_model(MODEL_KEY)
tokenizer.padding_side = "left"


def build_prompt_ids(question):
    content = DEFAULT_TEMPLATE.replace("{question}", question)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def generate_trace_batch(prompt_ids_list):
    max_len = max(len(x) for x in prompt_ids_list)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.tensor(
        [[pad_id] * (max_len - len(x)) + x for x in prompt_ids_list], device=DEVICE
    )
    attention_mask = torch.tensor(
        [[0] * (max_len - len(x)) + [1] * len(x) for x in prompt_ids_list], device=DEVICE
    )
    prompt_len = input_ids.shape[1]
    proc_r = ForceEosOnRepetition(prompt_len, tokenizer.eos_token_id)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=TRACE_BUDGET,
            do_sample=True, **SAMPLE_KW,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList([proc_r]),
        )
    results = []
    for j in range(len(prompt_ids_list)):
        row = out[j, prompt_len:].tolist()
        if tokenizer.eos_token_id in row:
            row = row[: row.index(tokenizer.eos_token_id) + 1]
        results.append(row)
    return results


def force_branch_batch(prefixes_and_tokens):
    batch_ids = [prefix + [tok] for prefix, tok in prefixes_and_tokens]
    max_len = max(len(x) for x in batch_ids)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.tensor(
        [[pad_id] * (max_len - len(x)) + x for x in batch_ids], device=DEVICE
    )
    attention_mask = torch.tensor(
        [[0] * (max_len - len(x)) + [1] * len(x) for x in batch_ids], device=DEVICE
    )
    prompt_len = input_ids.shape[1]
    proc_r = ForceEosOnRepetition(prompt_len, tokenizer.eos_token_id)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=CEILING,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList([proc_r]),
        )
    results = []
    for j in range(len(prefixes_and_tokens)):
        row = out[j, prompt_len:].tolist()
        hit_eos = tokenizer.eos_token_id in row
        if hit_eos:
            row = row[: row.index(tokenizer.eos_token_id) + 1]
        results.append((row, hit_eos))
    return results


import random
random.seed(7000)
all_indices = list(range(len(math_test)))
random.shuffle(all_indices)
problem_indices = all_indices[SHARD_OFFSET:SHARD_OFFSET + N_PROBLEMS]

n_done = 0
t_start = time.time()

with open(OUT_PATH, "w") as f:
    for gbatch_start in range(0, len(problem_indices), GEN_BATCH_SIZE):
        gbatch_indices = problem_indices[gbatch_start:gbatch_start + GEN_BATCH_SIZE]
        examples = [math_test[idx] for idx in gbatch_indices]
        prompt_ids_batch = [build_prompt_ids(ex["problem"]) for ex in examples]
        gen_ids_batch = generate_trace_batch(prompt_ids_batch)

        # pick 3 points per trace (25/50/75%), force the single most likely
        # alternative token at each (greedy continuation, matching the real
        # pipeline's branch-forcing step)
        forced_pairs = []
        meta = []  # (idx, cut, ref) per forced pair, same order as forced_pairs
        for idx, ex, prompt_ids, gen_ids in zip(gbatch_indices, examples, prompt_ids_batch, gen_ids_batch):
            n_pos = len(gen_ids)
            if n_pos < 4:
                continue
            base = len(prompt_ids)
            full_ids = prompt_ids + gen_ids
            cuts = sorted(set(int(n_pos * f) for f in (0.25, 0.5, 0.75)))
            with torch.no_grad():
                hs = model(input_ids=torch.tensor([full_ids], device=DEVICE), output_hidden_states=True).hidden_states[-1][0]
            for cut in cuts:
                pos = base + cut - 1
                logits = model.lm_head(hs[pos])
                probs = torch.softmax(logits, dim=-1)
                top_tok = int(torch.argmax(probs).item())
                prefix = prompt_ids + gen_ids[:cut]  # forced token REPLACES gen_ids[cut] (hs at base+cut-1 predicts it)
                forced_pairs.append((prefix, top_tok))
                meta.append((idx, cut, ex["reference_answer"]))

        if not forced_pairs:
            continue
        # CEILING=3000 means a single sequence's KV cache can get huge, so
        # keep the branch-forcing micro-batch small regardless of how many
        # problems were in the outer generation batch.
        MAX_BRANCH_FORCE_BATCH = 6
        branch_results = []
        for start in range(0, len(forced_pairs), MAX_BRANCH_FORCE_BATCH):
            branch_results.extend(force_branch_batch(forced_pairs[start:start + MAX_BRANCH_FORCE_BATCH]))

        for (idx, cut, ref), (row, hit_eos) in zip(meta, branch_results):
            is_loop = detect_repetition(row) if not hit_eos else False
            hit_ceiling = not hit_eos and not is_loop
            text = tokenizer.decode(row, skip_special_tokens=True)
            pred = extract_answer(text)
            correct = pred is not None and ref is not None and abs(pred - ref) < 1e-4
            record = {
                "math_test_index": idx,
                "cut": cut,
                "length": len(row),
                "hit_natural_eos": hit_eos,
                "stopped_by_repetition_guard": is_loop,
                "hit_ceiling": hit_ceiling,
                "correct": correct,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

        n_done += len(gbatch_indices)
        elapsed = time.time() - t_start
        rate = elapsed / n_done
        eta = (N_PROBLEMS - n_done) * rate
        print(f"[{MODEL_KEY}] {n_done} problems done, elapsed={elapsed/60:.1f}min, eta={eta/60:.1f}min", flush=True)

print(f"\nDone. {n_done} problems. Results in {OUT_PATH}")
