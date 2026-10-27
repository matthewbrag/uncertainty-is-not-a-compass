"""
Diagnostic: what's the TRUE natural length distribution of MATH-L4 reasoning
traces, if we stop generation only on EOS or the repetition guard, not an
arbitrary token budget? Answers "what cap would actually be enough" instead
of guessing at a number (500->650 turned out to still truncate more than
expected - this measures the real distribution directly).

CEILING is set deliberately generous (3000) so it should essentially never
be the actual stopping reason - if it IS hit often, that's itself an
important finding (means repetition-guard + natural EOS alone don't resolve
in reasonable time for a meaningful fraction of problems).

Usage: CUDA_VISIBLE_DEVICES=<gpu> python measure_natural_length.py <model_key> <level> <n_problems> [offset] [out_path]
"""
import json
import re
import sys
import time

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import LogitsProcessorList

from forkprobe.hf import (
    get_model, DEFAULT_TEMPLATE, ForceEosAfterAnswer, ForceEosOnRepetition,
    DEVICE, detect_repetition,
)

MODEL_KEY = sys.argv[1]
LEVEL = int(sys.argv[2])
N_PROBLEMS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
SHARD_OFFSET = int(sys.argv[4]) if len(sys.argv) > 4 else 0
OUT_PATH = sys.argv[5] if len(sys.argv) > 5 else f"natural_length_level{LEVEL}_{MODEL_KEY}_offset{SHARD_OFFSET}.jsonl"

CEILING = 3000  # deliberately generous -- should rarely/never be the real stopping reason
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)
GEN_BATCH_SIZE = 16  # smaller than usual since sequences can get much longer now


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


def generate_batch(prompt_ids_list):
    max_len = max(len(x) for x in prompt_ids_list)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.tensor(
        [[pad_id] * (max_len - len(x)) + x for x in prompt_ids_list], device=DEVICE
    )
    attention_mask = torch.tensor(
        [[0] * (max_len - len(x)) + [1] * len(x) for x in prompt_ids_list], device=DEVICE
    )
    prompt_len = input_ids.shape[1]
    # NOTE: no ForceEosAfterAnswer here on purpose -- we want the TRUE natural
    # stopping point (does it keep rambling after reaching an answer?), only
    # the repetition guard stays, matching what every other script already uses.
    proc_r = ForceEosOnRepetition(prompt_len, tokenizer.eos_token_id)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=CEILING,
            do_sample=True, **SAMPLE_KW,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList([proc_r]),
        )
    results = []
    for j in range(len(prompt_ids_list)):
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
        results = generate_batch(prompt_ids_batch)

        for idx, ex, (gen_ids, hit_eos) in zip(gbatch_indices, examples, results):
            is_loop = detect_repetition(gen_ids) if not hit_eos else False
            hit_ceiling = not hit_eos and not is_loop
            record = {
                "math_test_index": idx,
                "length": len(gen_ids),
                "hit_natural_eos": hit_eos,
                "stopped_by_repetition_guard": is_loop,
                "hit_ceiling": hit_ceiling,
                "reference_answer": ex["reference_answer"],
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            n_done += 1

        elapsed = time.time() - t_start
        rate = elapsed / n_done
        eta = (N_PROBLEMS - n_done) * rate
        print(f"[{MODEL_KEY}] {n_done} done, elapsed={elapsed/60:.1f}min, eta={eta/60:.1f}min", flush=True)

print(f"\nDone. {n_done} problems. Results in {OUT_PATH}")
