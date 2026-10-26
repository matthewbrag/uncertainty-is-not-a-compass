"""
Formalizes a gap flagged late in the project: every tournament so far forces
the model's TOP-N most-likely continuations at a branch point (deterministic).
The alternative -- sample N continuations from the softmax distribution
itself, weighted by probability, without replacement -- was never tested.
Does it surface more / less outcome diversity than top-N at the same budget?

Isolates the branch-GENERATION axis from the branch-SELECTION axis: point
selection is fixed to `uniform` (the proven winner from the diversity
tournament), so this is a clean A/B on top-N vs softmax-sampled branches at
the identical positions, same budget (5 branches either way).

Usage: CUDA_VISIBLE_DEVICES=<gpu> python softmax_branches_check.py <model_key> <level> <n_problems> [offset] [out_path]
"""
import json
import re
import sys
import time
import random

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import LogitsProcessorList

from app import (
    get_model, DEFAULT_TEMPLATE, extract_answer,
    ForceEosAfterAnswer, ForceEosOnRepetition, DEVICE,
)

MODEL_KEY = sys.argv[1]
LEVEL = int(sys.argv[2])
N_PROBLEMS = int(sys.argv[3]) if len(sys.argv) > 3 else 150
SHARD_OFFSET = int(sys.argv[4]) if len(sys.argv) > 4 else 0
OUT_PATH = sys.argv[5] if len(sys.argv) > 5 else f"softmax_branches_level{LEVEL}_{MODEL_KEY}_offset{SHARD_OFFSET}.jsonl"

K_POINTS = 5
N_BRANCHES = 5
GEN_MAX_NEW_TOKENS = 500
CONT_MAX_NEW_TOKENS = 300
MAX_BRANCH_BATCH = 40
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)
FINAL_LAYER = 28


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


def sample_without_replacement(probs, k):
    """Gumbel-top-k: a proper softmax-weighted sample of k distinct tokens,
    without replacement, in one shot (no duplicate branches)."""
    probs = probs.clamp(min=1e-12)
    gumbel = -torch.log(-torch.log(torch.rand_like(probs)))
    scores = torch.log(probs) + gumbel
    _, topi = torch.topk(scores, k)
    return topi.tolist()


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


def generate_batch_sampled(prompt_ids_list, max_new_tokens):
    max_len = max(len(x) for x in prompt_ids_list)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.tensor(
        [[pad_id] * (max_len - len(x)) + x for x in prompt_ids_list], device=DEVICE
    )
    attention_mask = torch.tensor(
        [[0] * (max_len - len(x)) + [1] * len(x) for x in prompt_ids_list], device=DEVICE
    )
    prompt_len = input_ids.shape[1]
    proc_a = ForceEosAfterAnswer(tokenizer, prompt_len, tokenizer.eos_token_id, False)
    proc_r = ForceEosOnRepetition(prompt_len, tokenizer.eos_token_id)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
            do_sample=True, **SAMPLE_KW,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList([proc_a, proc_r]),
        )
    results = []
    for j in range(len(prompt_ids_list)):
        row = out[j, prompt_len:].tolist()
        if tokenizer.eos_token_id in row:
            row = row[: row.index(tokenizer.eos_token_id) + 1]
        results.append(row)
    return results


def _force_batch(prefixes_and_tokens, max_new_tokens):
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
    proc_a = ForceEosAfterAnswer(tokenizer, prompt_len, tokenizer.eos_token_id, False)
    proc_r = ForceEosOnRepetition(prompt_len, tokenizer.eos_token_id)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList([proc_a, proc_r]),
        )
    texts = []
    for j, (prefix, tok) in enumerate(prefixes_and_tokens):
        row = out[j, prompt_len:].tolist()
        if tokenizer.eos_token_id in row:
            row = row[: row.index(tokenizer.eos_token_id) + 1]
        texts.append(tokenizer.decode([tok] + row, skip_special_tokens=True))
    return texts


def force_all_points(prefixes_and_tokens, max_new_tokens):
    texts = []
    for start in range(0, len(prefixes_and_tokens), MAX_BRANCH_BATCH):
        texts.extend(_force_batch(prefixes_and_tokens[start:start + MAX_BRANCH_BATCH], max_new_tokens))
    return texts


random.seed(7000)
all_indices = list(range(len(math_test)))
random.shuffle(all_indices)
problem_indices = all_indices[SHARD_OFFSET:SHARD_OFFSET + N_PROBLEMS]

GEN_BATCH_SIZE = 64
n_done = 0
n_skipped_correct_greedy = 0
n_skipped_too_short = 0
t_start = time.time()

with open(OUT_PATH, "w") as f:
  for gbatch_start in range(0, len(problem_indices), GEN_BATCH_SIZE):
    gbatch_indices = problem_indices[gbatch_start:gbatch_start + GEN_BATCH_SIZE]
    examples = [math_test[idx] for idx in gbatch_indices]
    prompt_ids_batch = [build_prompt_ids(ex["problem"]) for ex in examples]
    gen_ids_batch = generate_batch_sampled(prompt_ids_batch, GEN_MAX_NEW_TOKENS)

    for idx, ex, prompt_ids, gen_ids in zip(gbatch_indices, examples, prompt_ids_batch, gen_ids_batch):
        if len(gen_ids) < K_POINTS + 2:
            n_skipped_too_short += 1
            continue

        ref = ex["reference_answer"]
        greedy_answer = extract_answer(tokenizer.decode(gen_ids, skip_special_tokens=True))
        greedy_correct = greedy_answer is not None and ref is not None and abs(greedy_answer - ref) < 1e-4
        if greedy_correct:
            n_skipped_correct_greedy += 1
            continue
        n_done += 1

        full_ids = prompt_ids + gen_ids
        base = len(prompt_ids)
        n_pos = len(gen_ids)

        if n_pos <= K_POINTS:
            uniform_points = list(range(n_pos))
        else:
            uniform_points = sorted(set(int(i * (n_pos - 1) / (K_POINTS - 1)) for i in range(K_POINTS)))

        with torch.no_grad():
            out = model(input_ids=torch.tensor([full_ids], device=DEVICE))

        topk_pairs, sample_pairs, point_order = [], [], []
        for cut in uniform_points:
            pos = base + cut - 1
            logits = out.logits[0, pos]
            probs = torch.softmax(logits, dim=-1)
            topv, topi = torch.topk(probs, N_BRANCHES)
            sampled_idx = sample_without_replacement(probs, N_BRANCHES)
            prefix = prompt_ids + gen_ids[:cut]  # forced token REPLACES gen_ids[cut] (hs at base+cut-1 predicts it)
            topk_pairs.extend((prefix, tok) for tok in topi.tolist())
            sample_pairs.extend((prefix, tok) for tok in sampled_idx)
            point_order.append(cut)

        topk_texts = force_all_points(topk_pairs, CONT_MAX_NEW_TOKENS)
        sample_texts = force_all_points(sample_pairs, CONT_MAX_NEW_TOKENS)

        points_out = {}
        t_idx = 0
        for cut in point_order:
            topk_branch = topk_texts[t_idx: t_idx + N_BRANCHES]
            sample_branch = sample_texts[t_idx: t_idx + N_BRANCHES]
            t_idx += N_BRANCHES
            topk_answers = [extract_answer(t) for t in topk_branch]
            sample_answers = [extract_answer(t) for t in sample_branch]
            points_out[cut] = {"topk_answers": topk_answers, "sample_answers": sample_answers}

        def rescued(answers_key):
            return any(
                any(a is not None and ref is not None and abs(a - ref) < 1e-4 for a in points_out[c][answers_key])
                for c in point_order
            )

        record = {
            "math_test_index": idx,
            "trace_len": n_pos,
            "reference_answer": ref,
            "points": points_out,
            "topk_rescued": rescued("topk_answers"),
            "sample_rescued": rescued("sample_answers"),
        }
        f.write(json.dumps(record) + "\n")
        f.flush()

        elapsed = time.time() - t_start
        if n_done % 5 == 0:
            rate = elapsed / n_done
            eta = (N_PROBLEMS - n_done - n_skipped_correct_greedy - n_skipped_too_short) * rate
            print(f"[{MODEL_KEY}] {n_done} scored (skipped {n_skipped_correct_greedy} already-correct, "
                  f"{n_skipped_too_short} too-short), elapsed={elapsed/60:.1f}min, "
                  f"eta={eta/60:.1f}min", flush=True)

print(f"\nDone. {n_done} problems scored, {n_skipped_correct_greedy} skipped (greedy already correct), "
      f"{n_skipped_too_short} skipped (trace too short). Results in {OUT_PATH}")
