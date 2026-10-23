"""
Fixed version of the monotonic-noise + hybrid-strategy experiments: generates
each trace ONCE per problem and evaluates every arm (5 noise levels, the
spacing+softmax hybrid, plus uniform/random for free) against that SAME
trace, in one script execution. Removes the trace-resampling confound that
affected monotonic_noise_tournament.py and spaced_softmax_tournament.py
(those launched a separate process per arm, each independently regenerating
its own stochastic trace for "the same" problem). Points are unioned and
deduped before forcing branches, same efficiency trick as math_tournament_diversity.py,
so overlapping strategies don't pay for redundant forced completions.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python shared_trace_monotonic.py <model_key> <level> <diversity_probe_path> <n_problems> [offset]
"""
import json
import re
import sys
import time

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import LogitsProcessorList

from app import (
    get_model, DEFAULT_TEMPLATE, extract_answer,
    ForceEosAfterAnswer, ForceEosOnRepetition, DEVICE,
)

MODEL_KEY = sys.argv[1]
LEVEL = int(sys.argv[2])
DIVERSITY_PROBE_PATH = sys.argv[3]
N_PROBLEMS = int(sys.argv[4]) if len(sys.argv) > 4 else 300
SHARD_OFFSET = int(sys.argv[5]) if len(sys.argv) > 5 else 0
OUT_PATH_OVERRIDE = sys.argv[6] if len(sys.argv) > 6 else None

import math

K_POINTS = 5
N_BRANCHES = 5
GEN_MAX_NEW_TOKENS = 650
CONT_MAX_NEW_TOKENS = 450
MAX_BRANCH_BATCH = 40
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)
NOISE_SEED_BASE = 92000
HYBRID_TEMPERATURE = 1.0
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
OUT_PATH = OUT_PATH_OVERRIDE or f"shared_trace_monotonic_level{LEVEL}_{MODEL_KEY}_offset{SHARD_OFFSET}.jsonl"


def greedy_spaced_select(scores, n_pos, k, min_dist):
    remaining = set(range(n_pos))
    selected = []
    for _ in range(k):
        if not remaining:
            break
        best = max(remaining, key=lambda i: scores[i])
        selected.append(best)
        remaining = {i for i in remaining if abs(i - best) > min_dist}
    if len(selected) < k:
        chosen = set(selected)
        backfill = sorted((i for i in range(n_pos) if i not in chosen), key=lambda i: -scores[i])
        selected.extend(backfill[: k - len(selected)])
    return selected


def spaced_softmax_select(scores, n_pos, k, min_dist, temperature, rng):
    remaining = list(range(n_pos))
    selected = []
    for _ in range(k):
        if not remaining:
            break
        s = np.array([scores[i] for i in remaining]) / temperature
        u = rng.random(len(remaining))
        gumbel = -np.log(-np.log(np.clip(u, 1e-12, 1 - 1e-12)))
        perturbed = s + gumbel
        pick = remaining[int(np.argmax(perturbed))]
        selected.append(pick)
        remaining = [i for i in remaining if abs(i - pick) > min_dist]
    if len(selected) < k:
        chosen = set(selected)
        backfill = sorted((i for i in range(n_pos) if i not in chosen), key=lambda i: -scores[i])
        selected.extend(backfill[: k - len(selected)])
    return selected


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

with open(DIVERSITY_PROBE_PATH) as f:
    _diversity_artifact = json.load(f)
DIVERSITY_LAYER = _diversity_artifact["layer"]

tokenizer, model = get_model(MODEL_KEY)
tokenizer.padding_side = "left"

_div_mean = torch.tensor(_diversity_artifact["scaler_mean"], dtype=torch.float32, device=DEVICE)
_div_scale = torch.tensor(_diversity_artifact["scaler_scale"], dtype=torch.float32, device=DEVICE)
_div_coef = torch.tensor(_diversity_artifact["coef"], dtype=torch.float32, device=DEVICE)
_div_intercept = _diversity_artifact["intercept"]


def predict_diversity(hidden_state_vec):
    x = (hidden_state_vec.float() - _div_mean) / _div_scale
    return float(torch.dot(x, _div_coef).item() + _div_intercept)


def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def build_prompt_ids(question):
    content = DEFAULT_TEMPLATE.replace("{question}", question)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def generate_batch_sampled(prompt_ids_list, max_new_tokens):
    """Batched greedy-trace generation across many problems at once - the
    cross-problem batching win validated in bench_batching.py, applied here
    since the original version only batched branch-forcing, not this step."""
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
    results = []
    for j, (prefix, tok) in enumerate(prefixes_and_tokens):
        row = out[j, prompt_len:].tolist()
        if tokenizer.eos_token_id in row:
            row = row[: row.index(tokenizer.eos_token_id) + 1]
        text = tokenizer.decode([tok] + row, skip_special_tokens=True)
        results.append((text, len(row) + 1))  # +1 for the forced token itself
    return results


def force_all_points(prefixes_and_tokens, max_new_tokens):
    results = []
    for start in range(0, len(prefixes_and_tokens), MAX_BRANCH_BATCH):
        results.extend(_force_batch(prefixes_and_tokens[start:start + MAX_BRANCH_BATCH], max_new_tokens))
    return results


import random
random.seed(7000)
all_indices = list(range(len(math_test)))
random.shuffle(all_indices)
problem_indices = all_indices[SHARD_OFFSET:SHARD_OFFSET + N_PROBLEMS]

GEN_BATCH_SIZE = 64

n_done = 0
n_skipped_correct_greedy = 0
n_skipped_too_short = 0
strategy_names = [f"noise_alpha{a}" for a in ALPHAS] + ["hybrid_spaced_softmax", "random", "uniform"]
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
        with torch.no_grad():
            out = model(input_ids=torch.tensor([full_ids], device=DEVICE), output_hidden_states=True)
        base = len(prompt_ids)
        n_pos = len(gen_ids)

        diversity_trace = []
        for i in range(n_pos):
            pos = base + i - 1
            div_hs = out.hidden_states[DIVERSITY_LAYER][0, pos]
            diversity_trace.append(predict_diversity(div_hs))

        z = np.array(diversity_trace)
        z = (z - z.mean()) / (z.std() + 1e-8)

        rng = np.random.default_rng(NOISE_SEED_BASE + idx)
        noise = rng.standard_normal(n_pos)
        noise = (noise - noise.mean()) / (noise.std() + 1e-8)

        min_dist = max(1, n_pos // (K_POINTS * 2))

        strategies = {}
        for alpha in ALPHAS:
            mixed = np.sqrt(max(0.0, 1 - alpha)) * z + np.sqrt(max(0.0, alpha)) * noise
            strategies[f"noise_alpha{alpha}"] = sorted(range(n_pos), key=lambda i: -mixed[i])[:K_POINTS]

        hybrid_rng = np.random.default_rng(NOISE_SEED_BASE + 1 + idx)
        strategies["hybrid_spaced_softmax"] = spaced_softmax_select(
            z, n_pos, K_POINTS, min_dist, HYBRID_TEMPERATURE, hybrid_rng
        )

        random_points = list(range(n_pos))
        random.shuffle(random_points)
        strategies["random"] = random_points[:K_POINTS]

        if n_pos <= K_POINTS:
            strategies["uniform"] = list(range(n_pos))
        else:
            strategies["uniform"] = sorted(set(int(i * (n_pos - 1) / (K_POINTS - 1)) for i in range(K_POINTS)))

        all_points_union = sorted(set(p for pts in strategies.values() for p in pts))
        point_to_tokens = {}
        forced_pairs = []
        point_order = []
        for cut in all_points_union:
            pos = base + cut - 1
            hs_final = out.hidden_states[28][0, pos]
            logits = model.lm_head(hs_final)
            probs = torch.softmax(logits, dim=-1)
            topv, topi = torch.topk(probs, N_BRANCHES)
            forced_tokens = topi.tolist()
            point_to_tokens[cut] = forced_tokens
            prefix = prompt_ids + gen_ids[: cut + 1]
            forced_pairs.extend((prefix, tok) for tok in forced_tokens)
            point_order.append(cut)

        all_branches = force_all_points(forced_pairs, CONT_MAX_NEW_TOKENS)

        point_correct = {}
        point_lengths = {}
        t_idx = 0
        for cut in point_order:
            branch_results = all_branches[t_idx: t_idx + N_BRANCHES]
            t_idx += N_BRANCHES
            corrects = []
            lengths = []
            for text, length in branch_results:
                p = extract_answer(text)
                corrects.append(p is not None and ref is not None and abs(p - ref) < 1e-4)
                lengths.append(length)
            point_correct[cut] = corrects
            point_lengths[cut] = lengths

        strategy_results = {}
        for name, pts in strategies.items():
            any_correct = any(any(point_correct[p]) for p in pts)
            frac_correct = sum(sum(point_correct[p]) for p in pts) / (len(pts) * N_BRANCHES)
            strategy_results[name] = {"rescued": any_correct, "frac_correct": frac_correct, "points": pts}

        # per-forced-branch length + correctness, independent of strategy (a
        # branch's correctness/length doesn't depend on which named strategy
        # "picked" it) - lets us compute avg tokens-to-correct-answer, and
        # any other post-hoc length-based analysis, without re-running.
        points_detail = {
            cut: {"lengths": point_lengths[cut], "correct": point_correct[cut]}
            for cut in point_order
        }

        record = {
            "math_test_index": idx,
            "trace_len": n_pos,
            "greedy_correct": greedy_correct,
            "reference_answer": ref,
            "points_detail": points_detail,
            "strategy_results": strategy_results,
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
