"""
Model-parameterized version of strategy_tournament.py - the original hardcoded
qwen3-0.6b and relied on app.py's single global hardcoded probe. This loads
whatever probe artifact you point it at, for whatever model, so the same
tree-search-usefulness test can run on multiple model sizes.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python strategy_tournament_generic.py <model_key> <probe_path> <n_problems>
"""
import json
import sys
import time

import torch
from datasets import load_dataset
from transformers import LogitsProcessorList

from app import (
    get_model, DEFAULT_TEMPLATE, extract_answer, compute_entropy,
    ForceEosAfterAnswer, ForceEosOnRepetition, DEVICE,
)

MODEL_KEY = sys.argv[1]
PROBE_PATH = sys.argv[2]
PTARGET_PROBE_PATH = sys.argv[3]
N_PROBLEMS = int(sys.argv[4]) if len(sys.argv) > 4 else 400
SHARD_OFFSET = int(sys.argv[5]) if len(sys.argv) > 5 else 0

import math

K_POINTS = 5
N_BRANCHES = 5
GEN_MAX_NEW_TOKENS = 400
CONT_MAX_NEW_TOKENS = 250
MAX_BRANCH_BATCH = 40
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)
OUT_PATH = f"strategy_tournament_v4_{MODEL_KEY}_results_offset{SHARD_OFFSET}.jsonl"
SOFTMAX_TEMPERATURE = 0.1


def gumbel_noise(rand_fn):
    u = min(max(rand_fn(), 1e-12), 1 - 1e-12)
    return -math.log(-math.log(u))

with open(PROBE_PATH) as f:
    _probe_artifact = json.load(f)
PROBE_LAYER = _probe_artifact["layer"]

with open(PTARGET_PROBE_PATH) as f:
    _ptarget_artifact = json.load(f)
PTARGET_LAYER = _ptarget_artifact["layer"]

gsm8k_test = load_dataset("openai/gsm8k", "main")["test"]
tokenizer, model = get_model(MODEL_KEY)
tokenizer.padding_side = "left"

_probe_mean = torch.tensor(_probe_artifact["scaler_mean"], dtype=torch.float32, device=DEVICE)
_probe_scale = torch.tensor(_probe_artifact["scaler_scale"], dtype=torch.float32, device=DEVICE)
_probe_coef = torch.tensor(_probe_artifact["coef"], dtype=torch.float32, device=DEVICE)
_probe_intercept = _probe_artifact["intercept"]

_ptarget_mean = torch.tensor(_ptarget_artifact["scaler_mean"], dtype=torch.float32, device=DEVICE)
_ptarget_scale = torch.tensor(_ptarget_artifact["scaler_scale"], dtype=torch.float32, device=DEVICE)
_ptarget_coef = torch.tensor(_ptarget_artifact["coef"], dtype=torch.float32, device=DEVICE)
_ptarget_intercept = _ptarget_artifact["intercept"]


def predict_outcome_entropy(hidden_state_vec):
    x = (hidden_state_vec.float() - _probe_mean) / _probe_scale
    return float(torch.dot(x, _probe_coef).item() + _probe_intercept)


def predict_outcome_p(hidden_state_vec):
    x = (hidden_state_vec.float() - _ptarget_mean) / _ptarget_scale
    p = float(torch.dot(x, _ptarget_coef).item() + _ptarget_intercept)
    return min(max(p, 1e-6), 1 - 1e-6)


def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def layer_logits(hidden_state_vec, layer_idx):
    if layer_idx == 28:
        return model.lm_head(hidden_state_vec)
    return model.lm_head(model.model.norm(hidden_state_vec))


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


import random
random.seed(7000)
all_indices = list(range(len(gsm8k_test)))
random.shuffle(all_indices)
problem_indices = all_indices[SHARD_OFFSET:SHARD_OFFSET + N_PROBLEMS]

GEN_BATCH_SIZE = 64

n_done = 0
n_skipped_correct_greedy = 0
n_skipped_too_short = 0
strategy_names = ["probe_p", "probe_p_softmax", "probe_signed_entropy", "random", "uniform"]
t_start = time.time()

with open(OUT_PATH, "w") as f:
  for gbatch_start in range(0, len(problem_indices), GEN_BATCH_SIZE):
    gbatch_indices = problem_indices[gbatch_start:gbatch_start + GEN_BATCH_SIZE]
    examples = [gsm8k_test[idx] for idx in gbatch_indices]
    prompt_ids_batch = [build_prompt_ids(ex["question"]) for ex in examples]
    gen_ids_batch = generate_batch_sampled(prompt_ids_batch, GEN_MAX_NEW_TOKENS)

    for idx, ex, prompt_ids, gen_ids in zip(gbatch_indices, examples, prompt_ids_batch, gen_ids_batch):
        if len(gen_ids) < K_POINTS + 2:
            n_skipped_too_short += 1
            continue

        ref = extract_answer(ex["answer"])
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

        probe_trace = []
        ptarget_trace = []
        for i in range(n_pos):
            pos = base + i - 1
            probe_hs = out.hidden_states[PROBE_LAYER][0, pos]
            probe_trace.append(predict_outcome_entropy(probe_hs))
            ptarget_hs = out.hidden_states[PTARGET_LAYER][0, pos]
            ptarget_trace.append(predict_outcome_p(ptarget_hs))

        probe_points = sorted(range(n_pos), key=lambda i: -probe_trace[i])[:K_POINTS]

        # probe_p: chase predicted success probability directly (monotonic,
        # sidesteps entropy's direction-blindness)
        probe_p_points = sorted(range(n_pos), key=lambda i: -ptarget_trace[i])[:K_POINTS]

        # probe_signed_entropy: still "divided opinion matters", but only on
        # the side of this trace's own average that's actually useful -
        # points predicted ABOVE the trace's mean p, ranked by entropy of p
        trace_mean_p = sum(ptarget_trace) / n_pos
        above_mean = [i for i in range(n_pos) if ptarget_trace[i] > trace_mean_p]
        above_mean.sort(key=lambda i: -binary_entropy(ptarget_trace[i]))
        if len(above_mean) >= K_POINTS:
            signed_entropy_points = above_mean[:K_POINTS]
        else:
            remaining = [i for i in range(n_pos) if i not in set(above_mean)]
            remaining.sort(key=lambda i: -binary_entropy(ptarget_trace[i]))
            signed_entropy_points = above_mean + remaining[:K_POINTS - len(above_mean)]

        random_points = list(range(n_pos))
        random.shuffle(random_points)
        random_points = random_points[:K_POINTS]

        if n_pos <= K_POINTS:
            uniform_points = list(range(n_pos))
        else:
            uniform_points = sorted(set(int(i * (n_pos - 1) / (K_POINTS - 1)) for i in range(K_POINTS)))

        # probe_p_softmax: instead of hard top-K on predicted_p (which always
        # picks the same, possibly-clustered points), sample 5 DISTINCT
        # positions without replacement with probability proportional to
        # softmax(predicted_p / T) - Gumbel-top-k trick: add independent
        # Gumbel noise to each scaled score, take top-K of the perturbed
        # scores. Mostly grabs high scorers, sometimes medium, rarely low.
        perturbed = [ptarget_trace[i] / SOFTMAX_TEMPERATURE + gumbel_noise(random.random) for i in range(n_pos)]
        softmax_points = sorted(range(n_pos), key=lambda i: -perturbed[i])[:K_POINTS]

        strategies = {
            "probe_p": probe_p_points,
            "probe_p_softmax": softmax_points,
            "probe_signed_entropy": signed_entropy_points,
            "random": random_points,
            "uniform": uniform_points,
        }

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

        all_texts = force_all_points(forced_pairs, CONT_MAX_NEW_TOKENS)

        point_correct = {}
        t_idx = 0
        for cut in point_order:
            branch_texts = all_texts[t_idx: t_idx + N_BRANCHES]
            t_idx += N_BRANCHES
            corrects = []
            for t in branch_texts:
                p = extract_answer(t)
                corrects.append(p is not None and ref is not None and abs(p - ref) < 1e-4)
            point_correct[cut] = corrects

        strategy_results = {}
        for name, pts in strategies.items():
            any_correct = any(any(point_correct[p]) for p in pts)
            frac_correct = sum(sum(point_correct[p]) for p in pts) / (len(pts) * N_BRANCHES)
            strategy_results[name] = {"rescued": any_correct, "frac_correct": frac_correct, "points": pts}

        record = {
            "gsm8k_test_index": idx,
            "trace_len": n_pos,
            "greedy_correct": greedy_correct,
            "reference_answer": ref,
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
