"""
Tests whether probe_p / probe_signed_entropy actually beat random/uniform on
a problem distribution where baseline success sits BELOW 0.5 (MATH level 3
or 4 on 1.7B, per tonight's sweep: GSM8K=0.687 too easy, AIME=0.01 too hard,
MATH level3/4=~0.46 right in the target zone). If the entropy-direction
mechanism is real and causal, this is where probe-based selection should
start winning again - same model, same code, only the problem difficulty
changed.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python math_tournament.py <model_key> <level> <ptarget_probe_path> <n_problems> [offset]
"""
import json
import re
import sys
import time

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import LogitsProcessorList

from forkprobe.hf import (
    get_model, DEFAULT_TEMPLATE, extract_answer,
    ForceEosAfterAnswer, ForceEosOnRepetition, DEVICE,
)

MODEL_KEY = sys.argv[1]
LEVEL = int(sys.argv[2])
PTARGET_PROBE_PATH = sys.argv[3]
N_PROBLEMS = int(sys.argv[4]) if len(sys.argv) > 4 else 300
SHARD_OFFSET = int(sys.argv[5]) if len(sys.argv) > 5 else 0

import math

K_POINTS = 5
N_BRANCHES = 5
GEN_MAX_NEW_TOKENS = 500
CONT_MAX_NEW_TOKENS = 300
MAX_BRANCH_BATCH = 40
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)
OUT_PATH = f"math_tournament_filtered_level{LEVEL}_{MODEL_KEY}_offset{SHARD_OFFSET}.jsonl"


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

with open(PTARGET_PROBE_PATH) as f:
    _ptarget_artifact = json.load(f)
PTARGET_LAYER = _ptarget_artifact["layer"]

tokenizer, model = get_model(MODEL_KEY)
tokenizer.padding_side = "left"

_ptarget_mean = torch.tensor(_ptarget_artifact["scaler_mean"], dtype=torch.float32, device=DEVICE)
_ptarget_scale = torch.tensor(_ptarget_artifact["scaler_scale"], dtype=torch.float32, device=DEVICE)
_ptarget_coef = torch.tensor(_ptarget_artifact["coef"], dtype=torch.float32, device=DEVICE)
_ptarget_intercept = _ptarget_artifact["intercept"]


def predict_outcome_p(hidden_state_vec):
    x = (hidden_state_vec.float() - _ptarget_mean) / _ptarget_scale
    p = float(torch.dot(x, _ptarget_coef).item() + _ptarget_intercept)
    return min(max(p, 1e-6), 1 - 1e-6)


def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


FILLER_WORDS = {
    "step", "then", "now", "so", "the", "a", "an", "is", "to", "of", "and",
    "in", "on", "at", "this", "that", "we", "it", "its", "for", "with", "as",
    "by", "from", "which", "be", "are", "was", "were", "will", "would", "can",
    "could", "if", "but", "or", "not", "have", "has", "had", "do", "does",
    "did", "there", "here", "these", "those", "our", "us", "let", "lets",
    "i", "you", "your", "they", "he", "she", "him", "her", "next", "first",
    "second", "third", "finally", "therefore", "thus", "since",
    # LaTeX markup, per the actual frequency diagnostic - bigger junk
    # category than English stopwords for this MATH/LaTeX-heavy domain
    "frac", "sqrt", "quad", "qquad", "cdot", "times", "div", "pm", "mp",
    "infty", "left", "right", "begin", "end", "text", "mathrm", "boxed",
    "overline", "underline", "cdots", "ldots", "dots", "np", "nonumber",
    "displaystyle", "hline",
}


def is_filler_token(token_text):
    t = token_text.strip().lower()
    if not t:
        return True  # pure whitespace
    if re.fullmatch(r"[^a-zA-Z0-9]+", t):
        return True  # pure punctuation
    return t in FILLER_WORDS


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
all_indices = list(range(len(math_test)))
random.shuffle(all_indices)
problem_indices = all_indices[SHARD_OFFSET:SHARD_OFFSET + N_PROBLEMS]

GEN_BATCH_SIZE = 64

n_done = 0
n_skipped_correct_greedy = 0
n_skipped_too_short = 0
strategy_names = ["probe_p", "probe_p_filtered", "probe_signed_entropy", "probe_signed_entropy_filtered", "random", "uniform"]
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

        ptarget_trace = []
        for i in range(n_pos):
            pos = base + i - 1
            ptarget_hs = out.hidden_states[PTARGET_LAYER][0, pos]
            ptarget_trace.append(predict_outcome_p(ptarget_hs))

        is_filler = [is_filler_token(tokenizer.decode([gen_ids[i]])) for i in range(n_pos)]
        content_positions = [i for i in range(n_pos) if not is_filler[i]]
        if len(content_positions) < K_POINTS:
            content_positions = list(range(n_pos))  # fallback for degenerate short/filler-heavy traces

        def top_k_from(pool, key_fn):
            ranked = sorted(pool, key=lambda i: -key_fn(i))
            if len(ranked) >= K_POINTS:
                return ranked[:K_POINTS]
            rest = sorted((i for i in range(n_pos) if i not in set(ranked)), key=lambda i: -key_fn(i))
            return ranked + rest[:K_POINTS - len(ranked)]

        # probe_p: chase predicted success probability directly (monotonic,
        # sidesteps entropy's direction-blindness)
        probe_p_points = top_k_from(range(n_pos), lambda i: ptarget_trace[i])
        probe_p_filtered_points = top_k_from(content_positions, lambda i: ptarget_trace[i])

        # probe_signed_entropy: still "divided opinion matters", but only on
        # the side of this trace's own average that's actually useful -
        # points predicted ABOVE the trace's mean p, ranked by entropy of p
        def signed_entropy_select(pool):
            pool_mean_p = sum(ptarget_trace[i] for i in pool) / len(pool)
            above_mean = [i for i in pool if ptarget_trace[i] > pool_mean_p]
            return top_k_from(above_mean if len(above_mean) >= K_POINTS else pool,
                               lambda i: binary_entropy(ptarget_trace[i]))

        signed_entropy_points = signed_entropy_select(range(n_pos))
        signed_entropy_filtered_points = signed_entropy_select(content_positions)

        random_points = list(range(n_pos))
        random.shuffle(random_points)
        random_points = random_points[:K_POINTS]

        if n_pos <= K_POINTS:
            uniform_points = list(range(n_pos))
        else:
            uniform_points = sorted(set(int(i * (n_pos - 1) / (K_POINTS - 1)) for i in range(K_POINTS)))

        strategies = {
            "probe_p": probe_p_points,
            "probe_p_filtered": probe_p_filtered_points,
            "probe_signed_entropy": signed_entropy_points,
            "probe_signed_entropy_filtered": signed_entropy_filtered_points,
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
            prefix = prompt_ids + gen_ids[:cut]  # forced token REPLACES gen_ids[cut] (hs at base+cut-1 predicts it)
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
            "math_test_index": idx,
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
