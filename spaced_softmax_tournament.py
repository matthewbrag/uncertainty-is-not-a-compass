"""
Combines the two independently-validated "closes the gap toward uniform"
fixes from this project's history into one strategy, to see if stacking
them finally crosses from tied-with-uniform into actually beating it:

  1. Spacing (NMS-style): exclude a window around each pick before the next
     one, so picks can't cluster in the same neighborhood of the trace.
  2. Softmax sampling (Gumbel-top-k): instead of deterministically taking
     the single highest-scoring remaining candidate, sample weighted by
     softmax(score/T), so ties and near-ties get a real chance instead of
     always resolving the same way.

Here they're merged into one iterative procedure: at each of the 5 picks,
sample (Gumbel-softmax-weighted) from the REMAINING candidates, then apply
the same exclusion-zone removal before the next pick.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python spaced_softmax_tournament.py <model_key> <level> <diversity_probe_path> <temperature> <n_problems> [offset] [out_path]
"""
import json
import re
import sys
import time
import random

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
TEMPERATURE = float(sys.argv[4])
N_PROBLEMS = int(sys.argv[5]) if len(sys.argv) > 5 else 150
SHARD_OFFSET = int(sys.argv[6]) if len(sys.argv) > 6 else 0
OUT_PATH = sys.argv[7] if len(sys.argv) > 7 else f"spaced_softmax_t{TEMPERATURE}_level{LEVEL}_{MODEL_KEY}_offset{SHARD_OFFSET}.jsonl"

K_POINTS = 5
N_BRANCHES = 5
GEN_MAX_NEW_TOKENS = 500
CONT_MAX_NEW_TOKENS = 300
MAX_BRANCH_BATCH = 40
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)
NOISE_SEED_BASE = 91000


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

        with torch.no_grad():
            out = model(input_ids=torch.tensor([full_ids], device=DEVICE), output_hidden_states=True)

        diversity_trace = []
        for i in range(n_pos):
            pos = base + i - 1
            div_hs = out.hidden_states[DIVERSITY_LAYER][0, pos]
            diversity_trace.append(predict_diversity(div_hs))
        z = np.array(diversity_trace)
        z = (z - z.mean()) / (z.std() + 1e-8)

        rng = np.random.default_rng(NOISE_SEED_BASE + idx)
        min_dist = max(1, n_pos // (K_POINTS * 2))
        points = spaced_softmax_select(z, n_pos, K_POINTS, min_dist, TEMPERATURE, rng)

        prefix_cuts = []
        for cut in points:
            prefix = prompt_ids + gen_ids[:cut]  # forced token REPLACES gen_ids[cut] (hs at base+cut-1 predicts it)
            prefix_cuts.append((cut, prefix))

        forced_pairs = []
        point_order = []
        for cut, prefix in prefix_cuts:
            hs_final = out.hidden_states[28][0, base + cut - 1]
            logits = model.lm_head(hs_final)
            probs = torch.softmax(logits, dim=-1)
            topv, topi = torch.topk(probs, N_BRANCHES)
            forced_pairs.extend((prefix, tok) for tok in topi.tolist())
            point_order.append(cut)

        all_texts = force_all_points(forced_pairs, CONT_MAX_NEW_TOKENS)

        rescued = False
        frac_correct_num, frac_correct_den = 0, 0
        t_idx = 0
        for cut in point_order:
            branch_texts = all_texts[t_idx: t_idx + N_BRANCHES]
            t_idx += N_BRANCHES
            for t in branch_texts:
                p = extract_answer(t)
                frac_correct_den += 1
                if p is not None and ref is not None and abs(p - ref) < 1e-4:
                    frac_correct_num += 1
                    rescued = True

        record = {
            "math_test_index": idx,
            "trace_len": n_pos,
            "temperature": TEMPERATURE,
            "reference_answer": ref,
            "points": points,
            "rescued": rescued,
            "frac_correct": frac_correct_num / frac_correct_den,
        }
        f.write(json.dumps(record) + "\n")
        f.flush()

        elapsed = time.time() - t_start
        if n_done % 5 == 0:
            rate = elapsed / n_done
            eta = (N_PROBLEMS - n_done - n_skipped_correct_greedy - n_skipped_too_short) * rate
            print(f"[T={TEMPERATURE}] {n_done} scored (skipped {n_skipped_correct_greedy} already-correct, "
                  f"{n_skipped_too_short} too-short), "
                  f"elapsed={elapsed/60:.1f}min, eta={eta/60:.1f}min", flush=True)

print(f"\n[T={TEMPERATURE}] Done. {n_done} problems scored, {n_skipped_correct_greedy} skipped (greedy already correct), "
      f"{n_skipped_too_short} skipped (trace too short). Results in {OUT_PATH}")
