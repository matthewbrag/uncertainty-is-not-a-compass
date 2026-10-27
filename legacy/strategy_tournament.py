"""
Tests whether the linear entropy probe is actually USEFUL for guiding a
pruned tree search, as opposed to just being a good regressor of outcome-
entropy (everything tested on it so far). Compares 4 candidate-point
selection strategies head to head, same budget each (K points x N_BRANCHES
forced continuations), on GSM8K TEST (held out from probe training):

  - probe        top-K positions by linear-probe-predicted outcome-entropy
  - raw_entropy  top-K positions by per-layer logit-lens entropy (the
                 existing "entropy" selection method from overnight_collect)
  - random       uniformly random positions (existing control)
  - uniform      evenly-spaced positions (new, cheap deterministic baseline)

Metric is deliberately NOT "did the selected points have high outcome
entropy" - that's circular, since each strategy is partly defined by trying
to be high on some notion of entropy. Instead: for problems where the
greedy/default single-path trace gets the WRONG answer, does branching at
this strategy's chosen points recover a correct answer among the forced
branches? That's the actual tree-search payoff question.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python strategy_tournament.py <n_problems>
"""
import json
import sys
import time

import torch
from datasets import load_dataset
from transformers import LogitsProcessorList

from forkprobe.hf import (
    get_model, DEFAULT_TEMPLATE, extract_answer, compute_entropy,
    ForceEosAfterAnswer, ForceEosOnRepetition, DEVICE,
    predict_outcome_entropy, PROBE_LAYER,
)

MODEL_KEY = "qwen3-0.6b"
LAYERS = [6, 10, 14, 18, 22, 26, 28]
K_POINTS = 5
N_BRANCHES = 5
GEN_MAX_NEW_TOKENS = 400
CONT_MAX_NEW_TOKENS = 250
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)
N_PROBLEMS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
OUT_PATH = "strategy_tournament_results.jsonl"

gsm8k_test = load_dataset("openai/gsm8k", "main")["test"]
tokenizer, model = get_model(MODEL_KEY)
tokenizer.padding_side = "left"


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


def generate_once_sampled(prompt_ids, max_new_tokens):
    input_ids = torch.tensor([prompt_ids], device=DEVICE)
    prompt_len = input_ids.shape[1]
    proc_a = ForceEosAfterAnswer(tokenizer, prompt_len, tokenizer.eos_token_id, False)
    proc_r = ForceEosOnRepetition(prompt_len, tokenizer.eos_token_id)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True, **SAMPLE_KW,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList([proc_a, proc_r]),
        )
    row = out[0, prompt_len:].tolist()
    if tokenizer.eos_token_id in row:
        row = row[: row.index(tokenizer.eos_token_id) + 1]
    return row


MAX_BRANCH_BATCH = 40  # bounds memory regardless of GPU/strategy-overlap variance


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
problem_indices = all_indices[:N_PROBLEMS]

n_done = 0
n_skipped_correct_greedy = 0
n_skipped_too_short = 0
strategy_names = ["probe", "raw_entropy", "random", "uniform"]
overlap_counter = {"total_pairs": 0}
t_start = time.time()

with open(OUT_PATH, "w") as f:
    for idx in problem_indices:
        ex = gsm8k_test[idx]
        prompt_ids = build_prompt_ids(ex["question"])
        gen_ids = generate_once_sampled(prompt_ids, GEN_MAX_NEW_TOKENS)
        if len(gen_ids) < K_POINTS + 2:
            n_skipped_too_short += 1
            continue

        ref = extract_answer(ex["answer"])
        greedy_answer = extract_answer(tokenizer.decode(gen_ids, skip_special_tokens=True))
        greedy_correct = greedy_answer is not None and ref is not None and abs(greedy_answer - ref) < 1e-4
        if greedy_correct:
            # tree search only matters when the default path is already wrong -
            # nothing to "rescue" otherwise, and including these would dilute
            # the rescue-rate signal with problems where every strategy "wins"
            # trivially by not mattering
            n_skipped_correct_greedy += 1
            continue
        n_done += 1

        full_ids = prompt_ids + gen_ids
        with torch.no_grad():
            out = model(input_ids=torch.tensor([full_ids], device=DEVICE), output_hidden_states=True)
        base = len(prompt_ids)
        n_pos = len(gen_ids)

        # per-position signals, computed once per problem regardless of how
        # many strategies end up using them
        raw_entropy_trace = {L: [] for L in LAYERS}
        probe_trace = []
        for i in range(n_pos):
            pos = base + i - 1
            for L in LAYERS:
                hs = out.hidden_states[L][0, pos]
                raw_entropy_trace[L].append(compute_entropy(layer_logits(hs, L)))
            probe_hs = out.hidden_states[PROBE_LAYER][0, pos]
            probe_trace.append(predict_outcome_entropy(probe_hs))

        # strategy: probe - top-K by predicted outcome-entropy
        probe_points = sorted(range(n_pos), key=lambda i: -probe_trace[i])[:K_POINTS]

        # strategy: raw_entropy - same method overnight_collect.py used for
        # its "entropy" candidates (top-3 per layer, union, then take K)
        candidate_set = set()
        for L in LAYERS:
            trace = raw_entropy_trace[L]
            ranked = sorted(range(n_pos), key=lambda i: -trace[i])
            candidate_set.update(ranked[:3])
        raw_entropy_points = list(candidate_set)
        random.shuffle(raw_entropy_points)
        raw_entropy_points = raw_entropy_points[:K_POINTS]

        # strategy: random
        random_points = list(range(n_pos))
        random.shuffle(random_points)
        random_points = random_points[:K_POINTS]

        # strategy: uniform - evenly spaced
        if n_pos <= K_POINTS:
            uniform_points = list(range(n_pos))
        else:
            uniform_points = sorted(set(int(i * (n_pos - 1) / (K_POINTS - 1)) for i in range(K_POINTS)))

        strategies = {
            "probe": probe_points,
            "raw_entropy": raw_entropy_points,
            "random": random_points,
            "uniform": uniform_points,
        }

        # dedupe forced-branch compute: build the union of all (point) across
        # strategies, force branches once per unique point, then attribute
        # results back to whichever strategy(ies) picked that point
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

        point_correct = {}  # cut -> list of bool, one per forced branch
        t_idx = 0
        for cut in point_order:
            branch_texts = all_texts[t_idx: t_idx + N_BRANCHES]
            t_idx += N_BRANCHES
            corrects = []
            for t in branch_texts:
                p = extract_answer(t)
                corrects.append(p is not None and ref is not None and abs(p - ref) < 1e-4)
            point_correct[cut] = corrects

        # per-strategy outcome: rescued if ANY forced branch at ANY of its
        # chosen points is correct
        strategy_results = {}
        for name, pts in strategies.items():
            any_correct = any(any(point_correct[p]) for p in pts)
            frac_correct = sum(sum(point_correct[p]) for p in pts) / (len(pts) * N_BRANCHES)
            strategy_results[name] = {"rescued": any_correct, "frac_correct": frac_correct, "points": pts}

        # pairwise point-overlap bookkeeping (are strategies even picking
        # different points, or mostly agreeing?)
        for i, a in enumerate(strategy_names):
            for b in strategy_names[i + 1:]:
                key = f"{a}_vs_{b}"
                overlap_counter.setdefault(key, []).append(
                    len(set(strategies[a]) & set(strategies[b])) / K_POINTS
                )

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
            print(f"{n_done} scored (skipped {n_skipped_correct_greedy} already-correct, "
                  f"{n_skipped_too_short} too-short), elapsed={elapsed/60:.1f}min, "
                  f"eta={eta/60:.1f}min", flush=True)

print(f"\nDone. {n_done} problems scored, {n_skipped_correct_greedy} skipped (greedy already correct), "
      f"{n_skipped_too_short} skipped (trace too short). Results in {OUT_PATH}")
