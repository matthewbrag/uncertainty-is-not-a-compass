"""
Overnight data collection for training an outcome-entropy probe. Combines
everything validated tonight:
  - non-thinking + explicit step template (near-zero truncation, no loops)
  - forced top-N token branching for outcome measurement, not random
    resampling (Entropy-Tree/EDEN's actual method - produces more genuine
    uncertainty and correctly-signed layer-14 correlation vs resampling)
  - full hidden-state vectors saved at a spread of layers (not reduced to a
    scalar) so a probe can use the full representation
  - full prompt/gen token IDs saved (positions stay inspectable, features
    are recoverable later without needing to regenerate)

Pulls from the GSM8K TRAIN split specifically - the test split has been used
for every eval/triage/pilot tonight, so training a probe on test data would
quietly contaminate future evaluation of that probe.

Usage: python overnight_collect.py <shard_id> <num_shards> <max_problems_per_shard>
"""
import json
import math
import random
import sys
import time

import torch
from datasets import load_dataset
from transformers import LogitsProcessorList

from app import (
    get_model, DEFAULT_TEMPLATE, extract_answer,
    ForceEosAfterAnswer, ForceEosOnRepetition, compute_entropy, DEVICE,
)

SHARD_ID = int(sys.argv[1])
NUM_SHARDS = int(sys.argv[2])
MAX_PROBLEMS_PER_SHARD = int(sys.argv[3])

MODEL_KEY = "qwen3-0.6b"
LAYERS = [6, 10, 14, 18, 22, 26, 28]
N_CANDIDATES_PER_TRACE = 5
N_RANDOM_CANDIDATES_PER_TRACE = 2
N_BRANCHES = 5
GEN_MAX_NEW_TOKENS = 400
CONT_MAX_NEW_TOKENS = 250
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)
OUT_PATH = f"overnight_shard{SHARD_ID}.jsonl"

gsm8k_train = load_dataset("openai/gsm8k", "main")["train"]
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


def greedy_branches(prefix_ids, forced_first_tokens, max_new_tokens):
    batch_ids = [prefix_ids + [tok] for tok in forced_first_tokens]
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
    for j in range(len(forced_first_tokens)):
        row = out[j, prompt_len:].tolist()
        if tokenizer.eos_token_id in row:
            row = row[: row.index(tokenizer.eos_token_id) + 1]
        full_row = [forced_first_tokens[j]] + row
        texts.append(tokenizer.decode(full_row, skip_special_tokens=True))
    return texts


def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


random.seed(3000 + SHARD_ID)
all_indices = list(range(len(gsm8k_train)))
shard_indices = [i for i in all_indices if i % NUM_SHARDS == SHARD_ID]
random.shuffle(shard_indices)

n_done = 0
n_points = 0
t_start = time.time()

with open(OUT_PATH, "w") as f:
    for idx in shard_indices:
        if n_done >= MAX_PROBLEMS_PER_SHARD:
            break
        ex = gsm8k_train[idx]
        prompt_ids = build_prompt_ids(ex["question"])
        gen_ids = generate_once_sampled(prompt_ids, GEN_MAX_NEW_TOKENS)
        if len(gen_ids) < 20:
            continue
        n_done += 1

        ref = extract_answer(ex["answer"])
        full_ids = prompt_ids + gen_ids
        with torch.no_grad():
            out = model(input_ids=torch.tensor([full_ids], device=DEVICE), output_hidden_states=True)
        base = len(prompt_ids)

        layer_traces = {L: [] for L in LAYERS}
        for i in range(len(gen_ids)):
            pos = base + i - 1
            for L in LAYERS:
                hs = out.hidden_states[L][0, pos]
                layer_traces[L].append(compute_entropy(layer_logits(hs, L)))

        candidate_set = set()
        for L in LAYERS:
            trace = layer_traces[L]
            ranked = sorted(range(len(trace)), key=lambda i: -trace[i])
            candidate_set.update(ranked[:3])
        entropy_candidates = list(candidate_set)
        random.shuffle(entropy_candidates)
        entropy_candidates = entropy_candidates[:N_CANDIDATES_PER_TRACE]

        # random-baseline control: uniformly sampled positions, excluded from the
        # entropy set where possible, so the two groups stay comparable later
        all_positions = set(range(len(gen_ids)))
        remaining = list(all_positions - set(entropy_candidates))
        random.shuffle(remaining)
        random_candidates = remaining[:N_RANDOM_CANDIDATES_PER_TRACE]

        tagged_candidates = [(c, "entropy") for c in entropy_candidates] + \
                             [(c, "random") for c in random_candidates]

        for cut, selection_method in tagged_candidates:
            pos = base + cut - 1
            hs_final = out.hidden_states[28][0, pos]
            logits = model.lm_head(hs_final)
            probs = torch.softmax(logits, dim=-1)
            topv, topi = torch.topk(probs, N_BRANCHES)
            forced_tokens = topi.tolist()
            forced_probs = topv.tolist()

            prefix = prompt_ids + gen_ids[: cut + 1]
            texts = greedy_branches(prefix, forced_tokens, CONT_MAX_NEW_TOKENS)

            n_correct = 0
            answers = []
            for t in texts:
                p = extract_answer(t)
                answers.append(p)
                if p is not None and ref is not None and abs(p - ref) < 1e-4:
                    n_correct += 1
            outcome_p = n_correct / N_BRANCHES
            oe = binary_entropy(outcome_p)

            record = {
                "shard": SHARD_ID,
                "gsm8k_train_index": idx,
                "cut_token": cut,
                "selection_method": selection_method,
                "trace_len": len(gen_ids),
                "layer_entropies": {str(L): layer_traces[L][cut] for L in LAYERS},
                "hidden_states": {str(L): out.hidden_states[L][0, pos].tolist() for L in LAYERS},
                "forced_token_probs": forced_probs,
                "forced_token_texts": [tokenizer.decode([t]) for t in forced_tokens],
                "n_correct": n_correct,
                "outcome_p": outcome_p,
                "outcome_entropy": oe,
                "answers": answers,
                "reference_answer": ref,
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": gen_ids,
            }
            f.write(json.dumps(record) + "\n")
            n_points += 1
        f.flush()

        elapsed = time.time() - t_start
        if n_done % 5 == 0 or n_done == MAX_PROBLEMS_PER_SHARD:
            rate = elapsed / n_done
            eta = (MAX_PROBLEMS_PER_SHARD - n_done) * rate
            print(f"[shard {SHARD_ID}] {n_done}/{MAX_PROBLEMS_PER_SHARD} -> {n_points} points, "
                  f"elapsed={elapsed/60:.1f}min, eta={eta/60:.1f}min", flush=True)

print(f"\n[shard {SHARD_ID}] Done. {n_points} data points written to {OUT_PATH}")
