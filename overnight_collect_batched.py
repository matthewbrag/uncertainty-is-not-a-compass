"""
Batched version of overnight_collect.py - same methodology, same recorded
fields, same selection logic (forced top-N branching + entropy/random
candidate selection), but restructured to use the two batching wins measured
today:
  - initial trace generation batched across GEN_BATCH_SIZE problems at once
    (was batch=1 per problem) -> ~8-10x on that step alone
  - branch-forcing batched across ALL candidates for a problem at once
    (was 7 sequential batch-5 calls) -> ~1.5-1.8x on that step

Parameterized by model so this doubles as the generalization check: does the
linear-probe signal (the thing that mattered all night, not the heuristics
that failed) replicate on a different model size?

Usage: python overnight_collect_batched.py <model_key> <shard_id> <num_shards> <max_problems_per_shard>
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

MODEL_KEY = sys.argv[1]
SHARD_ID = int(sys.argv[2])
NUM_SHARDS = int(sys.argv[3])
MAX_PROBLEMS_PER_SHARD = int(sys.argv[4])

LAYERS = [6, 10, 14, 18, 22, 26, 28]
N_CANDIDATES_PER_TRACE = 5
N_RANDOM_CANDIDATES_PER_TRACE = 2
N_BRANCHES = 5
GEN_MAX_NEW_TOKENS = 400
CONT_MAX_NEW_TOKENS = 250
GEN_BATCH_SIZE = 64  # sweet spot measured for both 0.6b and 1.7b today
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)
OUT_PATH = f"overnight_{MODEL_KEY}_shard{SHARD_ID}.jsonl"

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


def generate_batch_sampled(prompt_ids_list, max_new_tokens):
    """Batched version of overnight_collect.py's generate_once_sampled."""
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


def greedy_branches_all_candidates(prefixes_and_tokens, max_new_tokens):
    """Batches ALL (prefix, forced_first_token) pairs for a problem into one
    generate() call, instead of looping per-candidate at batch=N_BRANCHES."""
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
        full_row = [tok] + row
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
shard_indices = shard_indices[:MAX_PROBLEMS_PER_SHARD]

n_done = 0
n_points = 0
t_start = time.time()

with open(OUT_PATH, "w") as f:
    for gbatch_start in range(0, len(shard_indices), GEN_BATCH_SIZE):
        gbatch_indices = shard_indices[gbatch_start:gbatch_start + GEN_BATCH_SIZE]
        prompt_ids_batch = [build_prompt_ids(gsm8k_train[idx]["question"]) for idx in gbatch_indices]
        gen_ids_batch = generate_batch_sampled(prompt_ids_batch, GEN_MAX_NEW_TOKENS)

        for idx, prompt_ids, gen_ids in zip(gbatch_indices, prompt_ids_batch, gen_ids_batch):
            if len(gen_ids) < 20:
                continue
            n_done += 1
            ex = gsm8k_train[idx]
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

            all_positions = set(range(len(gen_ids)))
            remaining = list(all_positions - set(entropy_candidates))
            random.shuffle(remaining)
            random_candidates = remaining[:N_RANDOM_CANDIDATES_PER_TRACE]

            tagged_candidates = [(c, "entropy") for c in entropy_candidates] + \
                                 [(c, "random") for c in random_candidates]

            # build the full set of forced (prefix, token) pairs for ALL
            # candidates in this problem, then force them in ONE batched call
            per_candidate_meta = []
            forced_pairs = []
            for cut, selection_method in tagged_candidates:
                pos = base + cut - 1
                hs_final = out.hidden_states[28][0, pos]
                logits = model.lm_head(hs_final)
                probs = torch.softmax(logits, dim=-1)
                topv, topi = torch.topk(probs, N_BRANCHES)
                forced_tokens = topi.tolist()
                forced_probs = topv.tolist()
                prefix = prompt_ids + gen_ids[: cut + 1]
                per_candidate_meta.append({
                    "cut": cut, "selection_method": selection_method, "pos": pos,
                    "forced_tokens": forced_tokens, "forced_probs": forced_probs,
                })
                forced_pairs.extend((prefix, tok) for tok in forced_tokens)

            all_texts = greedy_branches_all_candidates(forced_pairs, CONT_MAX_NEW_TOKENS)

            t_idx = 0
            for meta in per_candidate_meta:
                texts = all_texts[t_idx: t_idx + N_BRANCHES]
                t_idx += N_BRANCHES

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
                    "model": MODEL_KEY,
                    "gsm8k_train_index": idx,
                    "cut_token": meta["cut"],
                    "selection_method": meta["selection_method"],
                    "trace_len": len(gen_ids),
                    "layer_entropies": {str(L): layer_traces[L][meta["cut"]] for L in LAYERS},
                    "hidden_states": {str(L): out.hidden_states[L][0, meta["pos"]].tolist() for L in LAYERS},
                    "forced_token_probs": meta["forced_probs"],
                    "forced_token_texts": [tokenizer.decode([t]) for t in meta["forced_tokens"]],
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
        rate = elapsed / n_done if n_done else 0
        eta = (MAX_PROBLEMS_PER_SHARD - n_done) * rate
        print(f"[{MODEL_KEY} shard {SHARD_ID}] {n_done}/{MAX_PROBLEMS_PER_SHARD} -> {n_points} points, "
              f"elapsed={elapsed/60:.1f}min, eta={eta/60:.1f}min", flush=True)

print(f"\n[{MODEL_KEY} shard {SHARD_ID}] Done. {n_points} data points written to {OUT_PATH}")
