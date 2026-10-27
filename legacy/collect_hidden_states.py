"""
Scaled, shardable data collector for probe training. Unlike the pilot, this
saves FULL hidden-state vectors (not just scalar entropy) at a handful of
representative layers, alongside ground-truth outcome-entropy labels from
resampling. Meant to run as 3 parallel shards (one per GPU via
CUDA_VISIBLE_DEVICES), each writing its own output file.

Usage: python collect_hidden_states.py <shard_id> <num_shards> <problems_per_shard>
"""
import json
import math
import random
import sys
import time

import torch
from transformers import LogitsProcessorList

from forkprobe.hf import (
    get_model, gsm8k_test, DEFAULT_TEMPLATE, extract_answer,
    ForceEosAfterAnswer, ForceEosOnRepetition, compute_entropy, DEVICE,
)

SHARD_ID = int(sys.argv[1])
NUM_SHARDS = int(sys.argv[2])
PROBLEMS_PER_SHARD = int(sys.argv[3])

MODEL_KEY = "qwen3-0.6b"
LAYERS = [10, 14, 20, 28]
N_CANDIDATES_PER_TRACE = 6
N_SAMPLES_PER_POINT = 5
GEN_MAX_NEW_TOKENS = 400
CONT_MAX_NEW_TOKENS = 250
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)
OUT_PATH = f"hidden_states_shard{SHARD_ID}.jsonl"

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


def generate_once(prompt_ids, max_new_tokens):
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


def generate_continuations(prefix_ids, n_samples, max_new_tokens):
    input_ids = torch.tensor([prefix_ids] * n_samples, device=DEVICE)
    prompt_len = input_ids.shape[1]
    proc_a = ForceEosAfterAnswer(tokenizer, prompt_len, tokenizer.eos_token_id, False)
    proc_r = ForceEosOnRepetition(prompt_len, tokenizer.eos_token_id)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True, **SAMPLE_KW,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList([proc_a, proc_r]),
        )
    texts = []
    for j in range(n_samples):
        row = out[j, prompt_len:].tolist()
        if tokenizer.eos_token_id in row:
            row = row[: row.index(tokenizer.eos_token_id) + 1]
        texts.append(tokenizer.decode(row, skip_special_tokens=True))
    return texts


def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


random.seed(1000 + SHARD_ID)
all_indices = list(range(len(gsm8k_test)))
shard_indices = [i for i in all_indices if i % NUM_SHARDS == SHARD_ID]
random.shuffle(shard_indices)

n_done = 0
n_points = 0
t_start = time.time()

with open(OUT_PATH, "w") as f:
    for idx in shard_indices:
        if n_done >= PROBLEMS_PER_SHARD:
            break
        ex = gsm8k_test[idx]
        prompt_ids = build_prompt_ids(ex["question"])
        gen_ids = generate_once(prompt_ids, GEN_MAX_NEW_TOKENS)
        if len(gen_ids) < 20:
            continue
        n_done += 1

        ref = extract_answer(ex["answer"])
        full_ids = prompt_ids + gen_ids
        with torch.no_grad():
            out = model(input_ids=torch.tensor([full_ids], device=DEVICE), output_hidden_states=True)
        base = len(prompt_ids)

        # entropy trace per layer (cheap scalar, used only for candidate selection)
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
        candidates = list(candidate_set)
        random.shuffle(candidates)
        candidates = candidates[:N_CANDIDATES_PER_TRACE]

        for cut in candidates:
            pos = base + cut - 1
            # full hidden-state vectors at this position, for each layer (the actual training features)
            hidden_vecs = {str(L): out.hidden_states[L][0, pos].tolist() for L in LAYERS}
            layer_ents = {str(L): layer_traces[L][cut] for L in LAYERS}

            prefix = prompt_ids + gen_ids[: cut + 1]
            texts = generate_continuations(prefix, N_SAMPLES_PER_POINT, CONT_MAX_NEW_TOKENS)
            n_correct = 0
            answers = []
            for t in texts:
                p = extract_answer(t)
                answers.append(p)
                if p is not None and ref is not None and abs(p - ref) < 1e-4:
                    n_correct += 1
            outcome_p = n_correct / N_SAMPLES_PER_POINT
            oe = binary_entropy(outcome_p)

            record = {
                "shard": SHARD_ID,
                "gsm8k_index": idx,
                "cut_token": cut,
                "trace_len": len(gen_ids),
                "layer_entropies": layer_ents,
                "hidden_states": hidden_vecs,
                "n_correct": n_correct,
                "outcome_p": outcome_p,
                "outcome_entropy": oe,
                "answers": answers,
                "reference_answer": ref,
            }
            f.write(json.dumps(record) + "\n")
            n_points += 1
        f.flush()

        elapsed = time.time() - t_start
        print(f"[shard {SHARD_ID}] problem {n_done}/{PROBLEMS_PER_SHARD} (gsm8k[{idx}]) "
              f"-> {n_points} points so far, elapsed={elapsed:.0f}s", flush=True)

print(f"\n[shard {SHARD_ID}] Done. {n_points} data points written to {OUT_PATH}")
