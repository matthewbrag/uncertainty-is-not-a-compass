"""
v2: tests entropy-selected vs attention-importance-selected candidate points
head-to-head against the same outcome-entropy ground truth. Saves full
prompt_ids/gen_ids per trace (fixes v1's gap - points are now inspectable
and other features are recoverable later without needing to regenerate).

Attention importance follows DRAGIN/Entropy-Tree: I_t = max_{j<t} A_{t,j},
using the last layer's attention, averaged across heads. Requires eager
attention (SDPA doesn't expose attention weights) - toggled on only for the
single extraction forward pass per trace, then back to sdpa for the (much
more frequent) generation calls.
"""
import json
import math
import random
import sys
import time

import torch
from transformers import LogitsProcessorList

from app import (
    get_model, gsm8k_test, DEFAULT_TEMPLATE, extract_answer,
    ForceEosAfterAnswer, ForceEosOnRepetition, compute_entropy, DEVICE,
)

SHARD_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NUM_SHARDS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
PROBLEMS_PER_SHARD = int(sys.argv[3]) if len(sys.argv) > 3 else 20

MODEL_KEY = "qwen3-0.6b"
LAYERS = [10, 14, 20, 28]
N_CANDIDATES_PER_STRATEGY = 3  # top-3 by entropy, top-3 by importance, per trace
N_SAMPLES_PER_POINT = 5
GEN_MAX_NEW_TOKENS = 400
CONT_MAX_NEW_TOKENS = 250
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)
OUT_PATH = f"v2_importance_shard{SHARD_ID}.jsonl"

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
    model.set_attn_implementation("sdpa")
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
    model.set_attn_implementation("sdpa")
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


def extract_entropy_and_importance_traces(full_ids, base, n_gen):
    """One eager-mode forward pass: entropy trace per layer + attention-importance trace."""
    model.set_attn_implementation("eager")
    with torch.no_grad():
        out = model(
            input_ids=torch.tensor([full_ids], device=DEVICE),
            output_hidden_states=True, output_attentions=True,
        )
    model.set_attn_implementation("sdpa")

    last_layer_attn = out.attentions[-1][0].mean(dim=0)  # (seq, seq), averaged over heads

    layer_traces = {L: [] for L in LAYERS}
    importance_trace = []
    hidden_state_cache = []
    for i in range(n_gen):
        pos = base + i - 1
        hs_at_pos = {}
        for L in LAYERS:
            hs = out.hidden_states[L][0, pos]
            hs_at_pos[L] = hs
            layer_traces[L].append(compute_entropy(layer_logits(hs, L)))
        hidden_state_cache.append(hs_at_pos)

        if pos > 0:
            row = last_layer_attn[pos, :pos]
            importance_trace.append(row.max().item())
        else:
            importance_trace.append(0.0)

    return layer_traces, importance_trace, hidden_state_cache


def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


random.seed(2000 + SHARD_ID)
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
        base = len(prompt_ids)
        layer_traces, importance_trace, hs_cache = extract_entropy_and_importance_traces(
            full_ids, base, len(gen_ids)
        )

        # strategy A: top-3 pooled-across-layers entropy (same as v1)
        entropy_candidates = set()
        for L in LAYERS:
            trace = layer_traces[L]
            ranked = sorted(range(len(trace)), key=lambda i: -trace[i])
            entropy_candidates.update(ranked[:2])
        entropy_candidates = set(list(entropy_candidates)[:N_CANDIDATES_PER_STRATEGY])

        # strategy B: top-3 by attention importance
        ranked_importance = sorted(range(len(importance_trace)), key=lambda i: -importance_trace[i])
        importance_candidates = set(ranked_importance[:N_CANDIDATES_PER_STRATEGY])

        all_candidates = entropy_candidates | importance_candidates

        for cut in all_candidates:
            strategy_tags = []
            if cut in entropy_candidates:
                strategy_tags.append("entropy")
            if cut in importance_candidates:
                strategy_tags.append("importance")

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
                "cut_token_text": tokenizer.decode([gen_ids[cut]]),
                "trace_len": len(gen_ids),
                "strategy_tags": strategy_tags,
                "layer_entropies": {str(L): layer_traces[L][cut] for L in LAYERS},
                "attention_importance": importance_trace[cut],
                "hidden_states": {str(L): hs_cache[cut][L].tolist() for L in LAYERS},
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
        print(f"[shard {SHARD_ID}] problem {n_done}/{PROBLEMS_PER_SHARD} (gsm8k[{idx}]) "
              f"-> {n_points} points so far, elapsed={elapsed:.0f}s", flush=True)

print(f"\n[shard {SHARD_ID}] Done. {n_points} data points written to {OUT_PATH}")
