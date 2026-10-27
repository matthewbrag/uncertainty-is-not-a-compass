"""
Short pilot: does entropy at some intermediate layer (via logit lens) predict
true outcome-entropy (ground truth from resampling) better than final-layer
entropy? Sweeps a spread of layers rather than committing to layer 14 alone.

For each of a small number of GSM8K traces:
  1. Generate once (non-thinking + step template, proven settings).
  2. One extra forward pass with output_hidden_states=True to get an entropy
     trace at each layer in LAYERS, for every generated token position.
  3. Pool the top-3-per-layer highest-entropy positions across all layers,
     dedupe, cap to a handful as candidate branch points.
  4. At each candidate, resample N times and compute ground-truth
     outcome-entropy (binary entropy of the resample success rate).
  5. Write (layer entropies, outcome entropy) pairs to disk, print a quick
     per-layer Pearson correlation as a first signal check.
"""
import json
import math
import random
import sys

import torch
from transformers import LogitsProcessorList

from forkprobe.hf import (
    get_model, gsm8k_test, DEFAULT_TEMPLATE, extract_answer,
    ForceEosAfterAnswer, ForceEosOnRepetition, compute_entropy, DEVICE,
)

MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else "qwen3-0.6b"
N_PROBLEMS = int(sys.argv[2]) if len(sys.argv) > 2 else 15
N_SAMPLES_PER_POINT = int(sys.argv[3]) if len(sys.argv) > 3 else 5
LAYERS = [6, 10, 14, 18, 22, 26, 28]  # 28 == final (post-norm already, handled specially)
N_CANDIDATES_PER_TRACE = 5
GEN_MAX_NEW_TOKENS = 400
CONT_MAX_NEW_TOKENS = 250
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)
OUT_PATH = f"layer_entropy_pilot_{MODEL_KEY}_n{N_SAMPLES_PER_POINT}.jsonl"

tokenizer, model = get_model(MODEL_KEY)
tokenizer.padding_side = "left"


def layer_entropy(hidden_state_vec, layer_idx):
    if layer_idx == 28:
        logits = model.lm_head(hidden_state_vec)
    else:
        logits = model.lm_head(model.model.norm(hidden_state_vec))
    return compute_entropy(logits)


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


random.seed(0)
indices = random.sample(range(len(gsm8k_test)), N_PROBLEMS * 3)

results = []
n_done = 0
for idx in indices:
    if n_done >= N_PROBLEMS:
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

    layer_traces = {L: [] for L in LAYERS}
    for i in range(len(gen_ids)):
        pos = base + i - 1
        for L in LAYERS:
            hs = out.hidden_states[L][0, pos]
            layer_traces[L].append(layer_entropy(hs, L))

    candidate_set = set()
    for L in LAYERS:
        trace = layer_traces[L]
        ranked = sorted(range(len(trace)), key=lambda i: -trace[i])
        candidate_set.update(ranked[:3])
    candidates = list(candidate_set)
    random.shuffle(candidates)
    candidates = candidates[:N_CANDIDATES_PER_TRACE]

    for cut in candidates:
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
            "gsm8k_index": idx,
            "cut_token": cut,
            "trace_len": len(gen_ids),
            "layer_entropies": {str(L): layer_traces[L][cut] for L in LAYERS},
            "n_correct": n_correct,
            "outcome_p": outcome_p,
            "outcome_entropy": oe,
            "answers": answers,
            "reference_answer": ref,
        }
        results.append(record)
        print(
            f"gsm8k[{idx}] cut={cut}/{len(gen_ids)} outcome_entropy={oe:.3f} (p={outcome_p:.2f}) "
            f"L14={layer_traces[14][cut]:.2f} L28={layer_traces[28][cut]:.2f}",
            flush=True,
        )

with open(OUT_PATH, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print(f"\nDone. {len(results)} data points written to {OUT_PATH}")

if len(results) >= 5:
    import statistics

    outcome_ents = [r["outcome_entropy"] for r in results]
    oe_mean = statistics.mean(outcome_ents)
    oe_std = statistics.pstdev(outcome_ents)
    print("\nPer-layer Pearson correlation with outcome entropy:")
    for L in LAYERS:
        vals = [r["layer_entropies"][str(L)] for r in results]
        v_mean = statistics.mean(vals)
        v_std = statistics.pstdev(vals)
        if v_std > 0 and oe_std > 0:
            cov = sum((v - v_mean) * (o - oe_mean) for v, o in zip(vals, outcome_ents)) / len(vals)
            r_corr = cov / (v_std * oe_std)
            print(f"  layer {L:2d}: r = {r_corr:+.3f}")
        else:
            print(f"  layer {L:2d}: no variance, skipping")
