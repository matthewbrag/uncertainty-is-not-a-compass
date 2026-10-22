"""
The initial trace-generation step turned out to be only ~20% of real
per-problem wall-clock in the overnight run (2500 problems / 7.25hr =
10.44s/problem vs 2.06s/problem measured for generation alone). The other
~80% is the branch-forcing step: 7 candidate points per problem, each
currently its own batch-5 generate() call (one per candidate, sequential).

This benchmarks batching ALL candidates' forced branches into a single
generate() call (batch = n_candidates * n_branches) against the current
sequential-per-candidate loop, using real traces and real top-5 forced
tokens (not synthetic placeholders).

Usage: CUDA_VISIBLE_DEVICES=<gpu> python bench_branch_batching.py <model_key>
"""
import sys
import time

import torch
from datasets import load_dataset
from transformers import LogitsProcessorList

from app import (
    get_model, DEFAULT_TEMPLATE,
    ForceEosAfterAnswer, ForceEosOnRepetition, DEVICE,
)

MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else "qwen3-0.6b"
N_PROBLEMS = 16
N_CANDIDATES = 7
N_BRANCHES = 5
CONT_MAX_NEW_TOKENS = 250
GEN_MAX_NEW_TOKENS = 400

gsm8k_train = load_dataset("openai/gsm8k", "main")["train"]
tokenizer, model = get_model(MODEL_KEY)
tokenizer.padding_side = "left"


def build_prompt_ids(question):
    content = DEFAULT_TEMPLATE.replace("{question}", question)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def generate_trace(prompt_ids):
    input_ids = torch.tensor([prompt_ids], device=DEVICE)
    prompt_len = input_ids.shape[1]
    proc_a = ForceEosAfterAnswer(tokenizer, prompt_len, tokenizer.eos_token_id, False)
    proc_r = ForceEosOnRepetition(prompt_len, tokenizer.eos_token_id)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=True,
            temperature=0.7, top_k=20, top_p=0.8,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList([proc_a, proc_r]),
        )
    row = out[0, prompt_len:].tolist()
    if tokenizer.eos_token_id in row:
        row = row[: row.index(tokenizer.eos_token_id) + 1]
    return row


def run_forced_batch(prefixes_and_tokens, max_new_tokens):
    """prefixes_and_tokens: list of (prefix_ids, forced_first_token)."""
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
    return out.shape[0]


print(f"model={MODEL_KEY} device={DEVICE} n_problems={N_PROBLEMS}\n", flush=True)

# build realistic (prefix, forced_token) pairs for N_CANDIDATES per problem,
# exactly like overnight_collect.py: real trace, real top-5 logits at real cut points
all_candidates_per_problem = []
for i in range(N_PROBLEMS):
    ex = gsm8k_train[i]
    prompt_ids = build_prompt_ids(ex["question"])
    gen_ids = generate_trace(prompt_ids)
    if len(gen_ids) < N_CANDIDATES + 2:
        continue
    full_ids = prompt_ids + gen_ids
    with torch.no_grad():
        out = model(input_ids=torch.tensor([full_ids], device=DEVICE))
    base = len(prompt_ids)
    cuts = [int(c) for c in torch.linspace(0, len(gen_ids) - 2, N_CANDIDATES)]
    candidates = []
    for cut in cuts:
        pos = base + cut - 1
        logits = out.logits[0, pos]
        probs = torch.softmax(logits, dim=-1)
        topv, topi = torch.topk(probs, N_BRANCHES)
        prefix = prompt_ids + gen_ids[: cut + 1]
        for tok in topi.tolist():
            candidates.append((prefix, tok))
    all_candidates_per_problem.append(candidates)

print(f"built {len(all_candidates_per_problem)} problems x {N_CANDIDATES * N_BRANCHES} forced branches\n", flush=True)

# approach A: current behavior - sequential, one candidate at a time (batch=N_BRANCHES)
torch.cuda.synchronize()
t0 = time.time()
for candidates in all_candidates_per_problem:
    for c_start in range(0, len(candidates), N_BRANCHES):
        run_forced_batch(candidates[c_start:c_start + N_BRANCHES], CONT_MAX_NEW_TOKENS)
torch.cuda.synchronize()
elapsed_sequential = time.time() - t0
print(f"[A] sequential (7x batch-5 per problem): {elapsed_sequential:.2f}s total, "
      f"{elapsed_sequential/len(all_candidates_per_problem):.2f}s/problem", flush=True)

# approach B: batch all candidates for a problem together (batch=N_CANDIDATES*N_BRANCHES=35)
torch.cuda.synchronize()
t0 = time.time()
for candidates in all_candidates_per_problem:
    run_forced_batch(candidates, CONT_MAX_NEW_TOKENS)
torch.cuda.synchronize()
elapsed_batched = time.time() - t0
print(f"[B] batched (1x batch-{N_CANDIDATES * N_BRANCHES} per problem): {elapsed_batched:.2f}s total, "
      f"{elapsed_batched/len(all_candidates_per_problem):.2f}s/problem", flush=True)

print(f"\nspeedup from batching branch-forcing across candidates: {elapsed_sequential/elapsed_batched:.2f}x", flush=True)
