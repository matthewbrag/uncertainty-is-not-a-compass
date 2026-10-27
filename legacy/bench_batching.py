"""
Benchmarks how much batching the initial trace-generation step (currently
batch=1 per problem in overnight_collect.py's generate_once_sampled) speeds
things up, plus how much further batching the branch-forcing step across
MULTIPLE candidate points at once (currently batch=N_BRANCHES=5 per single
candidate, one candidate at a time) adds on top.

Uses real GSM8K train questions and the exact same template/sampling params
as overnight_collect.py, so numbers are directly comparable to the ~7.25hr/
2500-problem overnight run.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python bench_batching.py <model_key>
"""
import sys
import time

import torch
from datasets import load_dataset
from transformers import LogitsProcessorList

from forkprobe.hf import (
    get_model, DEFAULT_TEMPLATE,
    ForceEosAfterAnswer, ForceEosOnRepetition, DEVICE,
)

MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else "qwen3-0.6b"
GEN_MAX_NEW_TOKENS = 400
N_PROBLEMS = 128  # total problems processed per trial, held fixed across batch sizes
BATCH_SIZES = [64, 96, 128]

gsm8k_train = load_dataset("openai/gsm8k", "main")["train"]
tokenizer, model = get_model(MODEL_KEY)
tokenizer.padding_side = "left"

questions = [gsm8k_train[i]["question"] for i in range(N_PROBLEMS)]


def build_text(question):
    content = DEFAULT_TEMPLATE.replace("{question}", question)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def run_batch(texts):
    enc = tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    prompt_len = enc["input_ids"].shape[1]
    proc_a = ForceEosAfterAnswer(tokenizer, prompt_len, tokenizer.eos_token_id, False)
    proc_r = ForceEosOnRepetition(prompt_len, tokenizer.eos_token_id)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=True,
            temperature=0.7, top_k=20, top_p=0.8,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList([proc_a, proc_r]),
        )
    n_new_tokens = 0
    for row in out[:, prompt_len:]:
        row = row.tolist()
        if tokenizer.eos_token_id in row:
            row = row[: row.index(tokenizer.eos_token_id) + 1]
        n_new_tokens += len(row)
    return n_new_tokens


print(f"model={MODEL_KEY} device={DEVICE} n_problems={N_PROBLEMS}\n", flush=True)
texts = [build_text(q) for q in questions]

baseline_elapsed = None
for bs in BATCH_SIZES:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    total_tokens = 0
    for start in range(0, N_PROBLEMS, bs):
        batch_texts = texts[start:start + bs]
        total_tokens += run_batch(batch_texts)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    if baseline_elapsed is None:
        baseline_elapsed = elapsed
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"batch_size={bs:>3}  wall={elapsed:7.2f}s  "
        f"problems/sec={N_PROBLEMS/elapsed:6.3f}  "
        f"tokens/sec={total_tokens/elapsed:7.1f}  "
        f"peak_mem={peak_mem_gb:5.2f}GB  "
        f"speedup_vs_bs1={baseline_elapsed/elapsed:5.2f}x",
        flush=True,
    )
