"""
Runs sustained generation at a fixed batch size for a while, so a concurrent
nvidia-smi poller gets a long enough window to measure actual SM utilization
(not just wall-clock) during steady-state decoding.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python bench_util_sustained.py <model_key> <batch_size> <n_iters>
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
BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 32
N_ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
GEN_MAX_NEW_TOKENS = 400

gsm8k_train = load_dataset("openai/gsm8k", "main")["train"]
tokenizer, model = get_model(MODEL_KEY)
tokenizer.padding_side = "left"

questions = [gsm8k_train[i]["question"] for i in range(BATCH_SIZE)]


def build_text(question):
    content = DEFAULT_TEMPLATE.replace("{question}", question)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


texts = [build_text(q) for q in questions]
enc = tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
prompt_len = enc["input_ids"].shape[1]

print(f"model={MODEL_KEY} batch_size={BATCH_SIZE} n_iters={N_ITERS} -- starting, ready for profiler", flush=True)
time.sleep(2)  # give the poller a moment to attach

t0 = time.time()
for i in range(N_ITERS):
    proc_a = ForceEosAfterAnswer(tokenizer, prompt_len, tokenizer.eos_token_id, False)
    proc_r = ForceEosOnRepetition(prompt_len, tokenizer.eos_token_id)
    with torch.no_grad():
        model.generate(
            **enc, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=True,
            temperature=0.7, top_k=20, top_p=0.8,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList([proc_a, proc_r]),
        )
    print(f"iter {i+1}/{N_ITERS} done, elapsed={time.time()-t0:.1f}s", flush=True)

print("DONE", flush=True)
