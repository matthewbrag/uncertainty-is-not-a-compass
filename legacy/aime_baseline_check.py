"""
Quick baseline-accuracy check: does 1.7B's success rate drop meaningfully
below 0.5 on a harder, less-benchmaxed problem set (AIME, 2021+)? This is
step 1 before investing in a full tournament on a new distribution - if
baseline p doesn't actually drop, there's no point testing the mechanism
here. No forced branching, no hidden states, no probes - just batched
generation + answer extraction, to keep this cheap and fast.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python aime_baseline_check.py [n_problems]
"""
import sys

import torch
from datasets import load_dataset
from transformers import LogitsProcessorList

from app import (
    get_model, DEFAULT_TEMPLATE, extract_answer,
    ForceEosAfterAnswer, ForceEosOnRepetition, DEVICE,
)

MODEL_KEY = "qwen3-1.7b"
GEN_MAX_NEW_TOKENS = 500  # AIME problems need more room than GSM8K
BATCH_SIZE = 32
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)
YEAR_CUTOFF = 2021

def clean_answer(a):
    try:
        return float(a)
    except (TypeError, ValueError):
        return None


ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
problems = [r for r in ds if r["Year"] >= YEAR_CUTOFF and clean_answer(r["Answer"]) is not None]
n_skipped = sum(1 for r in ds if r["Year"] >= YEAR_CUTOFF) - len(problems)
print(f"{len(problems)} AIME problems from {YEAR_CUTOFF}+ ({n_skipped} skipped, malformed answer)", flush=True)

n_problems = int(sys.argv[1]) if len(sys.argv) > 1 else len(problems)
problems = problems[:n_problems]

tokenizer, model = get_model(MODEL_KEY)
tokenizer.padding_side = "left"


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
        results.append(tokenizer.decode(row, skip_special_tokens=True))
    return results


n_correct = 0
n_total = 0
n_unparsed = 0
for start in range(0, len(problems), BATCH_SIZE):
    batch = problems[start:start + BATCH_SIZE]
    prompt_ids_batch = [build_prompt_ids(r["Question"]) for r in batch]
    texts = generate_batch_sampled(prompt_ids_batch, GEN_MAX_NEW_TOKENS)
    for r, text in zip(batch, texts):
        ref = clean_answer(r["Answer"])
        pred = extract_answer(text)
        n_total += 1
        if pred is None:
            n_unparsed += 1
        elif abs(pred - ref) < 1e-4:
            n_correct += 1
    print(f"{n_total}/{len(problems)} done, running accuracy={n_correct/n_total:.3f}, "
          f"unparsed={n_unparsed}", flush=True)

print(f"\nFinal: {n_correct}/{n_total} correct = {n_correct/n_total:.3f}  "
      f"({n_unparsed} unparsed answers)")
