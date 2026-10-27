"""
Overnight task: find where 1.7B's baseline success rate crosses 0.5 on the
MATH dataset's difficulty levels (1-5), to identify a problem distribution
with real headroom for the tree-search experiments (GSM8K=0.687 too easy,
AIME=0.01 too hard, per tonight's findings). Filters to problems with a
plain-numeric answer (int/decimal/simple fraction) so our existing
extract_answer works without new parsing logic.

Cheap: just batched generation + extraction, no branching, no hidden states.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python math_level_sweep.py
"""
import re

import torch
from datasets import load_dataset
from transformers import LogitsProcessorList

from forkprobe.hf import get_model, extract_answer, ForceEosAfterAnswer, ForceEosOnRepetition, DEVICE

MODEL_KEY = "qwen3-1.7b"
GEN_MAX_NEW_TOKENS = 500
BATCH_SIZE = 32
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)

TEMPLATE = (
    "{question}\n\nSolve this step by step. Write each step on its own line starting with "
    "'Step 1:', 'Step 2:', etc. If the final answer is not a whole number, write it as a "
    "decimal, not a fraction. After the final step, on its own line, write your final "
    "numeric answer in exactly this format: '#### <number>' (for example: '#### 0.5')."
)


def parse_numeric(a):
    a = a.strip()
    try:
        return float(a)
    except ValueError:
        pass
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", a)
    if m:
        sign = -1.0 if m.group(1) else 1.0
        return sign * float(m.group(2)) / float(m.group(3))
    return None


ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
by_level = {l: [] for l in range(1, 6)}
for r in ds:
    v = parse_numeric(r["answer"])
    if v is not None and r["level"] in by_level:
        by_level[r["level"]].append((r["problem"], v))

tokenizer, model = get_model(MODEL_KEY)
tokenizer.padding_side = "left"


def build_prompt_ids(question):
    content = TEMPLATE.replace("{question}", question)
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


print(f"model={MODEL_KEY}\n", flush=True)
for level in range(1, 6):
    problems = by_level[level]
    n_correct = 0
    n_total = 0
    for start in range(0, len(problems), BATCH_SIZE):
        batch = problems[start:start + BATCH_SIZE]
        prompt_ids_batch = [build_prompt_ids(q) for q, _ in batch]
        texts = generate_batch_sampled(prompt_ids_batch, GEN_MAX_NEW_TOKENS)
        for (q, ref), text in zip(batch, texts):
            pred = extract_answer(text)
            n_total += 1
            if pred is not None and abs(pred - ref) < 1e-3:
                n_correct += 1
    acc = n_correct / n_total if n_total else float("nan")
    print(f"level {level}: {n_correct}/{n_total} = {acc:.3f}", flush=True)

print("\nDone.")
