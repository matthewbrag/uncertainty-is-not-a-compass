"""
Full-dataset candidate scan using the validated non-thinking + explicit
step-template scaffolding: short generations, high conclusion rate, no loops.

Classifies every GSM8K test example as one of:
  - correct
  - concluded_wrong  (genuinely concluded with '#### <number>', just wrong -
                      these are the "clean failure" candidates worth tracing)
  - looped           (caught by the repetition guard)
  - other_truncated  (ran out of budget without concluding or looping)

Writes one JSON line per example to OUT_PATH as it goes, so progress persists
and is inspectable even mid-run or after this process exits.
"""
import json
import sys
import time

import torch
from transformers import LogitsProcessorList

from forkprobe.hf import (
    get_model, gsm8k_test, DEFAULT_TEMPLATE, extract_answer,
    ForceEosAfterAnswer, ForceEosOnRepetition, DEVICE,
)

MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else "qwen3-0.6b"
OUT_PATH = f"scan_results_{MODEL_KEY}.jsonl"
BATCH_SIZE = 32
MAX_NEW_TOKENS = 400
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)  # non-thinking official settings
START = 0
END = len(gsm8k_test)

tokenizer, model = get_model(MODEL_KEY)
tokenizer.padding_side = "left"


def build_prompt_ids(question):
    content = DEFAULT_TEMPLATE.replace("{question}", question)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def pad_batch_ids(id_lists):
    max_len = max(len(ids) for ids in id_lists)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.tensor([[pad_id] * (max_len - len(ids)) + ids for ids in id_lists], device=DEVICE)
    attention_mask = torch.tensor(
        [[0] * (max_len - len(ids)) + [1] * len(ids) for ids in id_lists], device=DEVICE
    )
    return input_ids, attention_mask


def generate_batch(prompt_id_lists):
    input_ids, attention_mask = pad_batch_ids(prompt_id_lists)
    prompt_len = input_ids.shape[1]
    proc_a = ForceEosAfterAnswer(tokenizer, prompt_len, tokenizer.eos_token_id, False)
    proc_r = ForceEosOnRepetition(prompt_len, tokenizer.eos_token_id)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS, do_sample=True, **SAMPLE_KW,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList([proc_a, proc_r]),
        )
    results = []
    for j in range(len(prompt_id_lists)):
        row = out[j, prompt_len:].tolist()
        eos_hit = tokenizer.eos_token_id in row
        if eos_hit:
            row = row[: row.index(tokenizer.eos_token_id) + 1]
        text = tokenizer.decode(row, skip_special_tokens=True)
        results.append({
            "prompt_ids": prompt_id_lists[j],
            "gen_ids": row,
            "text": text,
            "answer_eos": bool(proc_a.matched[j]),
            "repetition_eos": bool(proc_r.matched[j]),
        })
    return results


counts = {"correct": 0, "concluded_wrong": 0, "looped": 0, "other_truncated": 0}
t_start = time.time()

with open(OUT_PATH, "w") as f:
    for batch_start in range(START, END, BATCH_SIZE):
        batch_idx = list(range(batch_start, min(batch_start + BATCH_SIZE, END)))
        examples = [gsm8k_test[i] for i in batch_idx]
        prompt_id_lists = [build_prompt_ids(ex["question"]) for ex in examples]
        gen_results = generate_batch(prompt_id_lists)

        for i, g in zip(batch_idx, gen_results):
            ex = gsm8k_test[i]
            ref = extract_answer(ex["answer"])
            pred = extract_answer(g["text"])
            correct = pred is not None and ref is not None and abs(pred - ref) < 1e-4

            if g["repetition_eos"]:
                category = "looped"
            elif correct:
                category = "correct"
            elif g["answer_eos"] or "####" in g["text"]:
                category = "concluded_wrong"
            else:
                category = "other_truncated"
            counts[category] += 1

            record = {
                "gsm8k_index": i,
                "category": category,
                "question": ex["question"],
                "reference_answer": ref,
                "predicted_answer": pred,
                "generated_text": g["text"],
                "n_tokens": len(g["gen_ids"]),
                "prompt_token_ids": g["prompt_ids"],
                "generated_token_ids": g["gen_ids"],
            }
            f.write(json.dumps(record) + "\n")
        f.flush()

        elapsed = time.time() - t_start
        done = batch_idx[-1] + 1 - START
        total = END - START
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else float("inf")
        print(f"[{done}/{total}] {counts}  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

print("\nFinal counts:", counts)
print(f"Results written to {OUT_PATH}")
