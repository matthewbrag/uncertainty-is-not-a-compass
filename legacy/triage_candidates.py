"""
Step-level outcome-sampling triage over the concluded_wrong candidates found
by scan_candidates.py. For each candidate, probe every discrete 'Step N:'
boundary (not just a few percentage points - steps are explicit now, so we
can afford to check all of them): sample N continuations from that point and
see what fraction recover the correct answer.

Classifies each candidate as:
  - doomed        : 0% recovery at every single step boundary
  - recoverable   : some step boundary has nonzero P(correct)
  - inconclusive  : too few steps to probe meaningfully (shouldn't happen, filtered upstream)

Writes one JSON record per candidate to OUT_PATH, with the full per-step
P(correct) profile, as it goes.
"""
import json
import random
import re
import sys
import time

import torch
from transformers import LogitsProcessorList

from app import get_model, extract_answer, ForceEosAfterAnswer, ForceEosOnRepetition, DEVICE

MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else "qwen3-0.6b"
IN_PATH = f"scan_results_{MODEL_KEY}.jsonl"
OUT_PATH = f"triage_results_{MODEL_KEY}.jsonl"
N_CANDIDATES = 200
N_SAMPLES_PER_STEP = 5
MAX_NEW_TOKENS = 250
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)
MIN_STEPS = 4

tokenizer, model = get_model(MODEL_KEY)
tokenizer.padding_side = "left"


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
    texts = []
    for j in range(len(prompt_id_lists)):
        row = out[j, prompt_len:].tolist()
        if tokenizer.eos_token_id in row:
            row = row[: row.index(tokenizer.eos_token_id) + 1]
        texts.append(tokenizer.decode(row, skip_special_tokens=True))
    return texts


def find_step_token_boundaries(gen_ids):
    """Token indices right after each 'Step N:' marker for N>=2 (cutting
    before step 1 is meaningless - nothing to preserve)."""
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    boundary_chars = [m.start() for m in re.finditer(r"Step \d+:", text)][1:]
    if not boundary_chars:
        return []
    token_boundaries = []
    running = ""
    bi = 0
    for i, tid in enumerate(gen_ids):
        running += tokenizer.decode([tid], skip_special_tokens=True)
        while bi < len(boundary_chars) and len(running) >= boundary_chars[bi]:
            token_boundaries.append(i)
            bi += 1
    return token_boundaries


# --- load candidates ---
records = []
with open(IN_PATH) as f:
    for line in f:
        r = json.loads(line)
        if r["category"] != "concluded_wrong":
            continue
        n_steps = len(re.findall(r"Step \d+:", r["generated_text"]))
        if n_steps >= MIN_STEPS:
            r["n_steps"] = n_steps
            records.append(r)

random.seed(0)
random.shuffle(records)
records = records[:N_CANDIDATES]
print(f"Triaging {len(records)} candidates (>= {MIN_STEPS} steps) from {IN_PATH}")

t_start = time.time()
doomed_count = 0
recoverable_count = 0

with open(OUT_PATH, "w") as out_f:
    for ci, rec in enumerate(records):
        prompt_ids = rec["prompt_token_ids"]
        gen_ids = rec["generated_token_ids"]
        boundaries = find_step_token_boundaries(gen_ids)

        step_results = []
        for cut in boundaries:
            prefix_ids = prompt_ids + gen_ids[:cut]
            batch = [prefix_ids] * N_SAMPLES_PER_STEP
            texts = generate_batch(batch)
            n_correct = 0
            answers = []
            for text in texts:
                pred = extract_answer(text)
                answers.append(pred)
                if pred is not None and rec["reference_answer"] is not None and abs(pred - rec["reference_answer"]) < 1e-4:
                    n_correct += 1
            step_results.append({"cut_token": cut, "n_correct": n_correct, "answers": answers})

        any_recoverable = any(s["n_correct"] > 0 for s in step_results)
        verdict = "recoverable" if any_recoverable else "doomed"
        if any_recoverable:
            recoverable_count += 1
        else:
            doomed_count += 1

        out_record = {
            "gsm8k_index": rec["gsm8k_index"],
            "question": rec["question"],
            "reference_answer": rec["reference_answer"],
            "predicted_answer": rec["predicted_answer"],
            "n_steps": rec["n_steps"],
            "verdict": verdict,
            "step_results": step_results,
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": gen_ids,
        }
        out_f.write(json.dumps(out_record) + "\n")
        out_f.flush()

        elapsed = time.time() - t_start
        eta = (len(records) - ci - 1) * elapsed / (ci + 1) if ci > 0 else 0
        print(f"[{ci+1}/{len(records)}] gsm8k[{rec['gsm8k_index']}] verdict={verdict} "
              f"doomed={doomed_count} recoverable={recoverable_count} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

print(f"\nDone. doomed={doomed_count} recoverable={recoverable_count}")
print(f"Results written to {OUT_PATH}")
