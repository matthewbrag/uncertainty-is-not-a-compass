"""
Find GSM8K problems where the 0.6B model answers wrong, then probe several
cut points along its own reasoning trace: from each point, sample N fresh
continuations and measure the fraction that still reach the correct final
answer. This is "outcome entropy" rather than token entropy - it tells you
whether a point in the trace is actually a recoverable decision point, or
whether the trajectory is already doomed regardless of what gets resampled
from there onward.
"""
import torch
from transformers import LogitsProcessorList

from forkprobe.hf import (
    get_model, gsm8k_test, DEFAULT_TEMPLATE, extract_answer,
    ForceEosAfterAnswer, DEVICE, THINK_CLOSE,
)

MODEL_KEY = "qwen3-0.6b"
SCAN_START = 40          # avoid re-covering indices already poked at manually
SCAN_BATCH = 8
MAX_SCAN_BATCHES = 10
GEN_MAX_NEW_TOKENS = 1000  # give it a real chance to actually conclude
N_WRONG_WANTED = 4
N_PROBE_POINTS = 5
N_SAMPLES_PER_POINT = 6
CONTINUATION_MAX_NEW_TOKENS = 350

# official baseline sampling settings
SAMPLE_KW = dict(temperature=0.6, top_k=20, top_p=0.95)

tokenizer, model = get_model(MODEL_KEY)
tokenizer.padding_side = "left"


def build_prompt_ids(question):
    content = DEFAULT_TEMPLATE.replace("{question}", question)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
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


def generate_batch(prompt_id_lists, max_new_tokens, do_sample):
    input_ids, attention_mask = pad_batch_ids(prompt_id_lists)
    prompt_len = input_ids.shape[1]
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        logits_processor=LogitsProcessorList([
            ForceEosAfterAnswer(tokenizer, prompt_len, tokenizer.eos_token_id, True)
        ]),
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, **SAMPLE_KW)
    else:
        gen_kwargs["do_sample"] = False
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
    results = []
    for j in range(len(prompt_id_lists)):
        row = out[j, prompt_len:].tolist()
        eos_hit = tokenizer.eos_token_id in row
        if eos_hit:
            row = row[: row.index(tokenizer.eos_token_id) + 1]
        text = tokenizer.decode(row, skip_special_tokens=True)
        results.append((prompt_id_lists[j], row, text, eos_hit))
    return results


# --- Step 1: scan for wrong answers that the model actually concluded
# (eos_hit=True means ForceEosAfterAnswer matched '#### <number>' after a
# closed </think> block - i.e. a genuine, deliberate wrong answer, not just
# truncated rambling where the fallback regex grabbed a stray number).
print(f"Scanning GSM8K test[{SCAN_START}:] for concluded-but-wrong answers (0.6b, official-baseline sampling)...")
wrong_examples = []
truncated_count = 0
scanned_count = 0
idx = SCAN_START
for _ in range(MAX_SCAN_BATCHES):
    batch_qs = [gsm8k_test[i] for i in range(idx, idx + SCAN_BATCH)]
    prompt_id_lists = [build_prompt_ids(ex["question"]) for ex in batch_qs]
    gen_results = generate_batch(prompt_id_lists, GEN_MAX_NEW_TOKENS, do_sample=True)
    for k, (prompt_ids, gen_ids, text, eos_hit) in enumerate(gen_results):
        ex = batch_qs[k]
        scanned_count += 1
        if not eos_hit:
            truncated_count += 1
            continue
        ref = extract_answer(ex["answer"])
        pred = extract_answer(text)
        correct = pred is not None and ref is not None and abs(pred - ref) < 1e-4
        if not correct:
            think_close_idx = text.find(THINK_CLOSE)
            # locate the token index of the </think> boundary by walking
            # cumulative decoded length (approximate but fine for picking cut points)
            running = ""
            think_len = len(gen_ids)
            for i, tid in enumerate(gen_ids):
                running += tokenizer.decode([tid], skip_special_tokens=True)
                if len(running) >= think_close_idx:
                    think_len = i
                    break
            wrong_examples.append({
                "gsm8k_index": idx + k,
                "question": ex["question"],
                "reference_answer": ref,
                "predicted_answer": pred,
                "prompt_ids": prompt_ids,
                "gen_ids": gen_ids,
                "text": text,
                "think_len": think_len,
            })
    idx += SCAN_BATCH
    print(f"  scanned up to index {idx}: {scanned_count} total, {truncated_count} truncated (never concluded), {len(wrong_examples)} concluded-but-wrong")
    if len(wrong_examples) >= N_WRONG_WANTED * 2:
        break

print(f"\nTruncation rate so far: {truncated_count}/{scanned_count} never reached a '#### <number>' conclusion within {GEN_MAX_NEW_TOKENS} tokens.")
print(f"Found {len(wrong_examples)} concluded-but-wrong examples in range [{SCAN_START}:{idx}]")

wrong_examples.sort(key=lambda e: -e["think_len"])
selected = wrong_examples[:N_WRONG_WANTED]

print(f"\nSelected {len(selected)} candidates (longest reasoning first) for pivot-point probing:\n")
for ex in selected:
    print(f"  gsm8k[{ex['gsm8k_index']}] think_len={ex['think_len']} pred={ex['predicted_answer']} ref={ex['reference_answer']}")
    print(f"    Q: {ex['question'][:100]}...")

# --- Step 2: probe cut points for each selected wrong example ---
print("\n" + "=" * 100)
for ex in selected:
    print(f"\n### gsm8k[{ex['gsm8k_index']}] — predicted {ex['predicted_answer']}, reference {ex['reference_answer']}")
    print(f"Q: {ex['question']}")

    think_len = ex["think_len"]
    if think_len < N_PROBE_POINTS:
        print("  (reasoning too short to probe meaningfully, skipping)")
        continue

    cut_points = sorted(set(
        max(1, int(think_len * frac))
        for frac in [0.1, 0.3, 0.5, 0.7, 0.9]
    ))

    for cut in cut_points:
        prefix_ids = ex["prompt_ids"] + ex["gen_ids"][:cut]
        prefix_text = tokenizer.decode(ex["gen_ids"][:cut], skip_special_tokens=True)

        batch_prefix_ids = [prefix_ids] * N_SAMPLES_PER_POINT
        gen_results = generate_batch(batch_prefix_ids, CONTINUATION_MAX_NEW_TOKENS, do_sample=True)

        n_correct = 0
        answers_seen = []
        for _, _, text, _ in gen_results:
            pred = extract_answer(text)
            answers_seen.append(pred)
            if pred is not None and ex["reference_answer"] is not None and abs(pred - ex["reference_answer"]) < 1e-4:
                n_correct += 1

        pct = int(100 * cut / think_len)
        tail = prefix_text[-60:].replace("\n", " ")
        print(f"  [{pct:3d}% through reasoning, tok {cut}/{think_len}] P(correct)={n_correct}/{N_SAMPLES_PER_POINT}  answers={answers_seen}")
        print(f"      ...last text before cut: ...{tail!r}")

print("\nDone.")
