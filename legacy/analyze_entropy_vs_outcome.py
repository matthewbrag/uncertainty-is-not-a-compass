"""
Does local token-entropy (ranked within one trace, not an absolute threshold)
actually predict outcome-criticality?

For each concluded-but-wrong GSM8K example: compute the full per-token raw
entropy trace over its own reasoning (one teacher-forced forward pass), pick
the top-3 local-maxima entropy points and the bottom-3 lowest-entropy points
within the <think> region, then run outcome-sampling (N continuations each)
at all 6 points. If entropy-ranking is a useful signal, P(correct) should
vary much more across the high-entropy points than across the low-entropy
ones.
"""
import torch
from transformers import LogitsProcessorList

from app import (
    get_model, gsm8k_test, DEFAULT_TEMPLATE, extract_answer, compute_entropy,
    ForceEosAfterAnswer, ForceEosOnRepetition, DEVICE, THINK_CLOSE,
)

MODEL_KEY = "qwen3-0.6b"
SCAN_START = 120  # continue past the range already covered by analyze_pivot_points.py
SCAN_BATCH = 8
MAX_SCAN_BATCHES = 12
GEN_MAX_NEW_TOKENS = 1000
N_WRONG_WANTED = 3
N_SAMPLES_PER_POINT = 6
CONTINUATION_MAX_NEW_TOKENS = 350
MIN_GAP_FRAC = 0.08  # minimum spacing between selected points, as a fraction of think_len

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
            ForceEosAfterAnswer(tokenizer, prompt_len, tokenizer.eos_token_id, True),
            ForceEosOnRepetition(prompt_len, tokenizer.eos_token_id),
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


def entropy_trace(prompt_ids, gen_ids):
    """Raw per-step entropy (bits) via one teacher-forced forward pass."""
    full = prompt_ids + gen_ids
    with torch.no_grad():
        logits_full = model(input_ids=torch.tensor([full], device=DEVICE)).logits[0]
    base = len(prompt_ids)
    return [compute_entropy(logits_full[base + i - 1]) for i in range(len(gen_ids))]


def pick_spaced_extremes(values, k, min_gap, descending):
    order = sorted(range(len(values)), key=lambda i: -values[i] if descending else values[i])
    picked = []
    for idx in order:
        if all(abs(idx - p) >= min_gap for p in picked):
            picked.append(idx)
        if len(picked) >= k:
            break
    return sorted(picked)


def probe_point(prompt_ids, gen_ids, cut, ref_answer):
    prefix_ids = prompt_ids + gen_ids[:cut]
    batch = [prefix_ids] * N_SAMPLES_PER_POINT
    gen_results = generate_batch(batch, CONTINUATION_MAX_NEW_TOKENS, do_sample=True)
    n_correct = 0
    answers = []
    for _, _, text, _ in gen_results:
        pred = extract_answer(text)
        answers.append(pred)
        if pred is not None and ref_answer is not None and abs(pred - ref_answer) < 1e-4:
            n_correct += 1
    return n_correct, answers


# --- Step 1: scan for concluded-but-wrong examples ---
print(f"Scanning GSM8K test[{SCAN_START}:] for concluded-but-wrong answers...")
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
            running = ""
            think_len = len(gen_ids)
            for i, tid in enumerate(gen_ids):
                running += tokenizer.decode([tid], skip_special_tokens=True)
                if len(running) >= think_close_idx:
                    think_len = i
                    break
            if think_len < 20:
                continue  # too short to meaningfully pick 3 spaced high/low points
            wrong_examples.append({
                "gsm8k_index": idx + k,
                "question": ex["question"],
                "reference_answer": ref,
                "predicted_answer": pred,
                "prompt_ids": prompt_ids,
                "gen_ids": gen_ids,
                "think_len": think_len,
            })
    idx += SCAN_BATCH
    print(f"  scanned to {idx}: {scanned_count} total, {truncated_count} truncated, {len(wrong_examples)} concluded-but-wrong (long enough)")
    if len(wrong_examples) >= N_WRONG_WANTED:
        break

print(f"\nFound {len(wrong_examples)} usable concluded-but-wrong examples.\n")
selected = wrong_examples[:N_WRONG_WANTED]

# --- Step 2: for each, compute entropy trace and probe high vs low points ---
for ex in selected:
    think_len = ex["think_len"]
    trace = entropy_trace(ex["prompt_ids"], ex["gen_ids"][:think_len])
    min_gap = max(1, int(think_len * MIN_GAP_FRAC))

    high_idx = pick_spaced_extremes(trace, 3, min_gap, descending=True)
    low_idx = pick_spaced_extremes(trace, 3, min_gap, descending=False)

    print("=" * 100)
    print(f"### gsm8k[{ex['gsm8k_index']}] — predicted {ex['predicted_answer']}, reference {ex['reference_answer']}, think_len={think_len}")
    print(f"Q: {ex['question']}")
    print(f"entropy trace: min={min(trace):.2f} max={max(trace):.2f} mean={sum(trace)/len(trace):.2f} bits")

    print("\n  -- HIGH entropy points (top-3 local maxima) --")
    high_results = []
    for tok_idx in high_idx:
        n_correct, answers = probe_point(ex["prompt_ids"], ex["gen_ids"], tok_idx + 1, ex["reference_answer"])
        high_results.append(n_correct)
        pct = int(100 * tok_idx / think_len)
        context = tokenizer.decode(ex["gen_ids"][max(0, tok_idx - 8):tok_idx + 2], skip_special_tokens=True)
        print(f"    tok {tok_idx}/{think_len} ({pct}%) entropy={trace[tok_idx]:.2f} bits  P(correct)={n_correct}/{N_SAMPLES_PER_POINT}  answers={answers}")
        print(f"      context: ...{context.replace(chr(10), ' ')!r}...")

    print("\n  -- LOW entropy points (bottom-3) --")
    low_results = []
    for tok_idx in low_idx:
        n_correct, answers = probe_point(ex["prompt_ids"], ex["gen_ids"], tok_idx + 1, ex["reference_answer"])
        low_results.append(n_correct)
        pct = int(100 * tok_idx / think_len)
        context = tokenizer.decode(ex["gen_ids"][max(0, tok_idx - 8):tok_idx + 2], skip_special_tokens=True)
        print(f"    tok {tok_idx}/{think_len} ({pct}%) entropy={trace[tok_idx]:.2f} bits  P(correct)={n_correct}/{N_SAMPLES_PER_POINT}  answers={answers}")
        print(f"      context: ...{context.replace(chr(10), ' ')!r}...")

    high_spread = max(high_results) - min(high_results)
    low_spread = max(low_results) - min(low_results)
    print(f"\n  SUMMARY: high-entropy P(correct) spread={high_spread}/{N_SAMPLES_PER_POINT} across {high_results}, "
          f"low-entropy spread={low_spread}/{N_SAMPLES_PER_POINT} across {low_results}")
    print()

print("Done.")
