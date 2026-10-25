"""
Tests whether RECURSIVE re-pivoting beats the single-pivot tournament from
shared_trace_monotonic.py. The tournament forks once per candidate point and
runs that fork straight to completion -- it never checks whether the forced
branch itself later hits another high-entropy spot worth correcting. This
script does: fork into LINEAGES alternatives at one point, then for each
lineage independently, run to completion, look inside that completion for
the next-highest-diversity point *after* the previous pivot, force the
model's best genuine alternative there (rank-2 token, since the completion
was greedy-decoded and rank-1 would be a no-op), and repeat for DEPTH total
pivots per lineage. A problem is "rescued" if any of the LINEAGES final
lineages lands on the correct answer -- plus we record the correctness
after each individual depth, so we can see whether going deeper than depth
1 (an ordinary single pivot with LINEAGES branches) actually buys anything.

Pivot selection uses the diversity probe (predict_diversity), same one used
throughout the project, applied to whatever text currently exists -- it
generalizes fine since it's just a probe on hidden states, not tied to
positions in the original trace.

Targets ONLY the known greedy-wrong problem indices from the three
2026-10-24 shared_trace_monotonic_level4_qwen3-1.7b_offset*.jsonl files
(no need to re-discover which problems are wrong), sharded 3 ways.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python recursive_pivot_search.py <model_key> <level> <diversity_probe_path> <shard_total> <shard_index> [out_path]
"""
import json
import re
import sys
import time

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import LogitsProcessorList

from app import (
    get_model, DEFAULT_TEMPLATE, extract_answer,
    ForceEosAfterAnswer, ForceEosOnRepetition, DEVICE,
)

MODEL_KEY = sys.argv[1]
LEVEL = int(sys.argv[2])
DIVERSITY_PROBE_PATH = sys.argv[3]
SHARD_TOTAL = int(sys.argv[4])
SHARD_INDEX = int(sys.argv[5])
OUT_PATH = sys.argv[6] if len(sys.argv) > 6 else f"recursive_pivot_level{LEVEL}_{MODEL_KEY}_shard{SHARD_INDEX}of{SHARD_TOTAL}.jsonl"

LINEAGES = 3          # branch factor at the FIRST pivot only
DEPTH = 3             # total pivot events per lineage (including the first)
GEN_MAX_NEW_TOKENS = 1500
CONT_MAX_NEW_TOKENS = 1300
MAX_BRANCH_BATCH = 15
GEN_BATCH_SIZE = 32
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)
MIN_NEW_REGION = 3    # need at least this many fresh tokens in the initial trace to try at all
MIN_REPIVOT_REGION = 20  # need at least this many fresh tokens after a pivot before re-pivoting again


def parse_numeric(a):
    a = a.strip().rstrip(".")
    try:
        return float(a)
    except ValueError:
        pass
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", a)
    if m:
        sign = -1.0 if m.group(1) else 1.0
        return sign * float(m.group(2)) / float(m.group(3))
    return None


def extract_boxed(sol):
    idx = sol.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(sol) and depth > 0:
        if sol[i] == "{":
            depth += 1
        elif sol[i] == "}":
            depth -= 1
        i += 1
    return sol[start:i - 1] if depth == 0 else None


_subjects = ["algebra", "counting_and_probability", "intermediate_algebra",
             "number_theory", "prealgebra", "precalculus"]
_full_test = concatenate_datasets(
    [load_dataset("EleutherAI/hendrycks_math", s, split="test") for s in _subjects]
)
math_test = []
for r in _full_test:
    if r["level"] != f"Level {LEVEL}":
        continue
    boxed = extract_boxed(r["solution"])
    ref = parse_numeric(boxed) if boxed is not None else None
    if ref is not None:
        math_test.append({"problem": r["problem"], "reference_answer": ref})
print(f"{len(math_test)} level-{LEVEL} numeric-parseable TEST problems available", flush=True)

# pull the known greedy-wrong indices straight from last night's FINAL
# (p95-calibrated) run, instead of re-discovering them -- same math_test
# construction, so indices line up directly. Named explicitly rather than
# globbed: offset280/offset560 are leftover files from the earlier,
# since-superseded 650/450-budget run (2026-10-22) and must NOT be included.
FINAL_RUN_FILES = [
    f"shared_trace_monotonic_level{LEVEL}_{MODEL_KEY}_offset0.jsonl",
    f"shared_trace_monotonic_level{LEVEL}_{MODEL_KEY}_offset277.jsonl",
    f"shared_trace_monotonic_level{LEVEL}_{MODEL_KEY}_offset554.jsonl",
]
known_wrong = set()
for fn in FINAL_RUN_FILES:
    with open(fn) as f:
        for line in f:
            known_wrong.add(json.loads(line)["math_test_index"])
known_wrong = sorted(known_wrong)
print(f"{len(known_wrong)} known greedy-wrong problems pulled from prior run", flush=True)

problem_indices = known_wrong[SHARD_INDEX::SHARD_TOTAL]
print(f"shard {SHARD_INDEX}/{SHARD_TOTAL}: {len(problem_indices)} problems", flush=True)

with open(DIVERSITY_PROBE_PATH) as f:
    _diversity_artifact = json.load(f)
DIVERSITY_LAYER = _diversity_artifact["layer"]

tokenizer, model = get_model(MODEL_KEY)
tokenizer.padding_side = "left"

_div_mean = torch.tensor(_diversity_artifact["scaler_mean"], dtype=torch.float32, device=DEVICE)
_div_scale = torch.tensor(_diversity_artifact["scaler_scale"], dtype=torch.float32, device=DEVICE)
_div_coef = torch.tensor(_diversity_artifact["coef"], dtype=torch.float32, device=DEVICE)
_div_intercept = _diversity_artifact["intercept"]


def predict_diversity(hidden_state_vec):
    x = (hidden_state_vec.float() - _div_mean) / _div_scale
    return float(torch.dot(x, _div_coef).item() + _div_intercept)


def build_prompt_ids(question):
    content = DEFAULT_TEMPLATE.replace("{question}", question)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def pad_batch(id_lists):
    max_len = max(len(x) for x in id_lists)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.tensor(
        [[pad_id] * (max_len - len(x)) + x for x in id_lists], device=DEVICE
    )
    attention_mask = torch.tensor(
        [[0] * (max_len - len(x)) + [1] * len(x) for x in id_lists], device=DEVICE
    )
    return input_ids, attention_mask


def generate_batch_sampled(prompt_ids_list, max_new_tokens):
    input_ids, attention_mask = pad_batch(prompt_ids_list)
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
        results.append(row)
    return results


def force_batch_full(prefixes_and_tokens, max_new_tokens):
    """Like shared_trace_monotonic's _force_batch, but also returns the raw
    completion token ids (needed to keep extending the lineage), not just
    decoded text."""
    batch_ids = [prefix + [tok] for prefix, tok in prefixes_and_tokens]
    input_ids, attention_mask = pad_batch(batch_ids)
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
    results = []
    for j, (prefix, tok) in enumerate(prefixes_and_tokens):
        row = out[j, prompt_len:].tolist()
        if tokenizer.eos_token_id in row:
            row = row[: row.index(tokenizer.eos_token_id) + 1]
        full_ids = prefix + [tok] + row
        text = tokenizer.decode([tok] + row, skip_special_tokens=True)
        results.append({"full_ids": full_ids, "text": text, "length": len(row) + 1})
    return results


def force_all_full(prefixes_and_tokens, max_new_tokens):
    results = []
    for start in range(0, len(prefixes_and_tokens), MAX_BRANCH_BATCH):
        results.extend(force_batch_full(prefixes_and_tokens[start:start + MAX_BRANCH_BATCH], max_new_tokens))
    return results


MAX_SCORE_BATCH = 4  # output_hidden_states=True keeps all 28 layers live at
# once; combined with growing sequence length (up to ~2900 tokens by depth
# 2) this OOMs well before MAX_BRANCH_BATCH's 15 would.


def score_and_pick_batch(id_lists, base_lens, search_froms, search_tos, layer, final_layer, topk):
    """Chunked diversity-probe pivot selection. Critically, each chunk's
    hidden-state tensors (28 layers x seq_len x hidden, kept alive
    simultaneously by output_hidden_states=True) are scored and converted to
    plain Python results, then freed, BEFORE the next chunk runs -- holding
    onto all chunks' hs tensors until the end (as an earlier version of this
    function did, by collecting raw entries first and scoring afterward)
    defeats the entire point of chunking, since by the last chunk you're
    holding as much GPU memory as one big unchunked batch would have.
    Returns one entry per input: None (no valid pivot) or (cut, topk_token_ids)."""
    results = [None] * len(id_lists)
    for start in range(0, len(id_lists), MAX_SCORE_BATCH):
        end = start + MAX_SCORE_BATCH
        chunk_ids = id_lists[start:end]
        input_ids, attention_mask = pad_batch(chunk_ids)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True, logits_to_keep=1, use_cache=False)
        hs = out.hidden_states
        pad_lens = [input_ids.shape[1] - len(x) for x in chunk_ids]
        for local_idx, pad_len in enumerate(pad_lens):
            gi = start + local_idx
            sf, st = search_froms[gi], search_tos[gi]
            if st - sf < 1:
                continue
            bl = base_lens[gi]
            best_cut, best_score = None, -1e18
            for i in range(sf, st):
                pos = pad_len + bl + i - 1
                score = predict_diversity(hs[layer][local_idx, pos])
                if score > best_score:
                    best_score, best_cut = score, i
            pos = pad_len + bl + best_cut
            logits = model.lm_head(hs[final_layer][local_idx, pos])
            probs = torch.softmax(logits, dim=-1)
            topi = torch.topk(probs, topk).indices.tolist()
            results[gi] = (best_cut, topi)
        del out, hs
        torch.cuda.empty_cache()
    return results


random_seed_state = np.random.default_rng(123456)
t_start = time.time()
n_done = 0

with open(OUT_PATH, "w") as f:
    for gbatch_start in range(0, len(problem_indices), GEN_BATCH_SIZE):
        gbatch_indices = problem_indices[gbatch_start:gbatch_start + GEN_BATCH_SIZE]
        examples = [math_test[idx] for idx in gbatch_indices]
        prompt_ids_batch = [build_prompt_ids(ex["problem"]) for ex in examples]
        gen_ids_batch = generate_batch_sampled(prompt_ids_batch, GEN_MAX_NEW_TOKENS)

        problems = []
        for idx, ex, prompt_ids, gen_ids in zip(gbatch_indices, examples, prompt_ids_batch, gen_ids_batch):
            if len(gen_ids) < MIN_NEW_REGION + 2:
                continue
            ref = ex["reference_answer"]
            greedy_answer = extract_answer(tokenizer.decode(gen_ids, skip_special_tokens=True))
            greedy_correct = greedy_answer is not None and ref is not None and abs(greedy_answer - ref) < 1e-4
            problems.append({
                "idx": idx, "ref": ref, "prompt_ids": prompt_ids, "gen_ids": gen_ids,
                "greedy_correct_on_regen": greedy_correct,
            })

        if not problems:
            continue

        # --- initial pivot: one point per problem, forked into LINEAGES alternatives ---
        full_ids_batch = [p["prompt_ids"] + p["gen_ids"] for p in problems]
        base_lens0 = [len(p["prompt_ids"]) for p in problems]
        search_froms0 = [0] * len(problems)
        search_tos0 = [len(p["gen_ids"]) for p in problems]
        picks0 = score_and_pick_batch(full_ids_batch, base_lens0, search_froms0, search_tos0, 28, 28, LINEAGES)

        lineages = []  # one entry per (problem, lineage): dict with live state
        for p, pick, base_len in zip(problems, picks0, base_lens0):
            if pick is None:
                continue
            cut, topi = pick
            prefix = p["prompt_ids"] + p["gen_ids"][:cut + 1]
            for li, tok in enumerate(topi):
                lineages.append({
                    "idx": p["idx"], "ref": p["ref"], "lineage": li,
                    "prefix": prefix, "tok": tok, "base_len": base_len,
                    "prev_cut": cut, "alive": True, "history": [],
                })

        # --- DEPTH rounds: force to completion, score, maybe pick next pivot ---
        for depth in range(DEPTH):
            live = [l for l in lineages if l["alive"]]
            if not live:
                break
            forced_pairs = [(l["prefix"], l["tok"]) for l in live]
            results = force_all_full(forced_pairs, CONT_MAX_NEW_TOKENS)
            for l, res in zip(live, results):
                pred = extract_answer(res["text"])
                correct = pred is not None and l["ref"] is not None and abs(pred - l["ref"]) < 1e-4
                l["history"].append({"depth": depth, "length": res["length"], "correct": correct})
                l["full_ids"] = res["full_ids"]

            if depth == DEPTH - 1:
                break

            # find the next pivot for each still-live lineage, strictly past its previous cut.
            # Require a real chunk of fresh text (MIN_REPIVOT_REGION) before bothering --
            # without this, the probe frequently "re-pivots" one token before the final
            # boxed-answer boilerplate, which isn't a meaningful second decision point.
            full_ids_now = [l["full_ids"] for l in live]
            base_lens_n = [l["base_len"] for l in live]
            search_froms_n = [l["prev_cut"] + 1 for l in live]
            search_tos_n = [
                len(l["full_ids"]) - l["base_len"] - (1 if l["full_ids"][-1] == tokenizer.eos_token_id else 0)
                for l in live
            ]
            eligible = [st - sf >= MIN_REPIVOT_REGION for sf, st in zip(search_froms_n, search_tos_n)]
            picks_n = score_and_pick_batch(full_ids_now, base_lens_n, search_froms_n, search_tos_n,
                                            DIVERSITY_LAYER, 28, 2)
            for l, pick, ok in zip(live, picks_n, eligible):
                if pick is None or not ok:
                    l["alive"] = False
                    continue
                cut, topi = pick
                # rank-2 token: the completion so far was greedy-decoded, so
                # rank-1 here is always a no-op (exactly what's already there).
                new_tok = topi[1]
                l["prefix"] = l["full_ids"][:l["base_len"] + cut + 1]
                l["tok"] = new_tok
                l["prev_cut"] = cut

        # --- write results, grouped back by problem ---
        by_problem = {}
        for l in lineages:
            by_problem.setdefault(l["idx"], []).append(l)
        for p in problems:
            ls = by_problem.get(p["idx"])
            if not ls:
                continue
            final_correct = [l["history"][-1]["correct"] if l["history"] else False for l in ls]
            any_depth_correct = [any(h["correct"] for h in l["history"]) for l in ls]
            record = {
                "math_test_index": p["idx"],
                "reference_answer": p["ref"],
                "greedy_correct_on_regen": p["greedy_correct_on_regen"],
                "lineages": [
                    {"lineage": l["lineage"], "history": l["history"]}
                    for l in ls
                ],
                "rescued_final_depth": any(final_correct),
                "rescued_any_depth": any(any_depth_correct),
                "rescued_first_pivot_only": any(l["history"][0]["correct"] if l["history"] else False for l in ls),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            n_done += 1

        elapsed = time.time() - t_start
        rate = elapsed / max(n_done, 1)
        remaining = len(problem_indices) - n_done
        print(f"[{MODEL_KEY}] {n_done}/{len(problem_indices)} done, "
              f"elapsed={elapsed/60:.1f}min, eta={remaining*rate/60:.1f}min", flush=True)

print(f"\nDone. {n_done} problems. Results in {OUT_PATH}")
