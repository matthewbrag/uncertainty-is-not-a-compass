"""
Replaces recursive_pivot_search.py's blind fixed-depth recursion with an
adaptive, outcome-diversity-gated version. The problem with blind recursion
(confirmed in a smoke test): it keeps forcing a divergent token at depth>0
even when the lineage is already correct, which can silently clobber a
right answer with a wrong one. There's no good way to predict in advance
whether a given point is "safe" to leave alone -- Part 1 of this project
already showed the trained diversity probe doesn't correlate well enough
with real outcomes to trust for that.

So don't predict -- observe. At every pivot, force BRANCH_FACTOR alternative
tokens (not one), run each to completion, and look at whether they actually
agree on a final answer:
  - If they agree (outcome diversity == 0): the model is genuinely settled
    here. Stop. Don't recurse further down this path -- there's nothing to
    gain and real risk of clobbering a correct consensus.
  - If they disagree: this is a REAL fork (same ground-truth signal as
    Part 1's outcome_entropy, just read off live instead of predicted).
    Recurse into EACH of the disagreeing branches independently, same rule
    applied again, up to MAX_DEPTH total pivot events.

Every forced completion at every node, at every depth, is a genuine
independent "rescue attempt" for that problem (not just the deepest one) --
a problem counts as rescued if ANY of them lands on the correct answer.
Pivot-point SELECTION still uses the diversity probe (it's still the best
available option for WHERE to cut); only the decide-whether-to-recurse
logic no longer trusts it.

Targets ONLY the known greedy-wrong problem indices from the final
(p95-calibrated) shared_trace_monotonic run, sharded 3 ways.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python adaptive_diversity_search.py <model_key> <level> <diversity_probe_path> <shard_total> <shard_index> [out_path]
"""
import json
import math
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
OUT_PATH = sys.argv[6] if len(sys.argv) > 6 else f"adaptive_diversity_level{LEVEL}_{MODEL_KEY}_shard{SHARD_INDEX}of{SHARD_TOTAL}.jsonl"

BRANCH_FACTOR = 3        # alternatives forced at every pivot, every depth
MAX_DEPTH = 3             # max pivot events along any one path (0, 1, 2)
GEN_MAX_NEW_TOKENS = 1500
CONT_MAX_NEW_TOKENS = 1300
MAX_BRANCH_BATCH = 15
MAX_SCORE_BATCH = 4
GEN_BATCH_SIZE = 32
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)
MIN_NEW_REGION = 3        # need at least this many tokens in the initial trace to try at all
MIN_PIVOT_REGION = 20     # need at least this many fresh tokens to attempt a pivot (else: settle, no forcing)
DIVERSITY_EPS = 1e-6      # treat diversity below this as "full agreement"


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

# Named explicitly, not globbed: offset280/offset560 are leftovers from the
# earlier, since-superseded 650/450-budget run (2026-10-22) and must NOT be
# included -- same mistake caught and fixed in recursive_pivot_search.py.
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


def outcome_diversity(answers, ref):
    """Shannon entropy over distinct final answers (Part 1's ground-truth
    metric), read off LIVE branches instead of predicted. All unparseable
    answers share one bucket, per the original convention."""
    buckets = []
    for a in answers:
        placed = False
        for b in buckets:
            if a is None and b["val"] is None:
                b["n"] += 1
                placed = True
                break
            if a is not None and b["val"] is not None and abs(a - b["val"]) < 1e-4:
                b["n"] += 1
                placed = True
                break
        if not placed:
            buckets.append({"val": a, "n": 1})
    n = len(answers)
    h = 0.0
    for b in buckets:
        p = b["n"] / n
        h -= p * math.log2(p)
    return h


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


def score_and_pick_batch(id_lists, base_lens, search_froms, search_tos, layer, final_layer, topk):
    """Chunked diversity-probe pivot selection. Scores each chunk and
    converts to plain Python results, then frees the hidden-state tensors,
    BEFORE the next chunk runs -- holding all chunks' hs tensors until the
    end would defeat the point of chunking. Returns one entry per input:
    None (no valid pivot) or (cut, topk_token_ids)."""
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


t_start = time.time()
n_done = 0

with open(OUT_PATH, "w") as f:
    for gbatch_start in range(0, len(problem_indices), GEN_BATCH_SIZE):
        gbatch_indices = problem_indices[gbatch_start:gbatch_start + GEN_BATCH_SIZE]
        examples = [math_test[idx] for idx in gbatch_indices]
        prompt_ids_batch = [build_prompt_ids(ex["problem"]) for ex in examples]
        gen_ids_batch = generate_batch_sampled(prompt_ids_batch, GEN_MAX_NEW_TOKENS)

        # one root node per problem
        nodes = []  # each: idx, ref, base_len, full_ids, prev_cut, depth
        problem_log = {}  # idx -> dict accumulating attempts + settle info
        for idx, ex, prompt_ids, gen_ids in zip(gbatch_indices, examples, prompt_ids_batch, gen_ids_batch):
            if len(gen_ids) < MIN_NEW_REGION + 2:
                continue
            ref = ex["reference_answer"]
            greedy_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            greedy_answer = extract_answer(greedy_text)
            greedy_correct = greedy_answer is not None and ref is not None and abs(greedy_answer - ref) < 1e-4
            problem_log[idx] = {
                "math_test_index": idx, "reference_answer": ref,
                "greedy_correct_on_regen": greedy_correct,
                "attempts": [], "nodes_settled_by_agreement": 0, "nodes_settled_by_depth_cap": 0,
                "nodes_settled_by_no_room": 0, "max_depth_reached": 0,
            }
            nodes.append({
                "idx": idx, "ref": ref, "base_len": len(prompt_ids),
                "full_ids": prompt_ids + gen_ids, "prev_cut": -1, "depth": 0,
            })

        for depth in range(MAX_DEPTH):
            if not nodes:
                break
            base_lens = [n["base_len"] for n in nodes]
            search_froms = [n["prev_cut"] + 1 for n in nodes]
            search_tos = []
            for n in nodes:
                gen_len = len(n["full_ids"]) - n["base_len"]
                eos_adjust = 1 if n["full_ids"][-1] == tokenizer.eos_token_id else 0
                search_tos.append(gen_len - eos_adjust)

            pivotable = [st - sf >= MIN_PIVOT_REGION for sf, st in zip(search_froms, search_tos)]
            full_ids_all = [n["full_ids"] for n in nodes]
            picks = score_and_pick_batch(full_ids_all, base_lens, search_froms, search_tos,
                                          DIVERSITY_LAYER, 28, BRANCH_FACTOR)

            next_nodes = []
            forced_pairs = []
            forced_owner = []  # (node, cut) for each forced pair, in order
            for n, pick, ok in zip(nodes, picks, pivotable):
                plog = problem_log[n["idx"]]
                if not ok or pick is None:
                    plog["nodes_settled_by_no_room"] += 1
                    # no room to pivot -- the node's own current text is its
                    # only outcome, record it as a (single, unforced) attempt
                    text = tokenizer.decode(n["full_ids"][n["base_len"]:], skip_special_tokens=True)
                    pred = extract_answer(text)
                    correct = pred is not None and n["ref"] is not None and abs(pred - n["ref"]) < 1e-4
                    plog["attempts"].append({"depth": n["depth"], "length": 0, "correct": correct, "forced": False})
                    continue
                cut, topi = pick
                prefix = n["full_ids"][:n["base_len"] + cut + 1]
                for tok in topi:
                    forced_pairs.append((prefix, tok))
                    forced_owner.append((n, cut))

            if forced_pairs:
                results = force_all_full(forced_pairs, CONT_MAX_NEW_TOKENS)
                # group branch results back by owning node
                by_node_id = {}
                for (n, cut), res in zip(forced_owner, results):
                    pred = extract_answer(res["text"])
                    correct = pred is not None and n["ref"] is not None and abs(pred - n["ref"]) < 1e-4
                    by_node_id.setdefault(id(n), []).append({
                        "node": n, "cut": cut, "res": res, "pred": pred, "correct": correct,
                    })

                for node_id, branches in by_node_id.items():
                    n = branches[0]["node"]
                    cut = branches[0]["cut"]
                    plog = problem_log[n["idx"]]
                    plog["max_depth_reached"] = max(plog["max_depth_reached"], n["depth"])
                    for b in branches:
                        plog["attempts"].append({
                            "depth": n["depth"], "length": b["res"]["length"],
                            "correct": b["correct"], "forced": True,
                        })
                    diversity = outcome_diversity([b["pred"] for b in branches], n["ref"])
                    if diversity <= DIVERSITY_EPS:
                        plog["nodes_settled_by_agreement"] += 1
                        continue
                    if depth == MAX_DEPTH - 1:
                        plog["nodes_settled_by_depth_cap"] += 1
                        continue
                    for b in branches:
                        next_nodes.append({
                            "idx": n["idx"], "ref": n["ref"], "base_len": n["base_len"],
                            "full_ids": b["res"]["full_ids"], "prev_cut": cut, "depth": n["depth"] + 1,
                        })

            nodes = next_nodes

        for idx, plog in problem_log.items():
            plog["rescued"] = any(a["correct"] for a in plog["attempts"])
            plog["n_attempts"] = len(plog["attempts"])
            f.write(json.dumps(plog) + "\n")
            f.flush()
            n_done += 1

        elapsed = time.time() - t_start
        rate = elapsed / max(n_done, 1)
        remaining = len(problem_indices) - n_done
        print(f"[{MODEL_KEY}] {n_done}/{len(problem_indices)} done, "
              f"elapsed={elapsed/60:.1f}min, eta={remaining*rate/60:.1f}min", flush=True)

print(f"\nDone. {n_done} problems. Results in {OUT_PATH}")
