"""
ABLATION of fork_prune_tournament.py: 2x2 of {semantic-product cuts, evenly
spaced cuts} x {lookahead pruning, none}. Every cell proposes 12 cuts + token 0
(4 alternatives each) and completes 20 branches, so cells differ only in which
cuts are proposed and whether rejoining/duplicate branches are pruned first.

Original description:
Lookahead pruning for fork branching. Most forced alternatives either
paraphrase their way back onto the original path (synonyms, formatting) or
duplicate a sibling, and those branches mostly re-vote for the trace's wrong
answer. Here every candidate branch first runs a short greedy lookahead;
branches that rejoin the original continuation or duplicate another branch's
lookahead are pruned, and the full budget goes to surviving branches at the
strongest forks. Lookahead tokens are charged to the arm's budget.

Arms (each scored by oracle "any correct" and by majority vote, against
best-of-N sampled restarts at the same token budget on the same problem):
  uniform_noprune     token 0 + 4 evenly spaced cuts, 4 alts each (original token dropped), greedy
  semprod_noprune     token 0 + top-4 semantic-product cuts (one per line), 4 alts each, greedy
  semprod_pruned      candidates: token 0 + top-12 semantic-product cuts (one per line) x 4 alts;
                      32-token lookahead; drop rejoiners/duplicates; complete the best 20
  restarts            32 sampled solutions, budget-matched at analysis time

"semantic product" = max(0, probe outcome prediction) x embedding-weighted local entropy.

Usage (.venv-vllm): CUDA_VISIBLE_DEVICES=<gpu> python experiments/3_tournament/fork_prune_ablation.py <probe_path> <queue_file> <out_path> [max_problems]
"""
import fcntl
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM

from forkprobe.stop_rules import DEFAULT_TEMPLATE, extract_answer
from forkprobe.vllm_gen import MODEL, make_llm, generate, _tok as tokenizer

PROBE_PATH, QUEUE, OUT_PATH = sys.argv[1], sys.argv[2], sys.argv[3]
MAX_PROBLEMS = int(sys.argv[4]) if len(sys.argv) > 4 else None
LEVEL, CHUNK = 4, 16
N_ALTS, N_RESTARTS = 4, 32
N_CAND_CUTS, LOOKAHEAD, KEEP = 12, 32, 20
GEN_MAX_NEW_TOKENS, CONT_MAX_NEW_TOKENS = 1500, 1300
ANSWER_MARKERS = ["Final Answer", "\\boxed", "####"]


def parse_numeric(a):
    a = a.strip().rstrip(".")
    try:
        return float(a)
    except ValueError:
        pass
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", a)
    return (-1.0 if m.group(1) else 1.0) * float(m.group(2)) / float(m.group(3)) if m else None


def extract_boxed(sol):
    idx = sol.rfind("\\boxed{")
    if idx == -1:
        return None
    start = i = idx + len("\\boxed{")
    depth = 1
    while i < len(sol) and depth > 0:
        depth += {"{": 1, "}": -1}.get(sol[i], 0)
        i += 1
    return sol[start:i - 1] if depth == 0 else None


subjects = ["algebra", "counting_and_probability", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]
math_test = []
for r in concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", s, split="test") for s in subjects]):
    if r["level"] != f"Level {LEVEL}":
        continue
    b = extract_boxed(r["solution"])
    ref = parse_numeric(b) if b is not None else None
    if ref is not None:
        math_test.append({"problem": r["problem"], "reference_answer": ref})
order = list(range(len(math_test)))
random.Random(7000).shuffle(order)
if MAX_PROBLEMS:
    order = order[:MAX_PROBLEMS]

probe = json.load(open(PROBE_PATH))
llm = make_llm(gpu_memory_utilization=0.6)
hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
with torch.no_grad():
    EMB = torch.nn.functional.normalize(hf.get_input_embeddings().weight.float(), dim=-1)
p_mean = torch.tensor(probe["scaler_mean"], dtype=torch.float32, device="cuda")
p_scale = torch.tensor(probe["scaler_scale"], dtype=torch.float32, device="cuda")
p_coef = torch.tensor(probe["coef"], dtype=torch.float32, device="cuda")


def claim_chunk():
    with open(QUEUE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        s = f.read().strip()
        start = int(s) if s else 0
        f.seek(0)
        f.truncate()
        f.write(str(start + CHUNK))
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return start


def build_prompt_ids(question):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": DEFAULT_TEMPLATE.replace("{question}", question)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def score(text, ref):
    a = extract_answer(text)
    return a, (a is not None and ref is not None and abs(a - ref) < 1e-4)


def answer_start(gen_ids):
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    pos = max(text.rfind(m) for m in ANSWER_MARKERS)
    if pos < 0:
        return len(gen_ids)
    lo, hi = 0, len(gen_ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenizer.decode(gen_ids[:mid + 1], skip_special_tokens=True)) > pos:
            hi = mid
        else:
            lo = mid + 1
    return lo


@torch.no_grad()
def signals(pids, gen):
    """Semantic product per position, plus top-5 ids of each position's distribution."""
    base, n = len(pids), len(gen)
    out = hf(input_ids=torch.tensor([pids + gen], device="cuda"), output_hidden_states=True)
    hs = out.hidden_states[probe["layer"]][0, base - 1: base + n - 1].float()
    pred = ((hs - p_mean) / p_scale) @ p_coef + probe["intercept"]
    sem, top5 = [], []
    for s in range(0, n, 256):
        logp = torch.log_softmax(out.logits[0, base - 1 + s: base - 1 + min(n, s + 256)].float(), -1)
        p20, i20 = logp.exp().topk(20, -1)
        p20 = p20 / p20.sum(-1, keepdim=True)
        e = EMB[i20]
        S = (e @ e.transpose(1, 2)).clamp(min=0)
        sem.append(-(p20 * torch.log2((S @ p20.unsqueeze(-1)).squeeze(-1).clamp(min=1e-12))).sum(-1))
        top5.append(i20[:, :5])
    del out
    semprod = (pred.clamp(min=0) * torch.cat(sem)).cpu().numpy()
    return semprod, torch.cat(top5).cpu().tolist()


def one_per_line_top(score_arr, gen_texts, lo, hi, k):
    line_of, line = [], 0
    for t in gen_texts:
        line_of.append(line)
        line += t.count("\n")
    picks, used = [], set()
    for i in sorted(range(lo, hi), key=lambda i: -score_arr[i]):
        if line_of[i] not in used:
            picks.append(i)
            used.add(line_of[i])
        if len(picks) == k:
            break
    return picks


def rejoins(lookahead_ids, gen, cut):
    """Branch has merged back onto the original path if its last 12 lookahead
    tokens appear verbatim in the original continuation shortly after the cut."""
    if len(lookahead_ids) < 12:
        return False
    # normalize away whitespace, markdown and LaTeX-wrapper differences ("_{6}" vs
    # "_\\text{6}", ".  \\n\\n" vs ".\\n\\n"): exact matching missed many real rejoins
    norm = lambda s: re.sub(r"\\text|[\s*#_{}$\\-]", "", s)
    tail = norm(tokenizer.decode(lookahead_ids[-12:], skip_special_tokens=True))
    window = norm(tokenizer.decode(gen[cut: cut + LOOKAHEAD + 64], skip_special_tokens=True))
    return len(tail) > 6 and tail in window


t_start = time.time()
n_done = n_skipped = 0
with open(OUT_PATH, "w") as f:
    while True:
        start = claim_chunk()
        if start >= len(order):
            break
        idxs = order[start:start + CHUNK]
        examples = [math_test[i] for i in idxs]
        prompts = [build_prompt_ids(ex["problem"]) for ex in examples]
        traces = generate(llm, prompts, GEN_MAX_NEW_TOKENS, sample=True, seeds=idxs)
        jobs = []
        for idx, ex, pids, gen in zip(idxs, examples, prompts, traces):
            ref = ex["reference_answer"]
            trace_answer, ok = score(tokenizer.decode(gen, skip_special_tokens=True), ref)
            if ok or len(gen) < 20:
                n_skipped += 1
                continue
            semprod, top5 = signals(pids, gen)
            texts = [tokenizer.decode([t]) for t in gen]
            hi = max(2, min(answer_start(gen), len(gen) - 1))
            alts = lambda c: [t for t in top5[c] if t != gen[c]][:N_ALTS]
            fam_cuts = {
                "semprod": sorted({0} | set(one_per_line_top(semprod, texts, 1, hi, N_CAND_CUTS))),
                "uniform": sorted({0} | {1 + (hi - 1) * (i + 1) // (N_CAND_CUTS + 1) for i in range(N_CAND_CUTS)}),
            }
            jobs.append(dict(idx=idx, ref=ref, pids=pids, gen=gen, trace_answer=trace_answer, semprod=semprod,
                             cands={fam: [(c, t) for c in cuts for t in alts(c)] for fam, cuts in fam_cuts.items()}))
        if jobs:
            # 1) one lookahead per unique (cut, token) across both candidate families
            uniq = [(j, c, t) for j, job in enumerate(jobs) for (c, t) in sorted(set(job["cands"]["semprod"]) | set(job["cands"]["uniform"]))]
            la_out = generate(llm, [jobs[j]["pids"] + jobs[j]["gen"][:c] + [t] for j, c, t in uniq], LOOKAHEAD, sample=False)
            la_of = {(j, c, t): la for (j, c, t), la in zip(uniq, la_out)}

            def select(job, j, fam, items, k):
                """Token-0 restarts first; then by semantic product (semprod family) or in a
                seeded random order (uniform family -- no heuristic involved)."""
                rest = [x for x in items if x[0] > 0]
                if fam == "semprod":
                    rest.sort(key=lambda x: -job["semprod"][x[0]])
                else:
                    random.Random(job["idx"] * 7 + 3).shuffle(rest)
                return ([x for x in items if x[0] == 0] + rest)[:k]

            for j, job in enumerate(jobs):
                job["plan"], job["overhead"], job["stats"] = {}, {}, {}
                for fam, cands in job["cands"].items():
                    survivors, seen, n_rejoin, n_dup = [], set(), 0, 0
                    for c, t in cands:
                        la = la_of[(j, c, t)]
                        key = tokenizer.decode([t] + la, skip_special_tokens=True).strip()
                        if c > 0 and rejoins(la, job["gen"], c):
                            n_rejoin += 1
                            continue
                        if key in seen:
                            n_dup += 1
                            continue
                        seen.add(key)
                        survivors.append((c, t))
                    kept = select(job, j, fam, survivors, KEEP)
                    job["plan"][f"{fam}_pruned"] = [(c, t, la_of[(j, c, t)]) for c, t in kept]
                    job["plan"][f"{fam}_noprune"] = [(c, t, []) for c, t in select(job, j, fam, list(cands), KEEP)]
                    spent = sum(len(la_of[(j, c, t)]) + 1 for c, t in cands)
                    kept_la = sum(len(la_of[(j, c, t)]) + 1 for c, t in kept)
                    job["overhead"][f"{fam}_pruned"] = spent - kept_la  # lookahead on branches then dropped
                    job["stats"][fam] = {"candidates": len(cands), "rejoin": n_rejoin, "duplicate": n_dup}
            # 2) completions (pruned arms continue from the end of their lookahead)
            full_in, full_owner = [], []
            for j, job in enumerate(jobs):
                for arm, items in job["plan"].items():
                    for c, t, la in items:
                        full_in.append(job["pids"] + job["gen"][:c] + [t] + la)
                        full_owner.append((j, arm, c, t, la))
            full_out = generate(llm, full_in, CONT_MAX_NEW_TOKENS, sample=False)
            restart_in = [job["pids"] for job in jobs for _ in range(N_RESTARTS)]
            restart_seeds = [job["idx"] * 10007 + 5000 + s for job in jobs for s in range(N_RESTARTS)]
            restart_out = generate(llm, restart_in, GEN_MAX_NEW_TOKENS, sample=True, seeds=restart_seeds)
            for job in jobs:
                job["arms"] = {arm: [] for arm in job["plan"]}
            for (j, arm, c, t, la), row in zip(full_owner, full_out):
                text_ids = [t] + la + row
                a, ok = score(tokenizer.decode(text_ids, skip_special_tokens=True), jobs[j]["ref"])
                jobs[j]["arms"][arm].append({"cut": c, "answer": a, "correct": ok, "length": len(text_ids)})
            for j, job in enumerate(jobs):
                rs = []
                for row in restart_out[j * N_RESTARTS:(j + 1) * N_RESTARTS]:
                    a, ok = score(tokenizer.decode(row, skip_special_tokens=True), job["ref"])
                    rs.append({"answer": a, "correct": ok, "length": len(row)})
                f.write(json.dumps({
                    "math_test_index": job["idx"], "reference_answer": job["ref"], "trace_answer": job["trace_answer"],
                    "trace_len": len(job["gen"]), "arms": job["arms"], "restarts": rs,
                    "prune_stats_by_family": job["stats"], "arm_overhead": job["overhead"],
                }) + "\n")
                n_done += 1
            f.flush()
        print(f"[claimed up to {start + len(idxs)}/{len(order)}] scored={n_done} skipped={n_skipped} "
              f"elapsed={(time.time() - t_start) / 60:.1f}min", flush=True)
print(f"\nDone. {n_done} scored, {n_skipped} skipped. Results in {OUT_PATH}")
