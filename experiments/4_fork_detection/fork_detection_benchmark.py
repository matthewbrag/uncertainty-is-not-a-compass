"""
Scores branch-point heuristics as FORK DETECTORS, independent of whether
branching at a fork rescues anything. Ground truth follows Forking Paths
(Bigelow et al., ICLR 2025): token t is a fork if committing it shifts where
the solution ends up. Estimated with natural sampling (no forcing):
  A = final answers of M samples continued from gen[:t]   (token t re-chosen)
  B = final answers of M samples continued from gen[:t+1] (token t kept)
  fork strength(t) = total variation distance between A and B's answer distributions
  natural outcome entropy(t) = entropy of A
Evaluated at (a) uniformly random positions before the answer line -- an
unbiased sample for rank correlations -- and (b) each heuristic's own top
picks, to see whether what it picks is actually forkier than random.

Heuristics scored: probe (predicted outcome diversity), local entropy,
semantic local entropy (embedding-weighted, synonyms discounted), their
products with the probe, and earliness (1 - relative position) as a
position-only baseline.

Usage (.venv-vllm): CUDA_VISIBLE_DEVICES=<gpu> python experiments/4_fork_detection/fork_detection_benchmark.py <probe_path> <queue_file> <out_path> [max_problems] [offset]
"""
import fcntl
import json
import math
import random
import re
import sys
import time
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM

from forkprobe.stop_rules import DEFAULT_TEMPLATE, extract_answer
from forkprobe.vllm_gen import MODEL, make_llm, generate, _tok as tokenizer

PROBE_PATH, QUEUE, OUT_PATH = sys.argv[1], sys.argv[2], sys.argv[3]
MAX_PROBLEMS = int(sys.argv[4]) if len(sys.argv) > 4 else None
LEVEL, CHUNK = 4, 8
M = 16              # samples per side per position (10 left the null TV 95th pct at 0.6)
N_RANDOM = 24       # random positions per trace
TOP_PER_HEUR = 3    # top picks per heuristic per trace
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
OFFSET = int(sys.argv[5]) if len(sys.argv) > 5 else 0  # extend the benchmark without reusing problems
order = order[OFFSET:]
if MAX_PROBLEMS:
    order = order[:MAX_PROBLEMS]

probe = json.load(open(PROBE_PATH))
llm = make_llm(gpu_memory_utilization=0.6)
hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
p_mean = torch.tensor(probe["scaler_mean"], dtype=torch.float32, device="cuda")
p_scale = torch.tensor(probe["scaler_scale"], dtype=torch.float32, device="cuda")
p_coef = torch.tensor(probe["coef"], dtype=torch.float32, device="cuda")
with torch.no_grad():
    EMB = torch.nn.functional.normalize(hf.get_input_embeddings().weight.float(), dim=-1)


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
    base, n = len(pids), len(gen)
    with torch.no_grad():
        out = hf(input_ids=torch.tensor([pids + gen], device="cuda"), output_hidden_states=True)
    hs = out.hidden_states[probe["layer"]][0, base - 1: base + n - 1].float()
    probe_pred = (((hs - p_mean) / p_scale) @ p_coef + probe["intercept"]).cpu().numpy()
    ent, sem = [], []
    for s in range(0, n, 256):
        logp = torch.log_softmax(out.logits[0, base - 1 + s: base - 1 + min(n, s + 256)].float(), -1)
        ent.append(-(logp.exp() * logp).sum(-1) / math.log(2))
        p20, i20 = logp.exp().topk(20, -1)
        p20 = p20 / p20.sum(-1, keepdim=True)
        e = EMB[i20]
        S = (e @ e.transpose(1, 2)).clamp(min=0)
        sem.append(-(p20 * torch.log2((S @ p20.unsqueeze(-1)).squeeze(-1).clamp(min=1e-12))).sum(-1))
    del out
    return probe_pred, torch.cat(ent).cpu().numpy(), torch.cat(sem).cpu().numpy()


def answer_dist(answers):
    c = Counter("none" if a is None else round(a, 4) for a in answers)
    return {k: v / len(answers) for k, v in c.items()}


def tv(p, q):
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))


def entropy(p):
    return -sum(v * math.log2(v) for v in p.values() if v > 0)


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
            a = extract_answer(tokenizer.decode(gen, skip_special_tokens=True))
            if (a is not None and abs(a - ref) < 1e-4) or len(gen) < 20:
                n_skipped += 1
                continue
            hi = max(2, min(answer_start(gen), len(gen) - 1))
            pp, le, se = signals(pids, gen)
            n = len(gen)
            heur = {
                "probe": pp, "local": le, "sem_local": se,
                "product": np.maximum(pp, 0) * le, "sem_product": np.maximum(pp, 0) * se,
                "earliness": 1 - np.arange(n) / n,
            }
            rng = random.Random(idx)
            rand_pos = sorted(rng.sample(range(1, hi), min(N_RANDOM, hi - 1)))
            top_pos = {h: sorted(range(1, hi), key=lambda i: -v[i])[:TOP_PER_HEUR] for h, v in heur.items()}
            positions = sorted(set(rand_pos) | {p for ps in top_pos.values() for p in ps})
            jobs.append(dict(idx=idx, ref=ref, pids=pids, gen=gen, hi=hi, heur=heur, rand_pos=rand_pos,
                             top_pos=top_pos, positions=positions))
        if jobs:
            inputs, owner, seeds = [], [], []
            for j, job in enumerate(jobs):
                for t in job["positions"]:
                    for side, cut in (("before", t), ("after", t + 1)):
                        for k in range(M):
                            inputs.append(job["pids"] + job["gen"][:cut])
                            owner.append((j, t, side))
                            seeds.append(job["idx"] * 100003 + t * 41 + (0 if side == "before" else 20) + k)
            outs = generate(llm, inputs, CONT_MAX_NEW_TOKENS, sample=True, seeds=seeds)
            ans = {}
            for (j, t, side), row in zip(owner, outs):
                ans.setdefault((j, t, side), []).append(extract_answer(tokenizer.decode(row, skip_special_tokens=True)))
            for j, job in enumerate(jobs):
                pos_out = {}
                for t in job["positions"]:
                    A, B = answer_dist(ans[(j, t, "before")]), answer_dist(ans[(j, t, "after")])
                    acc_a = sum(a is not None and abs(a - job["ref"]) < 1e-4 for a in ans[(j, t, "before")]) / M
                    pos_out[str(t)] = {
                        "fork_tv": round(tv(A, B), 3), "nat_entropy": round(entropy(A), 3), "acc_before": acc_a,
                        "ans_before": ans[(j, t, "before")], "ans_after": ans[(j, t, "after")],
                        "token": tokenizer.decode([job["gen"][t]]),
                        **{h: round(float(v[t]), 4) for h, v in job["heur"].items()},
                    }
                f.write(json.dumps({"math_test_index": job["idx"], "trace_len": len(job["gen"]), "hi": job["hi"],
                                    "random_positions": job["rand_pos"], "top_positions": job["top_pos"],
                                    "positions": pos_out}) + "\n")
                n_done += 1
            f.flush()
        print(f"[claimed up to {start + len(idxs)}/{len(order)}] traces={n_done} skipped={n_skipped} "
              f"elapsed={(time.time() - t_start) / 60:.1f}min", flush=True)
print(f"\nDone. {n_done} traces. Results in {OUT_PATH}")
