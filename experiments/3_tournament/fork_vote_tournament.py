"""
Can branching at the forks compete with independent sampling once a final
answer has to be CHOSEN (majority vote), not just exist somewhere (oracle)?

The previous tournament found branching competitive with best-of-N under the
oracle "any branch correct" metric, but far behind under majority vote (<=16%
vs ~24% at equal tokens): every branch inherits the trace's wrong prefix, so
the wrong answer keeps winning the vote. This tests whether decorrelating the
branches closes that gap. All arms share the same cut positions -- the token-0
restart plus the top local-entropy point in each of 4 segments before the
answer line -- and never re-run the trace's own token at a cut:

  fork_greedy     force each of the top-5 alternatives (minus the original
                  token, max 4), greedy continuation      -- previous best design
  fork_sampled    same forced tokens, SAMPLED continuation (T=0.7)
  fork_resample   no forcing: 4 sampled continuations from the prefix
  restarts        32 independent sampled solutions; matched to each arm's
                  token budget per problem at analysis time (best-of-N)

Usage (.venv-vllm): CUDA_VISIBLE_DEVICES=<gpu> python experiments/3_tournament/fork_vote_tournament.py <queue_file> <out_path> [max_problems]
"""
import fcntl
import json
import math
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

QUEUE, OUT_PATH = sys.argv[1], sys.argv[2]
MAX_PROBLEMS = int(sys.argv[3]) if len(sys.argv) > 3 else None
LEVEL, CHUNK = 4, 16
N_ALTS, N_RESTARTS, N_SEGMENTS = 4, 32, 4
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
random.Random(7000).shuffle(order)  # same problem order as every earlier tournament
if MAX_PROBLEMS:
    order = order[:MAX_PROBLEMS]

llm = make_llm(gpu_memory_utilization=0.6)
hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()


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


def local_entropy_and_topk(pids, gen):
    """Entropy (bits) and top-5 ids of the distribution each gen token was chosen from."""
    base, n = len(pids), len(gen)
    with torch.no_grad():
        out = hf(input_ids=torch.tensor([pids + gen], device="cuda"))
    ent, topi = [], []
    for s in range(0, n, 256):
        logp = torch.log_softmax(out.logits[0, base - 1 + s: base - 1 + min(n, s + 256)].float(), -1)
        ent.append(-(logp.exp() * logp).sum(-1) / math.log(2))
        topi.append(logp.topk(5, -1).indices)
    del out
    return torch.cat(ent).cpu().numpy(), torch.cat(topi).cpu().tolist()


def stratified(score_arr, lo, hi, k):
    min_gap = max(1, (hi - lo) // (k * 4))
    picks = []
    for s in range(k):
        a, b = lo + (hi - lo) * s // k, lo + (hi - lo) * (s + 1) // k
        if b <= a:
            continue
        seg = [i for i in range(a, b) if all(abs(i - p) >= min_gap for p in picks)] or list(range(a, b))
        picks.append(max(seg, key=lambda i: score_arr[i]))
    return picks


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
            trace_answer, trace_ok = score(tokenizer.decode(gen, skip_special_tokens=True), ref)
            if trace_ok or len(gen) < 12:
                n_skipped += 1
                continue
            local, topk = local_entropy_and_topk(pids, gen)
            hi = max(2, min(answer_start(gen), len(gen) - 1))
            cuts = sorted({0} | set(stratified(local, 1, hi, N_SEGMENTS)))
            alts = {c: [t for t in topk[c] if t != gen[c]][:N_ALTS] for c in cuts}
            jobs.append(dict(idx=idx, ref=ref, pids=pids, gen=gen, trace_answer=trace_answer,
                             cuts=cuts, alts=alts, local=local))

        if jobs:
            forced_in, forced_owner, resamp_in, resamp_owner, restart_in = [], [], [], [], []
            forced_seeds, resamp_seeds, restart_seeds = [], [], []
            for j, job in enumerate(jobs):
                for c in job["cuts"]:
                    for k, t in enumerate(job["alts"][c]):
                        forced_in.append(job["pids"] + job["gen"][:c] + [t])  # forced token REPLACES gen[c]
                        forced_owner.append((j, c, t))
                        forced_seeds.append(job["idx"] * 10007 + c * 31 + k)
                    for k in range(N_ALTS):
                        resamp_in.append(job["pids"] + job["gen"][:c])
                        resamp_owner.append((j, c))
                        resamp_seeds.append(job["idx"] * 10007 + c * 31 + 17 + k)
                restart_in += [job["pids"]] * N_RESTARTS
                restart_seeds += [job["idx"] * 10007 + 5000 + k for k in range(N_RESTARTS)]
            greedy_out = generate(llm, forced_in, CONT_MAX_NEW_TOKENS, sample=False)
            sampled_out = generate(llm, forced_in, CONT_MAX_NEW_TOKENS, sample=True, seeds=forced_seeds)
            resamp_out = generate(llm, resamp_in, CONT_MAX_NEW_TOKENS, sample=True, seeds=resamp_seeds)
            restart_out = generate(llm, restart_in, GEN_MAX_NEW_TOKENS, sample=True, seeds=restart_seeds)

            for job in jobs:
                job["arms"] = {"fork_greedy": [], "fork_sampled": [], "fork_resample": []}
                job["restarts"] = []
            for name, outs in (("fork_greedy", greedy_out), ("fork_sampled", sampled_out)):
                for (j, c, t), row in zip(forced_owner, outs):
                    a, ok = score(tokenizer.decode([t] + row, skip_special_tokens=True), jobs[j]["ref"])
                    jobs[j]["arms"][name].append({"cut": c, "answer": a, "correct": ok, "length": len(row) + 1})
            for (j, c), row in zip(resamp_owner, resamp_out):
                a, ok = score(tokenizer.decode(row, skip_special_tokens=True), jobs[j]["ref"])
                jobs[j]["arms"]["fork_resample"].append({"cut": c, "answer": a, "correct": ok, "length": len(row)})
            for k, row in enumerate(restart_out):
                job = jobs[k // N_RESTARTS]
                a, ok = score(tokenizer.decode(row, skip_special_tokens=True), job["ref"])
                job["restarts"].append({"answer": a, "correct": ok, "length": len(row)})

            for job in jobs:
                f.write(json.dumps({
                    "math_test_index": job["idx"], "reference_answer": job["ref"],
                    "trace_answer": job["trace_answer"], "trace_len": len(job["gen"]), "cuts": job["cuts"],
                    "local_entropy_at_cuts": [round(float(job["local"][c]), 3) for c in job["cuts"]],
                    "arms": job["arms"], "restarts": job["restarts"],
                }) + "\n")
                n_done += 1
            f.flush()
        print(f"[claimed up to {start + len(idxs)}/{len(order)}] scored={n_done} skipped={n_skipped} "
              f"elapsed={(time.time() - t_start) / 60:.1f}min", flush=True)

print(f"\nDone. {n_done} scored, {n_skipped} skipped. Results in {OUT_PATH}")
