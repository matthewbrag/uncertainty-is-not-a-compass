"""
vLLM version of advanced_tournament.py, plus the local-entropy strategies.
Generation (traces, forced branches, resampled branches, restarts) runs on
vLLM; the probe score and local entropy come from one HF forward pass per
trace, in the same process. Runs in .venv-vllm.

Every learned arm is "restart + 4 picks" so the token-0 restart is held equal:
  uniform                        5 evenly spaced cuts incl. token 0 and EOS (control)
  restart_uniform_interior       0 + 4 evenly spaced before the answer line (control for the arms below)
  restart_stratified_probe       0 + top probe score (non-digit) per segment -- the previous learned arm
  restart_local_only             0 + top local entropy per segment -- does local entropy alone do it?
  restart_product                0 + top max(0, outcome) x local entropy per segment
  restart_zsum                   0 + top z(outcome) + z(local entropy) per segment
  restart_product_guarded        0 + 4 cuts sampled in proportion to the product, at most one per
                                 line, skipping positions whose top alternatives are only
                                 whitespace / punctuation / markdown (formatting forks)
  resample_uniform_cuts          uniform's positions, 5 sampled continuations each, no forcing
  sampled_restarts               25 sampled full solutions (best-of-25)
"outcome" = the diversity probe's prediction (bits); "local entropy" = entropy of
the model's own next-token distribution (bits).

Workers pull chunks of problems from a shared counter file. Records keep the
trace token ids, probe z, local entropy and every branch outcome.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python experiments/3_tournament/advanced_tournament_vllm.py <probe_path> <queue_file> <out_path> [max_problems]
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

PROBE_PATH, QUEUE, OUT_PATH = sys.argv[1], sys.argv[2], sys.argv[3]
MAX_PROBLEMS = int(sys.argv[4]) if len(sys.argv) > 4 else None
LEVEL = 4
CHUNK = 16  # bigger than the HF version: vLLM batches across the whole chunk
N_BRANCHES = 5
N_RESTARTS = 25
GEN_MAX_NEW_TOKENS = 1500
CONT_MAX_NEW_TOKENS = 1300
ANSWER_MARKERS = ["Final Answer", "\\boxed", "####"]
FORMAT_ONLY = re.compile(r"^[\s\W_]*$")  # whitespace / punctuation / markdown, no letters or digits


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

probe = json.load(open(PROBE_PATH))
PROBE_LAYER = probe["layer"]
llm = make_llm(gpu_memory_utilization=0.6)  # leaves room for the HF scoring model below
hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
p_mean = torch.tensor(probe["scaler_mean"], dtype=torch.float32, device="cuda")
p_scale = torch.tensor(probe["scaler_scale"], dtype=torch.float32, device="cuda")
p_coef = torch.tensor(probe["coef"], dtype=torch.float32, device="cuda")


def claim_chunk():
    """Atomically take the next CHUNK problem positions from the shared counter."""
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
    """Token index where the final-answer line begins (len if no marker)."""
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


def trace_signals(pids, gen):
    """One HF forward pass: probe z, local entropy (bits) and top-N_BRANCHES
    alternatives at every generated position. Position i uses the hidden state
    at base+i-1, i.e. the distribution the model chose gen[i] from."""
    base, n = len(pids), len(gen)
    with torch.no_grad():
        out = hf(input_ids=torch.tensor([pids + gen], device="cuda"), output_hidden_states=True)
    hs = out.hidden_states[PROBE_LAYER][0, base - 1: base + n - 1].float()
    pred = ((hs - p_mean) / p_scale) @ p_coef + probe["intercept"]
    ent, topi = [], []
    for s in range(0, n, 256):  # chunked: full-vocab softmax over 1500 positions is ~1 GB in fp32
        logp = torch.log_softmax(out.logits[0, base - 1 + s: base - 1 + min(n, s + 256)].float(), -1)
        ent.append(-(logp.exp() * logp).sum(-1) / math.log(2))
        topi.append(logp.topk(N_BRANCHES, -1).indices)
    pred = pred.cpu().numpy()
    del out
    return pred, torch.cat(ent).cpu().numpy(), torch.cat(topi).cpu().tolist()


def segments(lo, hi, k):
    return [(lo + (hi - lo) * s // k, lo + (hi - lo) * (s + 1) // k) for s in range(k)]


def stratified(score_arr, lo, hi, k, allowed=None):
    """Top score per equal segment of [lo, hi), with a min gap so picks don't
    land either side of a segment boundary."""
    min_gap = max(1, (hi - lo) // (k * 4))
    picks = []
    for a, b in segments(lo, hi, k):
        if b <= a:
            continue
        seg = [i for i in range(a, b) if all(abs(i - p) >= min_gap for p in picks)] or list(range(a, b))
        cand = [i for i in seg if allowed is None or allowed[i]] or seg
        picks.append(max(cand, key=lambda i: score_arr[i]))
    return picks


def guarded_sample(product, topk_ids, line_of, lo, hi, k, rng):
    """Sample k cuts in proportion to the product, at most one per line, never at
    a formatting-only fork (all top alternatives are whitespace/punct/markdown)."""
    cand = []
    for i in range(lo, hi):
        alts = [tokenizer.decode([t]) for t in topk_ids[i][:4]]
        if sum(not FORMAT_ONLY.match(a) for a in alts) >= 2 and product[i] > 0:
            cand.append(i)
    picks, used_lines = [], set()
    while cand and len(picks) < k:
        w = np.array([product[i] for i in cand])
        i = cand[int(rng.choice(len(cand), p=w / w.sum()))]
        picks.append(i)
        used_lines.add(line_of[i])
        cand = [j for j in cand if line_of[j] not in used_lines]
    return picks


def zs(a):
    return (a - a.mean()) / (a.std() + 1e-8)


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

        jobs = []  # per scored problem: everything needed once generation returns
        for idx, ex, pids, gen in zip(idxs, examples, prompts, traces):
            ref = ex["reference_answer"]
            if score(tokenizer.decode(gen, skip_special_tokens=True), ref)[1] or len(gen) < 12:
                n_skipped += 1
                continue
            n_pos = len(gen)
            pred, local, topk_ids = trace_signals(pids, gen)
            z = zs(pred)
            product = np.maximum(pred, 0) * local
            zsum = zs(pred) + zs(local)
            tok_texts = [tokenizer.decode([t]) for t in gen]
            is_digit = [any(c.isdigit() for c in t) for t in tok_texts]
            line_of, line = [], 0
            for t in tok_texts:
                line_of.append(line)
                line += t.count("\n")
            ans_start = answer_start(gen)
            hi = max(2, min(ans_start, n_pos - 1))  # never past the answer marker or onto EOS
            rng = np.random.default_rng(idx)
            not_digit = [not d for d in is_digit]
            uniform = sorted(set(int(i * (n_pos - 1) / 4) for i in range(5)))
            arms = {
                "uniform": uniform,
                "restart_uniform_interior": sorted({0} | {hi * (i + 1) // 5 for i in range(4)}),
                "restart_stratified_probe": sorted({0} | set(stratified(z, 1, hi, 4, not_digit))),
                "restart_local_only": sorted({0} | set(stratified(local, 1, hi, 4))),
                "restart_product": sorted({0} | set(stratified(product, 1, hi, 4))),
                "restart_zsum": sorted({0} | set(stratified(zsum, 1, hi, 4))),
                "restart_product_guarded": sorted({0} | set(guarded_sample(product, topk_ids, line_of, 1, hi, 4, rng))),
            }
            cuts = sorted(set(p for pts in arms.values() for p in pts))
            jobs.append(dict(idx=idx, ref=ref, pids=pids, gen=gen, n_pos=n_pos, ans_start=ans_start, arms=arms,
                             cuts=cuts, topk_ids=topk_ids, uniform=uniform, z=z, local=local, product=product))

        if jobs:
            # one big generate per kind, across every problem in the chunk
            forced_in, forced_owner = [], []
            resamp_in, resamp_seeds, resamp_owner = [], [], []
            restart_in, restart_seeds = [], []
            for j, job in enumerate(jobs):
                for cut in job["cuts"]:
                    for t in job["topk_ids"][cut]:
                        forced_in.append(job["pids"] + job["gen"][:cut] + [t])  # forced token REPLACES gen[cut]
                        forced_owner.append((j, cut, t))
                for cut in job["uniform"]:
                    for b in range(N_BRANCHES):
                        resamp_in.append(job["pids"] + job["gen"][:cut])
                        resamp_seeds.append(job["idx"] * 1000 + cut * 7 + b)
                        resamp_owner.append((j, cut))
                restart_in += [job["pids"]] * N_RESTARTS
                restart_seeds += [job["idx"] * 1000 + 500 + b for b in range(N_RESTARTS)]
            forced_out = generate(llm, forced_in, CONT_MAX_NEW_TOKENS, sample=False)
            resamp_out = generate(llm, resamp_in, CONT_MAX_NEW_TOKENS, sample=True, seeds=resamp_seeds)
            restart_out = generate(llm, restart_in, GEN_MAX_NEW_TOKENS, sample=True, seeds=restart_seeds)

            for job in jobs:
                job["forced"] = {c: {"tokens": [], "answers": [], "correct": [], "lengths": []} for c in job["cuts"]}
                job["resampled"] = {c: {"answers": [], "correct": [], "lengths": []} for c in job["uniform"]}
                job["restarts"] = []
            for (j, cut, t), row in zip(forced_owner, forced_out):
                a, ok = score(tokenizer.decode([t] + row, skip_special_tokens=True), jobs[j]["ref"])
                d = jobs[j]["forced"][cut]
                d["tokens"].append(t); d["answers"].append(a); d["correct"].append(ok); d["lengths"].append(len(row) + 1)
            for (j, cut), row in zip(resamp_owner, resamp_out):
                a, ok = score(tokenizer.decode(row, skip_special_tokens=True), jobs[j]["ref"])
                d = jobs[j]["resampled"][cut]
                d["answers"].append(a); d["correct"].append(ok); d["lengths"].append(len(row))
            for k, row in enumerate(restart_out):
                job = jobs[k // N_RESTARTS]
                a, ok = score(tokenizer.decode(row, skip_special_tokens=True), job["ref"])
                job["restarts"].append({"answer": a, "correct": ok, "length": len(row)})

            for job in jobs:
                def arm_result(pts, table):
                    ok = {p: any(table[p]["correct"]) for p in pts}
                    return {"points": pts, "rescued": any(ok.values()),
                            "rescued_excl_answer_line": any(v for p, v in ok.items() if p < job["ans_start"]),
                            "tokens": sum(sum(table[p]["lengths"]) for p in pts)}
                results = {name: arm_result(pts, job["forced"]) for name, pts in job["arms"].items()}
                results["resample_uniform_cuts"] = arm_result(job["uniform"], job["resampled"])
                rs = job["restarts"]
                results["sampled_restarts"] = {"rescued": any(r["correct"] for r in rs),
                                               "rescued_first5": any(r["correct"] for r in rs[:5]),
                                               "tokens": sum(r["length"] for r in rs)}
                f.write(json.dumps({
                    "math_test_index": job["idx"], "reference_answer": job["ref"], "trace_len": job["n_pos"],
                    "answer_start": job["ans_start"], "prompt_token_ids": job["pids"],
                    "generated_token_ids": job["gen"],
                    "probe_z": [round(float(v), 3) for v in job["z"]],
                    "local_entropy": [round(float(v), 3) for v in job["local"]],
                    "forced": {str(k): v for k, v in job["forced"].items()},
                    "resampled": {str(k): v for k, v in job["resampled"].items()},
                    "restarts": rs, "results": results,
                }) + "\n")
                n_done += 1
            f.flush()
        el = time.time() - t_start
        print(f"[claimed up to {start + len(idxs)}/{len(order)}] scored={n_done} skipped={n_skipped} "
              f"elapsed={el / 60:.1f}min", flush=True)

print(f"\nDone. {n_done} scored, {n_skipped} skipped. Results in {OUT_PATH}")
