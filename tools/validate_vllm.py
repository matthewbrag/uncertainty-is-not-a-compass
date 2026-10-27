"""
Checks vllm_gen against HF on real work: re-runs the forced (greedy) branches
recorded in advanced_tournament_w*.jsonl through vLLM and compares answers
branch by branch, then times one problem's full workload (forced branches +
resampled branches + 25 restarts) so it can be compared with HF's ~4.5
GPU-minutes per scored problem.

Usage (in .venv-vllm): CUDA_VISIBLE_DEVICES=<gpu> python tools/validate_vllm.py <n_problems> <gpu_mem_fraction>
"""
import glob
import json
import sys
import time

from forkprobe.stop_rules import extract_answer
from forkprobe.vllm_gen import make_llm, generate, _tok as tok

N, MEM = int(sys.argv[1]), float(sys.argv[2])
recs = [json.loads(l) for f in sorted(glob.glob("data/advanced_tournament_w*.jsonl")) for l in open(f)][:N]
llm = make_llm(gpu_memory_utilization=MEM)

prompts, ref_rows = [], []
for r in recs:
    pids, gen = r["prompt_token_ids"], r["generated_token_ids"]
    for cut, d in r["forced"].items():
        for t, a, ok in zip(d["tokens"], d["answers"], d["correct"]):
            prompts.append(pids + gen[:int(cut)] + [t])
            ref_rows.append((r["reference_answer"], a, ok, t))
t0 = time.time()
rows = generate(llm, prompts, 1300, sample=False)
dt = time.time() - t0

same_ans = same_ok = 0
for (ref, a_hf, ok_hf, t), row in zip(ref_rows, rows):
    a = extract_answer(tok.decode([t] + row, skip_special_tokens=True))
    ok = a is not None and abs(a - ref) < 1e-4
    same_ans += (a == a_hf) or (a is not None and a_hf is not None and abs(a - a_hf) < 1e-4)
    same_ok += ok == ok_hf
n = len(rows)
gen_tok = sum(len(r) for r in rows)
print(f"forced branches: {n} from {len(recs)} problems in {dt:.1f}s  ({gen_tok / dt:,.0f} tok/s)")
print(f"answer agreement with HF: {same_ans / n:.1%}   correctness agreement: {same_ok / n:.1%}")
print(f"HF correct: {sum(x[2] for x in ref_rows)}   vLLM correct: "
      f"{sum(1 for (ref, *_), row, (_, _, _, t) in zip(ref_rows, rows, ref_rows) if (lambda a: a is not None and abs(a - ref) < 1e-4)(extract_answer(tok.decode([t] + row, skip_special_tokens=True))))}")

# full per-problem workload on one problem, like advanced_tournament does it
r = recs[0]
pids, gen = r["prompt_token_ids"], r["generated_token_ids"]
uniform = sorted(int(k) for k in r["resampled"])
t0 = time.time()
forced = [pids + gen[:int(c)] + [t] for c, d in r["forced"].items() for t in d["tokens"]]
resamp = [pids + gen[:c] for c in uniform for _ in range(5)]
restarts = [pids] * 25
generate(llm, forced, 1300, sample=False)
generate(llm, resamp, 1300, sample=True, seeds=range(len(resamp)))
generate(llm, restarts, 1500, sample=True, seeds=range(500, 525))
print(f"one problem's full workload ({len(forced)} forced + {len(resamp)} resampled + 25 restarts): "
      f"{time.time() - t0:.1f}s   (HF: ~270s per scored problem)")
