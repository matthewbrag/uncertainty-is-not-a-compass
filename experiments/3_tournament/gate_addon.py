"""
Synonym/divergence gate, added post hoc to the vLLM 9-arm tournament's saved
traces (advanced_vllm_w*.jsonl), integer-answer problems only (the decimal
stop-rule bug can't apply there, so new branches -- generated with the fixed
rule -- compare fairly with the old ones).

A real fork needs the model to be uncertain (local entropy), the outcome to
depend on the choice (probe), AND the alternatives to actually lead somewhere
different. Two cheap estimates of that last part, per candidate position,
over the top alternatives other than the trace's own token:
  next-step divergence: probability-weighted Jensen-Shannon divergence between
      the model's next-token distributions after each alternative (top-20 support)
  lookahead divergence: probability-weighted share of alternatives whose
      16-token greedy lookahead does NOT rejoin the original continuation
Candidates: per quarter of the trace, top-8 by product and top-8 by local
entropy. New arms (token-0 restart + one pick per quarter, like the tournament):
  restart_product_x_nextdiv, restart_product_x_lookdiv, restart_local_x_lookdiv
Cuts already forced in the tournament reuse their branches; new cuts are forced
exactly as there (top-5 tokens, greedy, 1300-token cap).

Usage (.venv-vllm): CUDA_VISIBLE_DEVICES=<gpu> python experiments/3_tournament/gate_addon.py <queue_file> <out_path>
"""
import fcntl
import glob
import json
import math
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from forkprobe.stop_rules import extract_answer
from forkprobe.vllm_gen import MODEL, make_llm, generate, _tok as tokenizer

QUEUE, OUT_PATH = sys.argv[1], sys.argv[2]
CHUNK, N_PER_SEG, LOOK, CONT_MAX = 8, 8, 16, 1300

R = [json.loads(l) for p in sorted(glob.glob("data/advanced_vllm_w*.jsonl")) for l in open(p)]
R = [r for r in R if float(r["reference_answer"]).is_integer()]
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


@torch.no_grad()
def topk_all(pids, gen, k=5):
    base, n = len(pids), len(gen)
    logits = hf(input_ids=torch.tensor([pids + gen], device="cuda"), logits_to_keep=n + 1).logits[0, :n]
    p = torch.softmax(logits.float(), -1)
    v, i = p.topk(k, -1)
    return i.cpu().tolist(), v.cpu().tolist()


def segments(lo, hi, k=4):
    return [(lo + (hi - lo) * s // k, lo + (hi - lo) * (s + 1) // k) for s in range(k)]


def zs(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / (a.std() + 1e-8)


norm = lambda s: re.sub(r"\\text|[\s*#_{}$\\-]", "", s)


def rejoins(la_ids, gen, cut):
    if len(la_ids) < 6:
        return False
    tail = norm(tokenizer.decode(la_ids[-8:], skip_special_tokens=True))
    window = norm(tokenizer.decode(gen[cut: cut + LOOK + 48], skip_special_tokens=True))
    return len(tail) > 4 and tail in window


def js(dists, weights):
    """Weighted Jensen-Shannon divergence (bits) of sparse distributions {token: p}."""
    keys = set().union(*dists)
    mix = {t: sum(w * d.get(t, 0.0) for d, w in zip(dists, weights)) for t in keys}
    h = lambda d: -sum(p * math.log2(p) for p in d.values() if p > 0)
    return h(mix) - sum(w * h(d) for d, w in zip(dists, weights))


def score(text, ref):
    a = extract_answer(text)
    return a, (a is not None and abs(a - ref) < 1e-4)


t0 = time.time()
n_done = 0
with open(OUT_PATH, "w") as f:
    while True:
        start = claim_chunk()
        if start >= len(R):
            break
        jobs = []
        for r in R[start:start + CHUNK]:
            pids, gen = r["prompt_token_ids"], r["generated_token_ids"]
            n = len(gen)
            hi = max(2, min(r["answer_start"], n - 1))
            z, le = np.array(r["probe_z"]), np.array(r["local_entropy"])
            product = np.maximum(z, 0) * le
            topi, topv = topk_all(pids, gen)
            cand_by_seg = []
            for a, b in segments(1, hi):
                if b <= a:
                    cand_by_seg.append([])
                    continue
                rng = range(a, b)
                c = set(sorted(rng, key=lambda i: -product[i])[:N_PER_SEG]) | set(sorted(rng, key=lambda i: -le[i])[:N_PER_SEG])
                cand_by_seg.append(sorted(c))
            cands = sorted({c for seg in cand_by_seg for c in seg})
            alts = {c: [(t, p) for t, p in zip(topi[c], topv[c]) if t != gen[c]][:4] for c in cands}
            jobs.append(dict(r=r, pids=pids, gen=gen, product=product, le=le, cand_by_seg=cand_by_seg,
                             cands=cands, alts=alts, topi=topi))
        # gate signals for every (candidate, alternative): one-step logprobs + a 16-token lookahead
        items = [(j, c, t, p) for j, job in enumerate(jobs) for c in job["cands"] for t, p in job["alts"][c]]
        prompts = [TokensPrompt(prompt_token_ids=jobs[j]["pids"] + jobs[j]["gen"][:c] + [t]) for j, c, t, _ in items]
        step = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0, logprobs=20, detokenize=False),
                            use_tqdm=False)
        look = generate(llm, [jobs[j]["pids"] + jobs[j]["gen"][:c] + [t] for j, c, t, _ in items], LOOK, sample=False)
        sig = {}
        for (j, c, t, p), o, la in zip(items, step, look):
            lp = o.outputs[0].logprobs[0]
            d = {tok: math.exp(x.logprob) for tok, x in lp.items()}
            s = sum(d.values()) or 1.0
            sig.setdefault((j, c), []).append(({k: v / s for k, v in d.items()}, p, not rejoins(la, jobs[j]["gen"], c)))
        for j, job in enumerate(jobs):
            job["nextdiv"], job["lookdiv"] = {}, {}
            for c in job["cands"]:
                entries = sig.get((j, c), [])
                if len(entries) < 2:
                    job["nextdiv"][c], job["lookdiv"][c] = 0.0, 0.0
                    continue
                w = np.array([p for _, p, _ in entries]); w = w / w.sum()
                job["nextdiv"][c] = js([d for d, _, _ in entries], list(w))
                job["lookdiv"][c] = float(sum(wi * dv for wi, (_, _, dv) in zip(w, entries)))
            arms = {
                "restart_product_x_nextdiv": lambda c, job=job: job["product"][c] * job["nextdiv"][c],
                "restart_product_x_lookdiv": lambda c, job=job: job["product"][c] * job["lookdiv"][c],
                "restart_local_x_lookdiv": lambda c, job=job: job["le"][c] * job["lookdiv"][c],
            }
            job["picks"] = {}
            for name, sc in arms.items():
                picks = [0]
                for seg in job["cand_by_seg"]:
                    if seg:
                        picks.append(max(seg, key=sc))
                job["picks"][name] = sorted(set(picks))
        # force any cut the tournament didn't already run
        todo, owner = [], []
        for j, job in enumerate(jobs):
            have = set(job["r"]["forced"])
            for c in sorted({c for ps in job["picks"].values() for c in ps}):
                if str(c) not in have:
                    for t in job["topi"][c]:
                        todo.append(job["pids"] + job["gen"][:c] + [t])
                        owner.append((j, c, t))
        outs = generate(llm, todo, CONT_MAX, sample=False) if todo else []
        new = {}
        for (j, c, t), row in zip(owner, outs):
            a, ok = score(tokenizer.decode([t] + row, skip_special_tokens=True), jobs[j]["r"]["reference_answer"])
            d = new.setdefault((j, c), {"tokens": [], "answers": [], "correct": [], "lengths": []})
            d["tokens"].append(t); d["answers"].append(a); d["correct"].append(ok); d["lengths"].append(len(row) + 1)
        for j, job in enumerate(jobs):
            forced = {**job["r"]["forced"], **{str(c): d for (jj, c), d in new.items() if jj == j}}
            results = {}
            for name, pts in job["picks"].items():
                ok = [any(forced[str(p)]["correct"]) for p in pts]
                results[name] = {"points": pts, "rescued": any(ok),
                                 "tokens": sum(sum(forced[str(p)]["lengths"]) for p in pts)}
            f.write(json.dumps({"math_test_index": job["r"]["math_test_index"], "results": results,
                                "forced_new": {str(c): d for (jj, c), d in new.items() if jj == j},
                                "gate": {str(c): [round(job["nextdiv"][c], 4), round(job["lookdiv"][c], 3)] for c in job["cands"]}}) + "\n")
            n_done += 1
        f.flush()
        print(f"[{start + len(jobs)}/{len(R)}] done={n_done} new_branches={len(todo)} elapsed={(time.time() - t0) / 60:.1f}min", flush=True)
print("Done.", n_done)
