"""
Race: fresh resampling vs a resample-then-branch hybrid, at equal tokens.

Per problem (the 250 integer-answer problems of the vLLM tournament, whose
first trace was wrong): 16 fresh samples. For each of the first 4 that is
wrong, pick one fork three ways and force the 4 top non-original tokens there
(greedy continuations, like the tournament):
  gated : top-16 by product (max(0, z(probe)) x local entropy), re-ranked by
          lookahead divergence (prob-weighted share of 16-token greedy
          lookaheads that do NOT rejoin the sample); lookahead tokens are
          recorded so the analysis charges them
  local : top local entropy
  random: uniform interior position
The analysis (race_analyze.py) replays policies over this shared pool, so all
comparisons are paired.

Usage (.venv-vllm): CUDA_VISIBLE_DEVICES=<gpu> python experiments/5_race/race.py <probe.json> <queue_file> <out_path> [sampled]
  sampled: fork branches are sampled like the fresh samples (controls for greedy decoding)
"""
import fcntl
import glob
import json
import math
import random
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

PROBE_PATH, QUEUE, OUT_PATH = sys.argv[1], sys.argv[2], sys.argv[3]
SAMPLED_BRANCHES = len(sys.argv) > 4 and sys.argv[4] == "sampled"
CHUNK, N_FRESH, N_FORKED, N_CAND, LOOK = 8, 16, 4, 16, 16
GEN_MAX, CONT_MAX = 1500, 1300
ANSWER_MARKERS = ["Final Answer", "\\boxed", "####"]

R = [json.loads(l) for p in sorted(glob.glob("data/advanced_vllm_w*.jsonl")) for l in open(p)]
R = [r for r in R if float(r["reference_answer"]).is_integer()]
probe = json.load(open(PROBE_PATH))
llm = make_llm(gpu_memory_utilization=0.6)
hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
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
    """Per generated token: probe prediction, local entropy (bits), top-5 (id, prob)."""
    base, n = len(pids), len(gen)
    out = hf(input_ids=torch.tensor([pids + gen], device="cuda"), output_hidden_states=True)
    hs = out.hidden_states[probe["layer"]][0, base - 1: base + n - 1].float()
    pp = (((hs - p_mean) / p_scale) @ p_coef + probe["intercept"]).cpu().numpy()
    ent, top5 = [], []
    for s in range(0, n, 256):
        logp = torch.log_softmax(out.logits[0, base - 1 + s: base - 1 + min(n, s + 256)].float(), -1)
        ent.append(-(logp.exp() * logp).sum(-1) / math.log(2))
        v, i = logp.exp().topk(5, -1)
        top5 += [list(zip(ii, vv)) for ii, vv in zip(i.tolist(), v.tolist())]
    del out
    return pp, torch.cat(ent).cpu().numpy(), top5


norm = lambda s: re.sub(r"\\text|[\s*#_{}$\\-]", "", s)


def rejoins(la_ids, gen, cut):
    if len(la_ids) < 6:
        return False
    tail = norm(tokenizer.decode(la_ids[-8:], skip_special_tokens=True))
    window = norm(tokenizer.decode(gen[cut: cut + LOOK + 48], skip_special_tokens=True))
    return len(tail) > 4 and tail in window


def correct(text, ref):
    a = extract_answer(text)
    return a is not None and abs(a - ref) < 1e-4


t0 = time.time()
n_done = 0
with open(OUT_PATH, "w") as f:
    while True:
        start = claim_chunk()
        if start >= len(R):
            break
        recs = R[start:start + CHUNK]
        fresh = generate(llm, [r["prompt_token_ids"] for r in recs for _ in range(N_FRESH)], GEN_MAX, sample=True,
                         seeds=[r["math_test_index"] * 1009 + 500 + k for r in recs for k in range(N_FRESH)])
        probs = []
        for i, r in enumerate(recs):
            samples = []
            for k, gen in enumerate(fresh[i * N_FRESH:(i + 1) * N_FRESH]):
                ok = correct(tokenizer.decode(gen, skip_special_tokens=True), r["reference_answer"])
                samples.append(dict(gen=gen, length=len(gen), correct=ok, forks={}))
            probs.append(dict(r=r, samples=samples))
        # pick forks on the first N_FORKED samples that are wrong
        look_items = []
        for i, pr in enumerate(probs):
            pids = pr["r"]["prompt_token_ids"]
            for k, s in enumerate(pr["samples"][:N_FORKED]):
                gen = s["gen"]
                if s["correct"] or len(gen) < 20:
                    continue
                hi = max(2, min(answer_start(gen), len(gen) - 1))
                pp, le, top5 = signals(pids, gen)
                z = (pp - pp.mean()) / (pp.std() + 1e-8)
                prod = np.maximum(z, 0) * le
                pool = sorted(range(1, hi), key=lambda t: -prod[t])[:N_CAND]
                s["_sig"] = dict(prod=prod, le=le, top5=top5, hi=hi, pool=pool)
                for t in pool:
                    for tk, p in [(tk, p) for tk, p in top5[t] if tk != gen[t]][:4]:
                        look_items.append((i, k, t, tk, p))
        looks = generate(llm, [probs[i]["r"]["prompt_token_ids"] + probs[i]["samples"][k]["gen"][:t] + [tk]
                               for i, k, t, tk, _ in look_items], LOOK, sample=False) if look_items else []
        gate = {}
        for (i, k, t, tk, p), la in zip(look_items, looks):
            g = gate.setdefault((i, k), {"tokens": 0, "div": {}})
            g["tokens"] += len(la)
            g["div"].setdefault(t, []).append((tk, p, not rejoins(la, probs[i]["samples"][k]["gen"], t)))
        todo, owner = [], []
        for i, pr in enumerate(probs):
            for k, s in enumerate(pr["samples"]):
                if "_sig" not in s:
                    continue
                sg, gen, g = s["_sig"], s["gen"], gate[(i, k)]
                lookdiv = {t: (sum(p * d for _, p, d in e) / max(1e-9, sum(p for _, p, _ in e))) for t, e in g["div"].items()}
                picks = {
                    "gated": max(sg["pool"], key=lambda t: sg["prod"][t] * lookdiv.get(t, 0.0)),
                    "local": max(range(1, sg["hi"]), key=lambda t: sg["le"][t]),
                    "random": random.Random(pr["r"]["math_test_index"] * 31 + k).randrange(1, sg["hi"]),
                }
                for name, t in picks.items():
                    alts = [(tk, pb) for tk, pb in sg["top5"][t] if tk != gen[t]][:4]
                    diverges = {tk: d for tk, _, d in g["div"].get(t, [])}
                    s["forks"][name] = {"pos": t, "token": tokenizer.decode([gen[t]]), "branches": [],
                                        "gate_tokens": g["tokens"] if name == "gated" else 0,
                                        "local": round(float(sg["le"][t]), 4), "product": round(float(sg["prod"][t]), 4),
                                        "lookdiv": round(lookdiv[t], 4) if t in lookdiv else None}
                    for tk, pb in alts:
                        todo.append(pr["r"]["prompt_token_ids"] + gen[:t] + [tk])
                        owner.append((i, k, name, tk, diverges.get(tk), pb))
        outs = generate(llm, todo, CONT_MAX, sample=SAMPLED_BRANCHES,
                        seeds=[probs[i]["r"]["math_test_index"] * 7919 + k * 101 + j for j, (i, k, *_) in enumerate(owner)]
                        if SAMPLED_BRANCHES else None) if todo else []
        for (i, k, name, tk, dv, pb), row in zip(owner, outs):
            s = probs[i]["samples"][k]
            ok = correct(tokenizer.decode(s["gen"][:s["forks"][name]["pos"]] + [tk] + row, skip_special_tokens=True),
                         probs[i]["r"]["reference_answer"])
            s["forks"][name]["branches"].append({"length": len(row) + 1, "correct": ok, "diverges": dv, "prob": pb})
        for pr in probs:
            f.write(json.dumps({"math_test_index": pr["r"]["math_test_index"],
                                "reference_answer": pr["r"]["reference_answer"],
                                "samples": [{"length": s["length"], "correct": s["correct"], "forks": s["forks"]}
                                            for s in pr["samples"]]}) + "\n")
            n_done += 1
        f.flush()
        print(f"[{start + len(recs)}/{len(R)}] done={n_done} branches={len(todo)} lookaheads={len(look_items)} "
              f"elapsed={(time.time() - t0) / 60:.1f}min", flush=True)
print("Done.", n_done)
