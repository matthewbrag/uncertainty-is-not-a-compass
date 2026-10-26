"""
Adds per-token local entropy (bits, next-token distribution at the position
that predicts each token) and the raw probe prediction (predicted outcome
diversity, bits) to debug_strategy_trace.py dumps, so the explorer can
highlight tokens by local entropy, the probe, or their combination.
Re-encodes the saved trace text and checks it round-trips to the same tokens.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python add_local_entropy.py <probe_path> <dump.json> [...]
"""
import json
import sys

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app import DEFAULT_TEMPLATE

# CPU by default: a handful of forward passes, and the GPUs are usually busy
# with tournament workers whose memory spikes would OOM if we squeezed in
DEVICE = os.environ.get("LOCAL_ENT_DEVICE", "cpu")
probe = json.load(open(sys.argv[1]))
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B", dtype=torch.float32 if DEVICE == "cpu" else torch.bfloat16).to(DEVICE).eval()
p_mean = torch.tensor(probe["scaler_mean"], device=DEVICE)
p_scale = torch.tensor(probe["scaler_scale"], device=DEVICE)
p_coef = torch.tensor(probe["coef"], device=DEVICE)

for path in sys.argv[2:]:
    dumps = json.load(open(path))
    for d in dumps:
        text = tok.apply_chat_template([{"role": "user", "content": DEFAULT_TEMPLATE.replace("{question}", d["problem"])}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pids = tok.encode(text, add_special_tokens=False)
        gen = tok.encode(d["trace"], add_special_tokens=False)
        if d["token_texts"][-1] == "<|im_end|>":
            gen = gen + [tok.convert_tokens_to_ids("<|im_end|>")]
        if [tok.decode([t]) for t in gen] != d["token_texts"]:
            print(f"#{d['test_index']}: re-encoded trace does not match saved tokens; skipping")
            continue
        base, n = len(pids), len(gen)
        with torch.no_grad():
            out = model(torch.tensor([pids + gen], device=DEVICE), output_hidden_states=True)
        logits = out.logits[0, base - 1: base + n - 1].float()
        logp = torch.log_softmax(logits, -1)
        ent = -(logp.exp() * logp).sum(-1) / torch.log(torch.tensor(2.0))
        hs = out.hidden_states[probe["layer"]][0, base - 1: base + n - 1].float()
        pred = ((hs - p_mean) / p_scale) @ p_coef + probe["intercept"]
        d["local_ent"] = [round(float(v), 3) for v in ent]
        d["probe_pred"] = [round(float(v), 3) for v in pred]
        # top-10 of the distribution that produced each token, for the explorer's logit view
        top_p, top_i = logp.exp().topk(10, dim=-1)
        d["topk"] = [[[tok.decode([int(t)]), round(float(p), 4)] for t, p in zip(ti, tp)]
                     for ti, tp in zip(top_i.tolist(), top_p.tolist())]
        print(f"#{d['test_index']}: {n} tokens, mean local entropy {ent.mean():.3f} bits, "
              f"{(ent < 0.01).float().mean():.0%} near-zero", flush=True)
        del out
    json.dump(dumps, open(path, "w"))
