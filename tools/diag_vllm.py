"""Why do vLLM greedy branches disagree with HF? Compares lengths, cap hits,
and whether text continues past the '#### N' answer line, on a few problems'
recorded forced branches. Run in .venv-vllm."""
import glob
import json
import sys

from forkprobe.stop_rules import ANSWER_MARKER_RE, extract_answer
from forkprobe.vllm_gen import make_llm, generate, _tok as tok

recs = [json.loads(l) for f in sorted(glob.glob("data/advanced_tournament_w*.jsonl")) for l in open(f)][:4]
llm = make_llm(gpu_memory_utilization=0.85)
prompts, hf = [], []
for r in recs:
    for cut, d in r["forced"].items():
        for t, a, n in zip(d["tokens"], d["answers"], d["lengths"]):
            prompts.append(r["prompt_token_ids"] + r["generated_token_ids"][:int(cut)] + [t])
            hf.append((a, n, t))
rows = generate(llm, prompts, 1300, sample=False)
past, capped, longer, n_dis = 0, 0, 0, 0
examples = []
for (a_hf, n_hf, t), row in zip(hf, rows):
    text = tok.decode([t] + row, skip_special_tokens=True)
    a = extract_answer(text)
    ms = list(ANSWER_MARKER_RE.finditer(text))
    if ms and len(text) - ms[0].end() > 5:
        past += 1
    capped += len(row) >= 1300
    longer += (len(row) + 1) > n_hf + 5
    if a != a_hf:
        n_dis += 1
        if len(examples) < 3:
            examples.append((n_hf, len(row) + 1, a_hf, a, text[-300:]))
n = len(rows)
print(f"{n} branches | vLLM capped: {capped}  text continues >5 chars past first '####': {past}  "
      f"vLLM longer than HF by >5 tok: {longer}  answer disagreements: {n_dis}")
print("HF capped:", sum(n_hf >= 1300 for _, n_hf, _ in hf))
for e in examples:
    print(f"\n--- HF len {e[0]} ans {e[2]} | vLLM len {e[1]} ans {e[3]}\n...{e[4]!r}")
