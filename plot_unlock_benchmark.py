"""Plot decode throughput vs batch size for two or more unlock_benchmark.py runs,
plus the after/before speedup per batch size.
Usage: python plot_unlock_benchmark.py out.png stock.json unlocked.json [...]"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out, runs = sys.argv[1], [json.load(open(p)) for p in sys.argv[2:]]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for r in runs:
    b = [d["batch"] for d in r["decode"]]
    ax1.plot(b, [d["output_tokens_per_s"] for d in r["decode"]], marker="o",
             label=f"{r['label']} (bf16 GEMM {r['gemm']['bf16']['tflops']} TFLOPS)")
ax1.set(xscale="log", yscale="log", xlabel="concurrent sequences (batch)", ylabel="decode output tokens / s",
        title="Qwen3-1.7B decode throughput, vLLM, one RTX 3090")
ax1.legend(fontsize=8)
base = runs[0]
for r in runs[1:]:
    sp = [d["output_tokens_per_s"] / b0["output_tokens_per_s"] for d, b0 in zip(r["decode"], base["decode"])]
    ax2.plot([d["batch"] for d in r["decode"]], sp, marker="o", label=f"{r['label']} / {base['label']}")
ax2.axhline(1.0, color="0.6", lw=1)
ax2.set(xscale="log", xlabel="concurrent sequences (batch)", ylabel="speedup", title="Decode speedup by batch size")
if len(runs) > 1:
    ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out, dpi=150)
for r in runs:
    print(r["label"], "| bf16 GEMM", r["gemm"]["bf16"]["tflops"], "TFLOPS | decode tok/s:",
          {d["batch"]: d["output_tokens_per_s"] for d in r["decode"]})
print("wrote", out)
