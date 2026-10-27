"""
Before/after benchmark for the gpu-unlock tensor-core rate change on one GPU.
Run once on the stock driver and once with the unlock active, same GPU and
same label scheme; compare with plot_unlock_benchmark.py.

  1. GEMM: bf16 / fp16 matmul TFLOPS (8192^3) -- confirms whether the unlock
     is actually active (stock 3090 BF16 is ~70-80 TFLOPS)
  2. decode sweep: batch 1 -> 512, 128-token prompts, exactly 256 generated
     tokens each (EOS ignored); output tokens/sec. Small batches are
     memory-bandwidth-bound and shouldn't move; large batches push the weight
     matmuls toward compute-bound, where the unlock should show up.
  3. prefill sweep: 1024-token prompts, 1 output token; prompt tokens/sec
GPU SM clock, power and temperature are sampled during every measurement, so
a power cap or thermal throttle shows up next to the throughput it explains.

Usage (.venv-vllm): CUDA_VISIBLE_DEVICES=<gpu> python unlock_benchmark.py <label> <out_json>
e.g. CUDA_VISIBLE_DEVICES=0 python unlock_benchmark.py stock unlock_bench_stock_gpu0.json
"""
import json
import subprocess
import sys
import threading
import time

import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

LABEL, OUT = sys.argv[1], sys.argv[2]
DECODE_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
PREFILL_BATCHES = [1, 4, 16, 64]
DECODE_PROMPT, DECODE_GEN, PREFILL_PROMPT = 128, 256, 1024
GPU_INDEX = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]


class GpuSampler:
    """Samples SM clock / power / temperature / utilization every 0.25 s."""

    def __init__(self):
        self.samples, self._stop = [], threading.Event()

    def _run(self):
        q = "clocks.sm,power.draw,temperature.gpu,utilization.gpu"
        while not self._stop.is_set():
            out = subprocess.run(["nvidia-smi", "-i", GPU_INDEX, f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.strip()
            try:
                self.samples.append([float(x) for x in out.split(",")])
            except ValueError:
                pass
            time.sleep(0.25)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()

    def summary(self):
        if not self.samples:
            return {}
        cols = list(zip(*self.samples))
        return {k: round(sum(c) / len(c), 1) for k, c in zip(["sm_clock_mhz", "power_w", "temp_c", "util_pct"], cols)}


def gemm_tflops(dtype, n=8192, iters=50):
    a = torch.randn(n, n, dtype=dtype, device="cuda")
    b = torch.randn(n, n, dtype=dtype, device="cuda")
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    with GpuSampler() as s:
        t = time.time()
        for _ in range(iters):
            a @ b
        torch.cuda.synchronize()
        dt = time.time() - t
    del a, b
    torch.cuda.empty_cache()
    return {"tflops": round(2 * n ** 3 * iters / dt / 1e12, 1), **s.summary()}


results = {"label": LABEL, "gpu_index": GPU_INDEX,
           "gpu": subprocess.run(["nvidia-smi", "-i", GPU_INDEX, "--query-gpu=name,driver_version,power.limit",
                                  "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
           "torch": torch.__version__}
print(results["gpu"], flush=True)
results["gemm"] = {"bf16": gemm_tflops(torch.bfloat16), "fp16": gemm_tflops(torch.float16)}
print("GEMM:", results["gemm"], flush=True)

llm = LLM(model="Qwen/Qwen3-1.7B", dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=2048,
          max_num_seqs=max(DECODE_BATCHES), enable_prefix_caching=False, seed=0)
vocab = llm.get_tokenizer().vocab_size
rng = torch.Generator().manual_seed(0)


def prompts(batch, length):
    # random token ids: no shared prefixes, so prefix caching can't flatter the numbers
    return [TokensPrompt(prompt_token_ids=torch.randint(1000, vocab - 1000, (length,), generator=rng).tolist())
            for _ in range(batch)]


def timed(batch, plen, gen):
    params = SamplingParams(max_tokens=gen, ignore_eos=True, temperature=0.0, detokenize=False)
    llm.generate(prompts(min(batch, 8), plen), params, use_tqdm=False)  # warmup
    p = prompts(batch, plen)
    with GpuSampler() as s:
        t = time.time()
        outs = llm.generate(p, params, use_tqdm=False)
        dt = time.time() - t
    return sum(len(o.outputs[0].token_ids) for o in outs), dt, s.summary()


results["decode"] = []
for b in DECODE_BATCHES:
    n_out, dt, gpu = timed(b, DECODE_PROMPT, DECODE_GEN)
    row = {"batch": b, "output_tokens_per_s": round(n_out / dt, 1), "seconds": round(dt, 2), **gpu}
    results["decode"].append(row)
    print("decode", row, flush=True)

results["prefill"] = []
for b in PREFILL_BATCHES:
    _, dt, gpu = timed(b, PREFILL_PROMPT, 1)
    row = {"batch": b, "prompt_tokens_per_s": round(b * PREFILL_PROMPT / dt, 1), "seconds": round(dt, 3), **gpu}
    results["prefill"].append(row)
    print("prefill", row, flush=True)

json.dump(results, open(OUT, "w"), indent=1)
print("wrote", OUT)
