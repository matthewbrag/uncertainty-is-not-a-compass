#!/usr/bin/env bash
# Waits for the fork-detection extension, then records stock-driver baselines on GPU0 and GPU1.
set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root
export HF_HUB_OFFLINE=1
LOG=data/fixed_pipeline.log
say() { echo "[$(date '+%a %H:%M')] $*" | tee -a "$LOG"; }
until grep -q "FORK-DETECTION BENCHMARK 2: \(done\|FAILED\)" "$LOG"; do sleep 30; done
say "UNLOCK BASELINE: starting (stock driver, GPU0 + GPU1)"
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g .venv-vllm/bin/python tools/unlock_benchmark/unlock_benchmark.py stock_gpu$g unlock_bench_stock_gpu$g.json > data/unlock_bench_stock_gpu$g.log 2>&1 &
done
wait
say "UNLOCK BASELINE: done"
