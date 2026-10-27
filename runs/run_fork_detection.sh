#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1
LOG=data/fixed_pipeline.log
say() { echo "[$(date '+%a %H:%M')] $*" | tee -a "$LOG"; }
echo 0 > data/fork_det_queue.txt
say "FORK-DETECTION BENCHMARK: starting (120 problems)"
pids=()
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g .venv-vllm/bin/python experiments/4_fork_detection/fork_detection_benchmark.py probes/entropy_probe_1.7b_diversity_level4_best_fixed.json \
    data/fork_det_queue.txt data/fork_det_w$g.jsonl 120 > data/fork_det_gpu$g.log 2>&1 &
  pids+=($!)
done
ok=1; for p in "${pids[@]}"; do wait "$p" || ok=0; done
[ $ok = 1 ] && say "FORK-DETECTION BENCHMARK: done" || say "FORK-DETECTION BENCHMARK: FAILED -- see data/fork_det_gpu*.log"
