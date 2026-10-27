#!/usr/bin/env bash
# Waits for the fork-prune tournament, then extends the fork-detection
# benchmark with 240 new problems (offset 120) to tighten the top-pick lifts.
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1
LOG=data/fixed_pipeline.log
say() { echo "[$(date '+%a %H:%M')] $*" | tee -a "$LOG"; }
until grep -q "FORK-PRUNE TOURNAMENT: \(done\|FAILED\)" "$LOG"; do sleep 30; done
echo 0 > data/fork_det2_queue.txt
say "FORK-DETECTION BENCHMARK 2: starting (240 problems, offset 120)"
pids=()
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g .venv-vllm/bin/python experiments/4_fork_detection/fork_detection_benchmark.py probes/entropy_probe_1.7b_diversity_level4_best_fixed.json \
    data/fork_det2_queue.txt data/fork_det2_w$g.jsonl 240 120 > data/fork_det2_gpu$g.log 2>&1 &
  pids+=($!)
done
ok=1; for p in "${pids[@]}"; do wait "$p" || ok=0; done
[ $ok = 1 ] && say "FORK-DETECTION BENCHMARK 2: done" || say "FORK-DETECTION BENCHMARK 2: FAILED -- see data/fork_det2_gpu*.log"
