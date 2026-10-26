#!/usr/bin/env bash
# Waits for the fork-prune tournament, then extends the fork-detection
# benchmark with 240 new problems (offset 120) to tighten the top-pick lifts.
set -uo pipefail
cd /home/matthew/projects/ml-experiments/tree2
export HF_HUB_OFFLINE=1
LOG=fixed_pipeline.log
say() { echo "[$(date '+%a %H:%M')] $*" | tee -a "$LOG"; }
until grep -q "FORK-PRUNE TOURNAMENT: \(done\|FAILED\)" "$LOG"; do sleep 30; done
echo 0 > fork_det2_queue.txt
say "FORK-DETECTION BENCHMARK 2: starting (240 problems, offset 120)"
pids=()
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g .venv-vllm/bin/python fork_detection_benchmark.py entropy_probe_1.7b_diversity_level4_best_fixed.json \
    fork_det2_queue.txt fork_det2_w$g.jsonl 240 120 > fork_det2_gpu$g.log 2>&1 &
  pids+=($!)
done
ok=1; for p in "${pids[@]}"; do wait "$p" || ok=0; done
[ $ok = 1 ] && say "FORK-DETECTION BENCHMARK 2: done" || say "FORK-DETECTION BENCHMARK 2: FAILED -- see fork_det2_gpu*.log"
