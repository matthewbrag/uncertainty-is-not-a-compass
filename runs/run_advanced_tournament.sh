#!/usr/bin/env bash
# Advanced-strategy tournament (MATH-L4, all 829 test problems) on 3 GPUs
# pulling from one shared work queue, then resumes the paused Part 1
# re-collection queue so the GPUs stay busy.
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
source .venv/bin/activate
export HF_HUB_OFFLINE=1
LOG=data/fixed_pipeline.log
say() { echo "[$(date '+%a %H:%M')] $*" | tee -a "$LOG"; }

echo 0 > data/advanced_queue.txt
say "ADVANCED TOURNAMENT: starting (3 workers, shared queue)"
pids=()
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g python legacy/advanced_tournament.py probes/entropy_probe_1.7b_diversity_level4_best_fixed.json \
    data/advanced_queue.txt data/advanced_tournament_w$g.jsonl > data/advanced_tournament_gpu$g.log 2>&1 &
  pids+=($!)
done
ok=1; for p in "${pids[@]}"; do wait "$p" || ok=0; done
[ $ok = 1 ] && say "ADVANCED TOURNAMENT: done" || say "ADVANCED TOURNAMENT: FAILED on >=1 worker -- see data/advanced_tournament_gpu*.log"

say "resuming follow-up re-collection queue"
./run_fixed_followups.sh
