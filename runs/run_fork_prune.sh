#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1
LOG=data/fixed_pipeline.log
say() { echo "[$(date '+%a %H:%M')] $*" | tee -a "$LOG"; }
echo 0 > data/fork_prune_queue.txt
say "FORK-PRUNE TOURNAMENT: starting (all 829)"
pids=()
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g .venv-vllm/bin/python experiments/3_tournament/fork_prune_tournament.py probes/entropy_probe_1.7b_diversity_level4_best_fixed.json \
    data/fork_prune_queue.txt data/fork_prune_w$g.jsonl > data/fork_prune_gpu$g.log 2>&1 &
  pids+=($!)
done
ok=1; for p in "${pids[@]}"; do wait "$p" || ok=0; done
[ $ok = 1 ] && say "FORK-PRUNE TOURNAMENT: done" || say "FORK-PRUNE TOURNAMENT: FAILED -- see data/fork_prune_gpu*.log"
