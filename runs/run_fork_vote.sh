#!/usr/bin/env bash
# Fork-vote tournament (majority vote vs compute-matched best-of-N), 3 GPUs, shared queue.
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1
LOG=data/fixed_pipeline.log
say() { echo "[$(date '+%a %H:%M')] $*" | tee -a "$LOG"; }
echo 0 > data/fork_vote_queue.txt
say "FORK-VOTE TOURNAMENT: starting"
pids=()
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g .venv-vllm/bin/python experiments/3_tournament/fork_vote_tournament.py data/fork_vote_queue.txt data/fork_vote_w$g.jsonl > data/fork_vote_gpu$g.log 2>&1 &
  pids+=($!)
done
ok=1; for p in "${pids[@]}"; do wait "$p" || ok=0; done
[ $ok = 1 ] && say "FORK-VOTE TOURNAMENT: done" || say "FORK-VOTE TOURNAMENT: FAILED on >=1 worker -- see data/fork_vote_gpu*.log"
