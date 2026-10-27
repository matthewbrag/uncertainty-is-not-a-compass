#!/usr/bin/env bash
# vLLM advanced tournament (9 arms incl. local-entropy strategies), 3 GPUs on
# one shared queue; then resumes the paused Part 1 re-collection queue.
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1
LOG=data/fixed_pipeline.log
say() { echo "[$(date '+%a %H:%M')] $*" | tee -a "$LOG"; }

echo 0 > data/advanced_vllm_queue.txt
say "ADVANCED TOURNAMENT (vLLM): starting (3 workers, shared queue)"
pids=()
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g .venv-vllm/bin/python experiments/3_tournament/advanced_tournament_vllm.py probes/entropy_probe_1.7b_diversity_level4_best_fixed.json \
    data/advanced_vllm_queue.txt data/advanced_vllm_w$g.jsonl > data/advanced_vllm_gpu$g.log 2>&1 &
  pids+=($!)
done
ok=1; for p in "${pids[@]}"; do wait "$p" || ok=0; done
[ $ok = 1 ] && say "ADVANCED TOURNAMENT (vLLM): done" || say "ADVANCED TOURNAMENT (vLLM): FAILED on >=1 worker -- see data/advanced_vllm_gpu*.log"

say "resuming follow-up re-collection queue"
./run_fixed_followups.sh
