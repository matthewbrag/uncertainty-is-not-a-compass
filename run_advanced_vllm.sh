#!/usr/bin/env bash
# vLLM advanced tournament (9 arms incl. local-entropy strategies), 3 GPUs on
# one shared queue; then resumes the paused Part 1 re-collection queue.
set -uo pipefail
cd /home/matthew/projects/ml-experiments/tree2
export HF_HUB_OFFLINE=1
LOG=fixed_pipeline.log
say() { echo "[$(date '+%a %H:%M')] $*" | tee -a "$LOG"; }

echo 0 > advanced_vllm_queue.txt
say "ADVANCED TOURNAMENT (vLLM): starting (3 workers, shared queue)"
pids=()
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g .venv-vllm/bin/python advanced_tournament_vllm.py entropy_probe_1.7b_diversity_level4_best_fixed.json \
    advanced_vllm_queue.txt advanced_vllm_w$g.jsonl > advanced_vllm_gpu$g.log 2>&1 &
  pids+=($!)
done
ok=1; for p in "${pids[@]}"; do wait "$p" || ok=0; done
[ $ok = 1 ] && say "ADVANCED TOURNAMENT (vLLM): done" || say "ADVANCED TOURNAMENT (vLLM): FAILED on >=1 worker -- see advanced_vllm_gpu*.log"

say "resuming follow-up re-collection queue"
./run_fixed_followups.sh
