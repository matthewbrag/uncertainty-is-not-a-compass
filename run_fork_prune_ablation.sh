#!/usr/bin/env bash
# 2x2 pruning ablation on all 829, then resumes the 240-problem fork-detection extension.
set -uo pipefail
cd /home/matthew/projects/ml-experiments/tree2
export HF_HUB_OFFLINE=1
LOG=fixed_pipeline.log
say() { echo "[$(date '+%a %H:%M')] $*" | tee -a "$LOG"; }
echo 0 > fork_ablation_queue.txt
say "PRUNE ABLATION: starting (2x2, all 829)"
pids=()
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g .venv-vllm/bin/python fork_prune_ablation.py entropy_probe_1.7b_diversity_level4_best_fixed.json \
    fork_ablation_queue.txt fork_ablation_w$g.jsonl > fork_ablation_gpu$g.log 2>&1 &
  pids+=($!)
done
ok=1; for p in "${pids[@]}"; do wait "$p" || ok=0; done
[ $ok = 1 ] && say "PRUNE ABLATION: done" || say "PRUNE ABLATION: FAILED -- see fork_ablation_gpu*.log"

echo 0 > fork_det2_queue.txt
say "FORK-DETECTION BENCHMARK 2: resuming (240 problems, offset 120)"
pids=()
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g .venv-vllm/bin/python fork_detection_benchmark.py entropy_probe_1.7b_diversity_level4_best_fixed.json \
    fork_det2_queue.txt fork_det2_w$g.jsonl 240 120 > fork_det2_gpu$g.log 2>&1 &
  pids+=($!)
done
ok=1; for p in "${pids[@]}"; do wait "$p" || ok=0; done
[ $ok = 1 ] && say "FORK-DETECTION BENCHMARK 2: done" || say "FORK-DETECTION BENCHMARK 2: FAILED -- see fork_det2_gpu*.log"
