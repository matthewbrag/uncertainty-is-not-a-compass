#!/usr/bin/env bash
# Critical-path rerun after the off-by-one forcing fix (MATH-L4, 1.7B):
#   1. re-collect ground truth (900 train problems, 3 GPU shards)
#   2. retrain the layer-14 diversity probe on it
#   3. rerun the shared-trace tournament (829 test problems, 3 GPU shards)
# Every output uses a *_fixed name so nothing from the pre-fix runs is touched.
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
source .venv/bin/activate
export HF_HUB_OFFLINE=1
LOG=data/fixed_pipeline.log
say() { echo "[$(date '+%a %H:%M')] $*" | tee -a "$LOG"; }

say "STAGE 1: collection (3 x 300 problems)"
pids=()
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g python experiments/1_collect/math_collect_batched.py qwen3-1.7b 4 300 3000 $((g * 300)) \
    data/math_level4_qwen3-1.7b_fixed_shard$g.jsonl > data/collect_fixed_gpu$g.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || { say "FAILED: collection shard (pid $p) -- see data/collect_fixed_gpu*.log"; exit 1; }; done
cat math_level4_qwen3-1.7b_fixed_shard{0,1,2}.jsonl > data/math_level4_qwen3-1.7b_fixed.jsonl
say "collection done: $(wc -l < data/math_level4_qwen3-1.7b_fixed.jsonl) points"

say "STAGE 2: probe export"
python experiments/2_estimators/export_probe_diversity.py qwen3-1.7b data/math_level4_qwen3-1.7b_fixed.jsonl 14 \
  probes/entropy_probe_1.7b_diversity_level4_layer14_fixed.json 2>&1 | tee -a "$LOG"

say "STAGE 3: tournament (277/277/275)"
pids=()
offs=(0 277 554); ns=(277 277 275)
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g python experiments/3_tournament/shared_trace_monotonic.py qwen3-1.7b 4 \
    probes/entropy_probe_1.7b_diversity_level4_layer14_fixed.json ${ns[$g]} ${offs[$g]} \
    data/shared_trace_fixed_level4_offset${offs[$g]}.jsonl > data/tournament_fixed_gpu$g.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || { say "FAILED: tournament shard (pid $p) -- see data/tournament_fixed_gpu*.log"; exit 1; }; done
say "ALL DONE"
