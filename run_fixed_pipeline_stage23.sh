#!/usr/bin/env bash
# Replaces stages 2-3 of run_fixed_pipeline.sh mid-run (2026-10-25 ~22:45).
# Change: the tournament probe uses the best held-out layer instead of a
# hard-coded 14. A preview on 1/3 of the fixed data put layer 28 at r=0.555
# vs layer 14 at 0.484 -- the old layer choice was made on the off-by-one
# data and no longer holds. Usage: run_fixed_pipeline_stage23.sh <pid>...
# (the still-running collection shards to wait on)
set -euo pipefail
cd /home/matthew/projects/ml-experiments/tree2
source .venv/bin/activate
export HF_HUB_OFFLINE=1
LOG=fixed_pipeline.log
say() { echo "[$(date '+%a %H:%M')] $*" | tee -a "$LOG"; }

for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
for g in 0 1 2; do
  grep -q "Done\." collect_fixed_gpu$g.log || { say "FAILED: collection shard $g did not finish -- see collect_fixed_gpu$g.log"; exit 1; }
done
cat math_level4_qwen3-1.7b_fixed_shard{0,1,2}.jsonl > math_level4_qwen3-1.7b_fixed.jsonl
say "collection done: $(wc -l < math_level4_qwen3-1.7b_fixed.jsonl) points"

say "STAGE 2: probe export (best held-out layer)"
PROBE=entropy_probe_1.7b_diversity_level4_best_fixed.json
python export_probe_diversity.py qwen3-1.7b math_level4_qwen3-1.7b_fixed.jsonl best $PROBE 2>&1 | tee -a "$LOG"

say "STAGE 3: tournament (277/277/275)"
pids=()
offs=(0 277 554); ns=(277 277 275)
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g python shared_trace_monotonic.py qwen3-1.7b 4 $PROBE ${ns[$g]} ${offs[$g]} \
    shared_trace_fixed_level4_offset${offs[$g]}.jsonl > tournament_fixed_gpu$g.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || { say "FAILED: tournament shard (pid $p) -- see tournament_fixed_gpu*.log"; exit 1; }; done
say "ALL DONE"
