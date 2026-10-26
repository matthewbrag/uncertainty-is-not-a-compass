#!/usr/bin/env bash
# Queued behind run_fixed_pipeline.sh: re-collect the rest of the Part 1
# table under the fixed forcing, then a MATH-L3 tournament to keep the GPUs
# busy. Each stage is independent -- a failure is logged and the queue moves
# on, so the GPUs don't sit idle until someone checks in.
set -uo pipefail
cd /home/matthew/projects/ml-experiments/tree2
source .venv/bin/activate
export HF_HUB_OFFLINE=1
LOG=fixed_pipeline.log
say() { echo "[$(date '+%a %H:%M')] $*" | tee -a "$LOG"; }

until grep -q "ALL DONE\|FAILED" "$LOG" 2>/dev/null; do sleep 60; done
say "FOLLOWUPS: starting (main pipeline status: $(grep -o 'ALL DONE\|FAILED.*' "$LOG" | tail -1))"

run3() {  # run3 <name> <cmd-template with {g}> ; waits for all 3 GPUs
  local name=$1 tmpl=$2 ok=1 pids=()
  for g in 0 1 2; do
    bash -c "CUDA_VISIBLE_DEVICES=$g ${tmpl//\{g\}/$g}" > "${name}_gpu$g.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || ok=0; done
  [ $ok = 1 ] && say "$name: done" || say "$name: FAILED on >=1 shard -- see ${name}_gpu*.log"
  return $((1 - ok))
}

say "F1: MATH-L3 1.7B collection (3 x 300)"
if run3 collect_fixed_l3 'python math_collect_batched.py qwen3-1.7b 3 300 3000 $(({g} * 300)) math_level3_qwen3-1.7b_fixed_shard{g}.jsonl'; then
  cat math_level3_qwen3-1.7b_fixed_shard{0,1,2}.jsonl > math_level3_qwen3-1.7b_fixed.jsonl
  python export_probe_diversity.py qwen3-1.7b math_level3_qwen3-1.7b_fixed.jsonl best \
    entropy_probe_1.7b_diversity_level3_best_fixed.json 2>&1 | tee -a "$LOG"
fi

say "F2: GSM8K 1.7B collection (3 x 500)"
if run3 collect_fixed_gsm17 'python overnight_collect_batched.py qwen3-1.7b {g} 3 500 overnight_qwen3-1.7b_fixed_shard{g}.jsonl'; then
  cat overnight_qwen3-1.7b_fixed_shard{0,1,2}.jsonl > gsm8k_qwen3-1.7b_fixed.jsonl
  python export_probe_diversity.py qwen3-1.7b gsm8k_qwen3-1.7b_fixed.jsonl best \
    entropy_probe_1.7b_diversity_gsm8k_best_fixed.json 2>&1 | tee -a "$LOG"
fi

say "F3: GSM8K 0.6B collection (3 x 500)"
if run3 collect_fixed_gsm06 'python overnight_collect_batched.py qwen3-0.6b {g} 3 500 overnight_qwen3-0.6b_fixed_shard{g}.jsonl'; then
  cat overnight_qwen3-0.6b_fixed_shard{0,1,2}.jsonl > gsm8k_qwen3-0.6b_fixed.jsonl
  python export_probe_diversity.py qwen3-0.6b gsm8k_qwen3-0.6b_fixed.jsonl best \
    entropy_probe_0.6b_diversity_gsm8k_best_fixed.json 2>&1 | tee -a "$LOG"
fi

if [ -f entropy_probe_1.7b_diversity_level3_best_fixed.json ]; then
  say "F4: MATH-L3 tournament with fixed L3 probe (3 shards)"
  run3 tournament_fixed_l3 'python shared_trace_monotonic.py qwen3-1.7b 3 entropy_probe_1.7b_diversity_level3_best_fixed.json 300 $(({g} * 300)) shared_trace_fixed_level3_offset$(({g} * 300)).jsonl'
fi
say "FOLLOWUPS: ALL DONE"
