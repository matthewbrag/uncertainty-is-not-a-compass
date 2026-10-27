#!/usr/bin/env bash
# Runs every experiment script on a handful of problems to check the pipeline
# end to end (imports, model loading, generation, output format). Not a
# reproduction: the numbers are meaningless at this size. Two lanes run in
# parallel, HF scripts on one GPU and vLLM scripts on another.
#
# Usage: tools/smoke_test.sh [hf_gpu] [vllm_gpu]     (defaults 0 1)
# Needs .venv and .venv-vllm (see README) and, for the steps that read earlier
# results, data/advanced_vllm_w*.jsonl and data/advanced_tournament_w*.jsonl.
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
HF_GPU=${1:-0}; VLLM_GPU=${2:-1}
OUT=data/smoke; mkdir -p $OUT
PROBE=probes/entropy_probe_1.7b_diversity_level4_best_fixed.json
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

step() {  # step <lane log> <name> <expected output file or -> <command...>
  local log=$1 name=$2 want=$3; shift 3
  local t0=$SECONDS
  if "$@" > "$OUT/$name.log" 2>&1 && { [ "$want" = - ] || [ -s "$want" ]; }; then
    echo "PASS  $name  ($((SECONDS - t0))s)" | tee -a "$log"
  else
    echo "FAIL  $name  ($((SECONDS - t0))s) -- see $OUT/$name.log" | tee -a "$log"
  fi
}

hf_lane() {
  local L=$OUT/lane_hf.txt; : > $L
  local PY="env CUDA_VISIBLE_DEVICES=$HF_GPU .venv/bin/python"
  step $L collect_math $OUT/collect_math.jsonl $PY experiments/1_collect/math_collect_batched.py qwen3-1.7b 4 5 3000 0 $OUT/collect_math.jsonl
  step $L collect_gsm8k $OUT/collect_gsm8k.jsonl $PY experiments/1_collect/overnight_collect_batched.py qwen3-0.6b 0 3 2 $OUT/collect_gsm8k.jsonl
  # 5 problems: the probe holds out a fifth of problems, so fewer leaves no test set
  step $L export_probe $OUT/probe.json .venv/bin/python experiments/2_estimators/export_probe_diversity.py qwen3-1.7b $OUT/collect_math.jsonl 18 $OUT/probe.json
  step $L embedding_entropy - $PY experiments/2_estimators/embedding_entropy_diversity_check.py 20 data/math_level4_qwen3-1.7b_fixed.jsonl qwen3-1.7b
  step $L dragin - $PY experiments/2_estimators/dragin_reproduction_check.py 20
  step $L noise_ablation_tournament $OUT/noise.jsonl $PY experiments/3_tournament/shared_trace_monotonic.py qwen3-1.7b 4 $PROBE 8 0 $OUT/noise.jsonl
  step $L explorer_dump $OUT/dump.json $PY experiments/explorer/debug_strategy_trace.py 4 $PROBE 1 $OUT/dump.json
  step $L explorer_entropy - $PY experiments/explorer/add_local_entropy.py $PROBE $OUT/dump.json
  step $L explorer_build $OUT/explorer.json .venv/bin/python experiments/explorer/build_explorer_data.py $OUT/explorer.json $OUT/dump.json
  step $L greedy_vs_sampled $OUT/gvs.json $PY tools/greedy_vs_sampled.py 4 2 0 $OUT/gvs.json
}

vllm_lane() {
  local L=$OUT/lane_vllm.txt; : > $L
  local PY="env CUDA_VISIBLE_DEVICES=$VLLM_GPU .venv-vllm/bin/python"
  q() { echo "$2" > "$OUT/$1.queue"; echo "$OUT/$1.queue"; }  # fresh work queue starting at $2
  step $L validate_vllm - $PY tools/validate_vllm.py 2 0.5
  step $L tournament $OUT/tournament.jsonl $PY experiments/3_tournament/advanced_tournament_vllm.py $PROBE "$(q tournament 0)" $OUT/tournament.jsonl 8
  step $L fork_vote $OUT/fork_vote.jsonl $PY experiments/3_tournament/fork_vote_tournament.py "$(q fork_vote 0)" $OUT/fork_vote.jsonl 8
  step $L fork_prune $OUT/fork_prune.jsonl $PY experiments/3_tournament/fork_prune_tournament.py $PROBE "$(q fork_prune 0)" $OUT/fork_prune.jsonl 8
  step $L fork_prune_ablation $OUT/fork_ablation.jsonl $PY experiments/3_tournament/fork_prune_ablation.py $PROBE "$(q fork_ablation 0)" $OUT/fork_ablation.jsonl 8
  step $L gate_addon $OUT/gate.jsonl $PY experiments/3_tournament/gate_addon.py "$(q gate 248)" $OUT/gate.jsonl
  step $L fork_detection $OUT/fork_det.jsonl $PY experiments/4_fork_detection/fork_detection_benchmark.py $PROBE "$(q fork_det 0)" $OUT/fork_det.jsonl 8
  step $L fork_detection_v2 $OUT/fork_det2.jsonl $PY experiments/4_fork_detection/fork_detection_benchmark_v2.py $PROBE "$(q fork_det2 0)" $OUT/fork_det2.jsonl 8
  step $L race $OUT/race.jsonl $PY experiments/5_race/race.py $PROBE "$(q race 246)" $OUT/race.jsonl sampled
}

# analyses run on CPU against whatever the lanes produced
analyses() {
  local L=$OUT/lane_analysis.txt; : > $L
  step $L analyze_tournament - .venv/bin/python experiments/3_tournament/analyze_tournament.py $OUT/tournament.jsonl
  step $L analyze_noise_ablation - .venv/bin/python experiments/3_tournament/analyze_noise_ablation.py $OUT/noise.jsonl
  step $L analyze_fork_vote - .venv/bin/python experiments/3_tournament/analyze_fork_vote.py $OUT/fork_vote.jsonl
  step $L analyze_gate - .venv/bin/python experiments/3_tournament/analyze_gate.py $OUT/gate.jsonl
  step $L ttfc_compare - .venv/bin/python experiments/3_tournament/ttfc_compare.py $OUT/gate.jsonl
  step $L analyze_fork_detection - .venv/bin/python experiments/4_fork_detection/analyze_fork_detection.py $OUT/fork_det.jsonl
  step $L race_analyze - .venv/bin/python experiments/5_race/race_analyze.py $OUT/race.jsonl
}

hf_lane & vllm_lane & wait
analyses
echo; echo "$(grep -ch PASS $OUT/lane_*.txt | paste -sd+ | bc) passed, $(grep -ch FAIL $OUT/lane_*.txt | paste -sd+ | bc) failed"
grep -h FAIL $OUT/lane_*.txt
