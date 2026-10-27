#!/bin/bash
# race on all GPUs, then the confirmatory fork-detection run (150 new problems)
cd "$(dirname "$0")/.."  # repo root
P=probes/entropy_probe_1.7b_diversity_level4_best_fixed.json
echo 0 > data/race_queue.txt
echo "[$(date '+%a %H:%M')] RACE: starting" >> data/fixed_pipeline.log
for g in 0 1 2; do CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 timeout 3h .venv-vllm/bin/python experiments/5_race/race.py $P data/race_queue.txt data/race_w$g.jsonl > data/race_gpu$g.log 2>&1 & done
wait
echo "[$(date '+%a %H:%M')] RACE: done; FORK-DETECTION confirm: starting (150 problems, offset 460)" >> data/fixed_pipeline.log
echo 0 > data/fork_det4_queue.txt
for g in 0 1 2; do CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 timeout 3h .venv-vllm/bin/python experiments/4_fork_detection/fork_detection_benchmark_v2.py $P data/fork_det4_queue.txt data/fork_det4_w$g.jsonl 150 460 > data/fork_det4_gpu$g.log 2>&1 & done
wait
echo "[$(date '+%a %H:%M')] FORK-DETECTION confirm: done" >> data/fixed_pipeline.log
