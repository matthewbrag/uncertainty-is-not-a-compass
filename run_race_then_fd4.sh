#!/bin/bash
# race on all GPUs, then the confirmatory fork-detection run (150 new problems)
cd /home/matthew/projects/ml-experiments/tree2
P=entropy_probe_1.7b_diversity_level4_best_fixed.json
echo 0 > race_queue.txt
echo "[$(date '+%a %H:%M')] RACE: starting" >> fixed_pipeline.log
for g in 0 1 2; do CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 timeout 3h .venv-vllm/bin/python race.py $P race_queue.txt race_w$g.jsonl > race_gpu$g.log 2>&1 & done
wait
echo "[$(date '+%a %H:%M')] RACE: done; FORK-DETECTION confirm: starting (150 problems, offset 460)" >> fixed_pipeline.log
echo 0 > fork_det4_queue.txt
for g in 0 1 2; do CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 timeout 3h .venv-vllm/bin/python fork_detection_benchmark_v2.py $P fork_det4_queue.txt fork_det4_w$g.jsonl 150 460 > fork_det4_gpu$g.log 2>&1 & done
wait
echo "[$(date '+%a %H:%M')] FORK-DETECTION confirm: done" >> fixed_pipeline.log
