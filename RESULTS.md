# Results map

Every number in [*Uncertainty Is Not A Compass*](https://matthewbragonje.com/writing/uncertainty-is-not-a-compass/),
with the command that produces it from saved results. Run from the repo root
with `.venv` (all of these are CPU-only). ✓ = re-run against the saved data
when this repo was published, and matches the article.

The MATH results use the corrected data collection (after the forced-branching
off-by-one fix; see NOTES.md). Rows the article labels as from the earlier
collection (GSM8K, embedding-weighted entropy, DRAGIN) were computed before the
fix and are kept for the record only.

## Part 1: entropy estimators

**Ground-truth stats, probe and simple-entropy correlations** ✓
(MATH-L4: 51.8% all agree, 7.1% all differ; probe r = 0.56 at layer 18; simple entropy r = −0.02)

```
python experiments/2_estimators/export_probe_diversity.py qwen3-1.7b data/math_level4_qwen3-1.7b_fixed.jsonl best <out.json>
```

Prints held-out (split by problem) r for every layer, simple (final-layer)
entropy r on the same held-out problems, and the agreement shares. Its export
is bit-identical to `probes/entropy_probe_1.7b_diversity_level4_best_fixed.json`.

MATH-L3 (partial collection, 323 problems): concatenate
`data/math_level3_qwen3-1.7b_fixed_shard{0,1,2}.jsonl` and run the same command.
Probe r = 0.57 at layer 26 ✓. Simple entropy gives **r = −0.05**; the
article's −0.08 for this cell is an error (same conclusion: no signal).

**Ridge / gradient boosting / bottleneck MLP, MATH-L4 row** ✓ (R² 0.305 / 0.271 / 0.067)

```
python experiments/2_estimators/diversity_architecture_compare.py data/math_level4_qwen3-1.7b_fixed.jsonl 18 L4fixed
```

About an hour on CPU (gradient boosting dominates).

**Natural-fork benchmark** ✓ (261 traces: random 9.9%; probe 0.83x, logit entropy 1.32x, product 1.34x, earliness 1.49x)

```
python experiments/4_fork_detection/analyze_fork_detection.py \
  data/fork_det_w{0,1,2}.jsonl data/fork_det2_w{0,1,2}.jsonl data/fork_det3_w{0,1,2}.jsonl data/fork_det4_w{0,1,2}.jsonl
```

The lookahead-gate sentence (1.21x → 1.35x on 106 traces) is the same script on
`fork_det3` + `fork_det4` only, the runs that computed the gate.

## Part 2: the tree search

**The 9-strategy tournament table** ✓ (250 integer-answer problems; fresh samples 79.2% / 35.6%, …)

```
python experiments/3_tournament/analyze_tournament.py --integer-only data/advanced_vllm_w{0,1,2}.jsonl
```

The smallest learned-vs-control p-value is 0.296 (the article rounds to "p ≥ 0.3").

**Noise ablation chart** ✓ (n = 245; 56.3 / 60.0 / 60.4 / 60.8 / 60.4; trend +3.6)

```
python experiments/3_tournament/analyze_noise_ablation.py data/shared_trace_fixed_level4_offset{0,277,554}.jsonl
```

The trend's bootstrap CI moves slightly with the random seed ([−2.8, +10.0]
here vs. [−2.9, +10.3] in the article).

**The race** ✓ (+4–8 points over resampling at 2–4k tokens across three runs;
more tokens on average; forced-token probability chart; 11.3% vs. 12.5%)

```
python experiments/5_race/race_analyze.py data/race_w{0,1,2}.jsonl      # greedy branches
python experiments/5_race/race_analyze.py data/race_s_w{0,1,2}.jsonl    # sampled branches
python experiments/5_race/race_analyze.py data/race_p_w{0,1,2}.jsonl    # sampled, probabilities saved (chart)
```

## Conclusion

- **Lookahead gate, best branching rescue rate (73.2%)** ✓:
  `python experiments/3_tournament/analyze_gate.py data/gate_w{0,1,2}.jsonl`
- **Tokens to first correct vs. resampling** ✓:
  `python experiments/3_tournament/ttfc_compare.py data/gate_w{0,1,2}.jsonl`
- **Branches landing back on the trace's own wrong answer** (all 382 problems,
  including decimal answers): `python experiments/3_tournament/analyze_fork_vote.py data/fork_vote_w{0,1,2}.jsonl`

## Engineering notes

The vLLM speedup comes from the logs of the same tournament on both backends:
`data/advanced_tournament_gpu*.log` (HF, stopped after 41 minutes) and
`data/advanced_vllm_gpu*.log` (vLLM, complete in 56 minutes).
`tools/validate_vllm.py` checks vLLM's forced branches against HF's.

## Fork Explorer

The page's `data.json` is rebuilt from per-trace dumps ✓ (byte-identical):

```
python experiments/explorer/debug_strategy_trace.py 4 probes/entropy_probe_1.7b_diversity_level4_best_fixed.json <n> data/explorer/dump.json
python experiments/explorer/add_local_entropy.py probes/entropy_probe_1.7b_diversity_level4_best_fixed.json data/explorer/dump.json
python experiments/explorer/build_explorer_data.py data.json data/explorer/debug_traces.json data/explorer/debug_traces_more.json
```
