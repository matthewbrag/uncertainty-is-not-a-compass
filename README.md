# Uncertainty Is Not A Compass

Code for the article [*Uncertainty Is Not A Compass*](https://matthewbragonje.com/writing/uncertainty-is-not-a-compass/):
can you predict where an LLM's reasoning forks, and does branching there help it
find answers it got wrong? Qwen3-0.6B and Qwen3-1.7B on GSM8K and MATH, run
locally on 3x RTX 3090.

Short version of what's here:

- **Ground truth.** At a token in a reasoning trace, force the model's top-5
  next tokens, run each to an answer, and measure how many *distinct* answers
  come out (diversity entropy). Later, a second ground truth for *natural*
  forks: resample from just before vs. just after a token and see whether the
  answer distribution shifts.
- **Estimators.** Logit entropy, embedding-weighted entropy, a DRAGIN-style
  score, and a linear probe on hidden states that predicts diversity entropy
  (held-out r = 0.56 on MATH Level 4), plus their product.
- **Search.** A tournament of branching strategies on 250 problems the model
  got wrong, a noise ablation, fork-detection benchmarks, and a race between
  resampling and resample-then-branch hybrids.

[RESULTS.md](RESULTS.md) maps every number in the article to the script and
data that produce it. [NOTES.md](NOTES.md) is the lab notebook: every run in
order, including the bugs and the results that didn't survive replication.

## Layout

```
forkprobe/            shared code (pip install -e .)
  stop_rules.py         prompt template, answer parsing, repetition detection
  hf.py                 HF model loading + generation stop rules
  vllm_gen.py           vLLM backend with the same stop rules
experiments/
  1_collect/            forced-branch data collection (ground truth + hidden states)
  2_estimators/         probe training and the estimator comparisons (Part 1)
  3_tournament/         branching tournament, noise ablation, gate, voting, pruning (Part 2)
  4_fork_detection/     natural-fork benchmarks
  5_race/               resample vs. resample-then-branch
  explorer/             builds the Fork Explorer's data from per-trace dumps
probes/               trained probe artifacts (the article's is entropy_probe_1.7b_diversity_level4_best_fixed.json)
runs/                 the launch scripts the article's runs used (3 GPUs, shared work queues)
tools/                vLLM-vs-HF validation, the token-tree explorer app, smoke test, GPU benchmark
legacy/               earlier pipeline versions, kept for the research trail; not maintained
data/                 local data and logs (gitignored; multi-GB, regeneratable)
```

Run everything from the repo root; scripts read and write `data/` and `probes/`.

## Setup

Two virtual environments, because vLLM pins its own torch:

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -e .

python3 -m venv .venv-vllm
.venv-vllm/bin/pip install -r requirements-vllm.txt
.venv-vllm/bin/pip install -e .
```

`.venv` runs collection, probes, the estimator checks and the explorer dumps;
`.venv-vllm` runs the tournaments, fork-detection benchmarks and the race (each
script's docstring says which). The first run downloads Qwen3 and the datasets
from Hugging Face; after that, `HF_HUB_OFFLINE=1` skips the network.

Check the whole pipeline end to end on a few problems (about 35 minutes on two GPUs):

```
tools/smoke_test.sh 0 1      # HF lane on GPU 0, vLLM lane on GPU 1
```

## Reproducing

The analyses in [RESULTS.md](RESULTS.md) run on CPU from saved results. Regenerating
those results from scratch follows `runs/`, in order:

1. `runs/run_fixed_pipeline.sh` collects MATH Level 4 forced-branch data (3 GPUs,
   about 5 hours), trains the probe, and runs the noise-ablation tournament
   (`run_fixed_pipeline_stage23.sh` is its second half, as actually run).
2. `runs/run_advanced_vllm.sh` runs the 9-strategy tournament (about 1 hour on vLLM).
3. The fork-detection, gate, voting, pruning and race launchers in `runs/`.

Multi-GPU runs share a file-locked work queue (`data/*_queue.txt`), so
any number of workers can split a run.

Two methodology traps worth knowing about, both found and fixed mid-project
(details in NOTES.md): forced branching originally had an off-by-one that
forced the wrong token slot, and the answer stop rule cut decimal answers short
("#### 4.5" became "4."). Every result in the article uses the fixed code;
the tournament analyses restrict to integer answers because some runs predate
the decimal fix.

## License

MIT
