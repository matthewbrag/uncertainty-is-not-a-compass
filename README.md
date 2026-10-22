# Decode-tree entropy research

Investigating cheap proxies for finding decision-critical branch points in LLM
decode trees — where in a reasoning trace does forcing a different token
actually change the outcome, and can you predict that cheaply, without
generating every alternative? Models: Qwen3-0.6B and Qwen3-1.7B, run locally
on 3x RTX 3090s. Datasets: GSM8K and MATH (levels 3-4).

Full writeup: *Entropy Isn't Enough* (link once published).

## Pipeline

1. **Collect forced-branch data** — `overnight_collect.py` / `overnight_collect_batched.py` (GSM8K)
   and `math_collect_batched.py` (MATH). At candidate token positions in a greedy
   trace, force the model's own top-5 most likely continuations, run each to
   completion, and record which ones land on the correct final answer. Saves
   per-position hidden states at 7 sampled layers (6, 10, 14, 18, 22, 26, 28),
   raw softmax/logit-lens entropy at those layers, and the full set of forced
   continuations' answers.

2. **Train a probe** — `export_entropy_probe.py` / `export_probe_generic.py` /
   `export_probe_p_target.py`. A Ridge regression on the saved hidden states,
   predicting either `outcome_p` (success probability) or `diversity_entropy`
   (Shannon entropy over the *distinct* final answers the 5 forced branches
   produce — a better target than binary correctness-entropy; see the writeup).
   Produces a small (~100-900KB) JSON artifact: scaler + coefficients + layer.

3. **Run the tournament** — `math_tournament.py` / `math_tournament_diversity.py`
   / `math_tournament_filtered.py`. On held-out test problems, pick 5 branch
   points per problem under different strategies (trained probe, uniformly
   spaced, random, filtered-for-filler-tokens) and check whether forcing
   branches there ever rescues a problem the greedy decode got wrong.

4. **Validate specific hypotheses**:
   - `embedding_entropy_check.py` / `embedding_entropy_diversity_check.py` —
     does weighting token-distribution entropy by embedding-space similarity
     (so near-synonyms don't inflate the score) correlate better with the
     ground truth than raw entropy?
   - `dragin_reproduction_check.py` — faithful reproduction of DRAGIN's
     published RIND score (Su et al., 2024, arXiv:2403.10081) against our
     ground truth, as a published-method comparison.
   - `filler_rate_check.py` — what words/tokens do entropy-based selection
     strategies actually pick, and does filtering them out change anything?

5. **`app.py`** — the interactive Token Tree Explorer used to manually poke at
   individual decode trees during early exploration (local-only, loads a real
   model; not needed to reproduce the written results).

## `legacy/`

Earlier pilots, superseded script versions (`strategy_tournament.py` through
`_v5.py`), and one-off benchmarking/architecture-search scripts
(`bottleneck_sweep.py`, `tune_entropy_head.py`, etc.) that fed into decisions
upstream of the scripts above but aren't part of the current pipeline.
Kept for the research trail, not meant to be run as-is.

## Data

Raw collected data (`*.jsonl`, multi-GB) is gitignored — it's fully
regeneratable by running the collection scripts against the public
datasets and model, and isn't the source of truth. Trained probe artifacts
(`entropy_probe_*.json`) are small enough to keep in git directly.

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers accelerate scikit-learn numpy
```

Requires an offline-cached copy of Qwen3-0.6B/1.7B and the GSM8K/MATH
datasets (`HF_HUB_OFFLINE=1` is set in all run commands; drop it for a
first run to let `transformers`/`datasets` download them).
