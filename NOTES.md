# Running notes

Started 2026-10-22, because we forgot to take them earlier and it cost us a
round-trip re-verifying which papers we'd already checked. Add to this as we
go. Most important entries first within each section.

## Methodology traps (read this before trusting a new result)

- **Trace resampling confound (found 2026-10-22, UNRESOLVED).** Every
  tournament/experiment script re-generates its own greedy trace per problem
  with `do_sample=True, temperature=0.7` in `generate_batch_sampled`. Only
  problem *ordering* is seeded (`random.seed(7000)`) — the actual generation
  is never seeded via `torch.manual_seed`. So "problem #482" in two separate
  runs can produce two completely different wrong traces (different length,
  different error location, different rescuability). This silently confounds
  ANY cross-run comparison, including supposedly "paired" designs like the
  monotonic-noise experiment (same problem indices across alpha arms, but
  NOT the same underlying decode trace per arm).
  - Confirmed concretely twice: (1) restoring a clobbered MATH-L4 shard with
    the same seed gave 173 scored vs the original 170 (3-problem flip); (2)
    `random`/`uniform` on the monotonic-noise run's problem subset scored
    30.7% in the original tournament file vs the noise experiment's
    independently-generated traces showing very different numbers for
    nominally the same problems.
  - **Fix to try**: generate the traces ONCE per problem set (prompt_ids +
    gen_ids, greedy_correct flag), save to a shared file, and have every
    variant (noise levels, hybrid strategy, whatever comes next) load that
    *same* fixed trace instead of calling `generate_batch_sampled` itself.
    Only the branch-forcing step should still touch the model fresh. This
    turns every multi-arm comparison into an actually-paired design instead
    of a confounded one. Haven't built this yet — next thing to do if we
    want to trust fine-grained comparisons (noise curve shape, hybrid vs
    uniform) rather than just large, obvious gaps.
  - Until this is fixed: trust big, consistent, multiply-replicated gaps
    (uniform beating probe_p by 5+pp across two difficulty levels). Don't
    trust small gaps from a single run, and don't directly compare rescue
    rates across *different scripts' runs* even on "the same" problems.

- **Always use a scratch output path for smoke tests, never a real one.**
  Did this wrong once already — overwrote a real 170-record tournament shard
  with a 3-record smoke test by reusing the production filename. Every new
  script's smoke test should point at `/tmp/.../scratchpad/...` explicitly.

- **Binary outcome_entropy (correctness) vs diversity_entropy are different
  metrics — r=0.278 between them.** Binary asks "did the branches agree on
  being right"; diversity asks "did the branches agree at all." A position
  where all 5 branches converge on the same WRONG answer scores 0 under
  both. A position where all 5 branches give 5 DIFFERENT wrong answers
  scores 0 under binary but ~log2(5)=2.32 (max) under diversity. Most of
  this project's early results (anything from before this was caught) used
  the wrong one. Diversity is the one to use going forward — check any old
  number's target before reusing it.

## Key findings so far

**Part 1 (entropy estimators, vs diversity_entropy target, all verified):**
- Simple Entropy, Embedding Weighted Entropy, DRAGIN (faithful RIND repro):
  all |r| < 0.15 on every dataset/model tested. Flat noise.
- Trained probe (Ridge, best layer): R²=0.32-0.50, r=0.57-0.71. The one real
  positive signal in Part 1.
- Architecture barely matters once the target is right: Ridge and gradient
  boosting trade the lead by noise-level margins everywhere. MLPs (32 and
  64→8 units) consistently worse than both.
- Low-dimensional-subspace hypothesis (does a 1-4 unit bottleneck match
  Ridge?) is domain-split, not universal: nearly matches/beats Ridge on
  GSM8K (0.6B bottleneck even beats it, R²=0.512 vs 0.497), goes NEGATIVE on
  both MATH levels. Real, surprising, worth featuring.

**Part 2 (tournament, MATH domain, 1.7B model):**
- `uniform` (deterministic even-spacing, zero learned signal) ties or beats
  every smart strategy tested, confirmed independently on MATH-L3 (wins
  outright, 45.7%) and MATH-L4 (ties probe_diversity_spaced, ~40.7-40.9%).
  Not a cost tradeoff either — uniform is cheapest AND best.
  CAVEAT (2026-10-22): the MATH-L4 "final" numbers (uniform 40.7%,
  probe_diversity_spaced 40.9%) came from a merged file where one shard was
  regenerated after the smoke-test-clobber incident above — so even this
  headline number has some trace-resampling noise baked in versus the
  original run. Directionally solid (replicated independently on MATH-L3
  too), but don't quote the exact decimal as gospel.
- `probe_p` (top-5 by predicted success probability, no spacing) is worst on
  every single test, every dataset. Consistently.
- Naive top-k on a real, correlated signal (`probe_diversity`) actually
  LOSES to doing nothing (uniform) — because top-scoring positions cluster
  spatially (signal is locally correlated), so you waste your 5-branch
  budget testing near-duplicate positions.
- Two independent fixes for that clustering problem (spacing/NMS-exclusion,
  and Gumbel-softmax sampling) each close the gap to "tied with uniform" —
  neither has ever been shown to beat it. This replicates an EARLIER round
  from before this session's summary too (probe_p → probe_p_softmax, same
  "tied, never beats" pattern) — now a repeated finding across two separate
  investigations.
- Filtering junk/filler tokens from the candidate pool: tested twice
  (tournament-level simulation + correlation-level), both clean nulls. ~70%
  of problems don't even have 5 non-junk candidates to filter to.
- Softmax-sampled vs top-k forced BRANCH TOKENS (different axis — this is
  about which tokens to force once you've picked a position, not which
  positions to pick): also null, and not even complementary (same rescues,
  not different ones).

**In flight / landed, needs re-verification:**
- Monotonic noise sweep (n=~99/arm, MATH-L4): alpha=0.0→34.0%, 0.25→32.3%,
  0.5→29.9% (dip), 0.75→pending, 1.0→42.9%. NOT monotonic so far, but see
  trace-resampling caveat above — alpha=1.0's jump matches what you'd see
  from trace variance alone, not necessarily a real effect. Don't trust the
  curve shape until re-run with shared traces.
- Spacing+softmax hybrid strategy: 39.4% (39/99), MATH-L4. Higher than the
  same-subset `random`/`uniform` (30.7%) and `probe_diversity_spaced`
  (35.2%) pulled from the original tournament file — but same
  trace-resampling confound applies, AND it's in tension with the oracle
  result just above (if even perfect diversity knowledge only beats random
  by 0.6pp, an 8-9pp jump from a selection-mechanism fix on the same noisy
  signal is a lot to explain). Treat as "interesting, re-verify" not
  "uniform finally beaten." Re-run after the trace-resampling fix before
  believing it.

## THE finding (2026-10-22, oracle test)

Ran the oracle version of the selection question: rank candidate points by
the REAL (not probe-predicted) diversity_entropy, computed directly from
already-recorded forced-branch answers in `math_level4_qwen3-1.7b.jsonl` (900
problems, zero new GPU compute, CPU-only reuse of existing data).

```
oracle (perfect diversity_entropy) top-5: 61.0% rescued
random 5 of available candidates:         60.4%
spread evenly among candidates:           60.3%
outcome flips between oracle and random on only 3.4% of problems
```

Even PERFECT, noiseless knowledge of true outcome diversity barely beats
random (0.6pp). This is not a probe-quality problem — it's structural.
Diversity entropy measures whether the 5 forced branches DISAGREE; it does
not measure whether the CORRECT answer is among them. A maximally-diverse
point (5 different wrong answers) can be completely unrescuable; a
low-diversity point can happen to contain the right answer anyway. These are
different properties. The entire premise behind Part 2 — "a good-enough
divergence measurement should let you pick better branch points" — doesn't
survive contact with the oracle. This reframes the whole uniform-wins result:
it's not that our probe approximation of diversity wasn't good enough, it's
that diversity (even known perfectly) may not be the right target for THIS
downstream task at all. Possibly the single most important result in the
project — bigger than the binary-vs-diversity correction, because it
suggests diversity itself (not just our measurement of it) is the wrong
thing to optimize for rescue-oriented branch selection.

Open question this raises: is there a DIFFERENT real-but-cheap property that
WOULD predict rescuability better than diversity does? (E.g., something like
"probability mass on the correct answer specifically" — but that requires
knowing the correct answer, which defeats the purpose of a cheap proxy. Or:
some property of HOW the branches disagree, not just THAT they disagree.)
Not yet explored.

**Candidate mechanism for why ("the step problem"):** forced-branch
continuations run up to 300 new tokens after the fork. Even a totally
meaningless early difference (phrasing, step-numbering) can cascade through
300 tokens of the model conditioning on its own prior output and land on a
different final number, with no real reasoning fork involved. This would
explain the article's own "step" observation (a pure formatting token
scored high entropy despite not being a decision at all) AND the oracle
result (diversity partly measures cascade-amplified noise, not genuine
decision-criticality). Testable prediction: diversity measured close to the
END of a trace (less room left to cascade) should correlate with
rescuability better than diversity measured early. NOT YET TESTED — would
need per-branch continuation lengths, which aren't currently saved; a clean
test would need a new small collection (short-capped vs full-length
continuations at the same points). Flagged, not run, as of 2026-10-22.

## Oracle MONOTONIC test (2026-10-22, clean, trust this one fully)

**Cut from the article's v1** (2026-10-22) — not because the result is
wrong, but scope/energy: cut late at night, wanted to avoid debugging one
more angle before publishing. The data below is still valid and still the
cleanest methodology of anything run tonight (CPU-only, zero
trace-resampling confound, n=900) — it's just not in the public writeup
yet. Worth reviving for a v2 once rested; it actually reinforces rather than
contradicts the live monotonic test (both say diversity doesn't predict
rescuability well, even measured perfectly in the oracle's case).


Extended the oracle test into a full noise curve: mix the real diversity_entropy
with Gaussian noise at alpha=0/0.25/0.5/0.75/1.0, same fixed data (900
problems, math_level4_qwen3-1.7b.jsonl), same pairing across every alpha
level (not a fresh GPU run — zero trace-resampling confound, unlike the
probe-based monotonic/hybrid GPU experiments above).

```
alpha=0.00 (pure oracle signal): 60.9%
alpha=0.25:                       61.1%
alpha=0.50:                       61.2%
alpha=0.75:                       61.0%
alpha=1.00 (pure noise):          60.2%
```

Completely flat. Not just "oracle barely beats random" (the single-point
result) — degrading the oracle signal by ANY amount, all the way to total
destruction, changes essentially nothing. This is the strongest, cleanest
result in the whole project. Fully resolves the article's "Oracle and Oracle
Monotonic" section — no methodology caveat needed, this one's trustworthy.

## MATH-L4 generation cap finding (2026-10-22, not yet acted on)

41.3% of unique MATH-L4 traces (372/900) hit the 500-token generation cap
(`GEN_MAX_NEW_TOKENS`). GSM8K: 0%, mean 147 tokens vs MATH-L4's mean 382.
Checked 10 capped traces directly with `detect_repetition` — 0/10 were
actual repetition loops; the guard (`ForceEosOnRepetition`) is working fine,
there mostly aren't loops here to catch. Reading the raw tail content, these
are genuinely varied, substantive reasoning (case enumeration, solving
systems, trig identities) that just doesn't finish in budget.

Implication: this is a correctness lever, not a speed one — the model may be
getting a meaningful fraction of MATH-L4 wrong simply because it's cut off
mid-reasoning, independent of any entropy/search-strategy question. Current
setup (GEN_MAX_NEW_TOKENS=500 + the repetition-loop guard) is good enough
for v1, not worth blocking on. Worth revisiting later: does raising
`GEN_MAX_NEW_TOKENS` for MATH-L4 specifically change baseline accuracy or
the rescue-rate numbers? Not tested.

## Papers checked (verified via direct fetch, not memory)

- **DRAGIN** (Su et al. 2024, arXiv:2403.10081) — real paper, real RIND
  formula = H_i * a_max(i) * s_i (entropy × max future attention × stopword
  gate). Faithfully reproduced; comes back ~0 correlation with our ground
  truth either target. 69.5% of our candidate points get zeroed by the
  stopword gate alone — our selection heuristic and DRAGIN's gate are
  selecting for near-opposite things.
- **EDEN** (Evans/Ganesh/Ardon, JP Morgan AI Research, arXiv:2605.09745) —
  never tests uniform-spacing-vs-smart-selection at all. Baselines are all
  generation strategies (greedy/sampling/beam/best-of-n), not
  position-selection strategies.
- **Entropy-Tree** (Wei et al., SJTU/Huawei/Nankai/NTU, arXiv:2601.15296) —
  DOES claim entropy-guided beats random branching (Section 4.4), but the
  methodology is underspecified enough that we can't confirm it's testing
  the same thing we tested (random POSITION vs random TOKEN-at-a-position
  are different axes, and the paper doesn't say which).
- **Zur et al.** ("Are LMs Aware of the Road Not Taken?", arXiv:2511.04527)
  — the paper this whole project extends. Linear probe, token-level,
  predicts outcome distribution of forced continuations. Only ever uses
  linear probes, never compares to nonlinear — so it's NOT a citation for
  "a published method found linear architecture wins," just for "linear
  probes have precedent here."
- **Branching Policy Optimization** (He, Chen, Zhang, Liu, arXiv:2607.14171)
  — Table 3: uniformly-random (64.5±1.4%), equally-spaced (65.2±1.5%), their
  entropy-guided method (67.8±1.3%). Entropy-guided genuinely beats uniform
  here, with real error bars, outside the overlap. Different domain though
  (WebShop agent tasks, not math reasoning) and different model (Qwen2.5-7B).
  Worth engaging with directly in the writeup rather than ignoring — it's a
  real counter-example, not a methodology gap like Entropy-Tree's.
- **"Reasoning Can Be Restored..."** (Shen et al., arXiv:2605.16874) — tests
  a stochastic random-position baseline, but at a 6x LARGER budget than
  their smart method (ρ=0.25 vs ρ≈0.04) and still loses. Different task
  (weak-model token correction via a stronger model), not directly
  comparable to our setup.

## Repo

Git-initialized, committed (`/home/matthew/projects/ml-experiments/tree2`).
Core pipeline at top level, superseded/exploratory scripts in `legacy/`.
README documents the pipeline. Raw data gitignored (regeneratable, too big
for git anyway).
