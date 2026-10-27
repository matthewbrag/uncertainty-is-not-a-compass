# Running notes

Started 2026-10-22, because we forgot to take them earlier and it cost us a
round-trip re-verifying which papers we'd already checked. Add to this as we
go. Most important entries first within each section.

## Methodology traps (read this before trusting a new result)

- **Stop rule truncated decimal answers (found + fixed 2026-10-26).** ForceEosAfterAnswer
  stopped as soon as one character followed "#### <digits>", so "#### 4.5" ended as
  "#### 4." -> parsed as 4. 37/382 "wrong" traces in the vLLM tournament were cut off
  this way (29 with non-integer references). Equal across all arms, so paired
  comparisons were fair; decimal-answer problems were nearly unrescuable, and some
  "rescues" were artifacts (forcing "####" early sidestepped the truncation). Re-analysis
  on the 250 integer-answer problems (bug can't apply): every conclusion unchanged;
  absolute rescue rates ~47% -> ~71%; paired win/loss counts nearly identical (all the
  discordant problems were integer ones). Fixed in app.py and vllm_gen.py: wait for a
  character that can't continue the number.

- **Off-by-one in branch forcing (found + fixed 2026-10-25, commit a35e858).
  Invalidates every forced-branch number before that commit.** Forced tokens
  came from `hs[base+cut-1]` (the distribution predicting `gen_ids[cut]`) but
  the prefix was `gen_ids[:cut+1]`, so they were appended AFTER the real
  token, in slot cut+1. Top-1 "alternative" was always a duplicate of the
  token just written ("of of", "she she" -- 13/13 checked). Affected: all
  ground-truth collection (GSM8K + MATH), every probe trained on it, every
  tournament, the noise curve, measure_natural_branch_length. NOT affected:
  trace-only measurements (trace lengths, cap-hit rates, repetition guard),
  and adaptive_diversity_search.py / recursive_pivot_search.py, which used
  the correct convention -- which also means the "adaptive 19.4% vs
  tournament 42.9%" comparison below mixes conventions and is not valid.
  Critical-path rerun (MATH-L4 1.7B collect -> probe -> tournament) via
  run_fixed_pipeline.sh, all outputs named *_fixed.

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

## FINAL comprehensive run (2026-10-23, trust this one — shared-trace, n=420)

Full 829-problem MATH-L4 test set, 3 shards, all completed. Shared-trace
design (no trace-resampling confound), raised budgets (GEN_MAX_NEW_TOKENS=650,
CONT_MAX_NEW_TOKENS=450, confirmed cap-hits are genuine reasoning not loops),
continuation-length tracking included. This supersedes every earlier
monotonic/hybrid number in this file — use these.

```
strategy                  rescued  avg tok/branch  avg tok(correct)  tok/rescue
uniform                    43.1%           314.2            292.4       18227
noise_alpha0.5              40.2%           327.5            319.7       20346
noise_alpha0.25             40.0%           323.1            316.4       20191
noise_alpha0.75             39.5%           333.0            319.9       21064
random                      39.3%           319.1            305.7       20305
hybrid_spaced_softmax       38.8%           324.3            306.8       20892
noise_alpha0.0 (pure probe) 38.3%           315.9            307.6       20605
noise_alpha1.0 (pure noise) 36.2%           319.4            301.8       22060
```

Uniform wins outright again, on every axis (rate, avg cost, total cost per
rescue) — now with the budget change in place too, so the generation-cap
issue wasn't hiding a different result.

**IMPORTANT CORRECTION (2026-10-23): the budget raise did NOT fix the
truncation problem, it only partially reduced it.** Checked directly:
initial-trace cap-hit rate actually got WORSE (41.3% at cap=500 -> 55.7% at
cap=650) even though the cap went up - mean trace length jumped 382->528,
meaning the model used the extra room rather than the problem going away.
Forced-branch cap-hit rate did improve (69.8%->50.2%) but that "before"
number came from a tiny n=3 smoke test, so treat the improvement loosely.
Real conclusion: this isn't "500 was almost enough," it's a genuinely
heavy-tailed distribution where a real subpopulation of MATH-L4 problems
needs much more than any modest budget bump provides. Don't claim this is
"fixed" in the article - it's smaller, not resolved. Would need either a
much larger cap (1000+?) or to just report it as a structural property of
the domain. Noise curve still doesn't show a clean
monotonic shape (0.5 > 0.25 > 0.75, not a straight line), but the extremes
are suggestive: alpha=0.0 and alpha=1.0 are the two lowest of the five noise
arms. Hybrid strategy (39.4% in the earlier confounded single-run) settled
at 38.8% here — below uniform, consistent with the oracle finding, confirms
that earlier number was mostly noise as suspected.

**Token efficiency (new this run, answers "avg tokens to correct a wrong
answer"):**
```
correct branches:   305.5 tokens avg (n=5428), 21.6% hit the cap
incorrect branches: 325.3 tokens avg (n=51107), 53.3% hit the cap
```
Wrong answers aren't just wrong — they're measurably more expensive too.
Correct branches converge and conclude; incorrect ones are 2.5x more likely
to run out of budget without resolving at all.

## Natural-length measurement (2026-10-23): replaced guessed budgets with real data

Rather than guess another budget bump, measured the TRUE natural length
distribution directly: generate with only the repetition guard + a generous
ceiling (no answer-forced-stop processor), record where generation actually
would have stopped on its own. Two separate measurements, since initial
traces and forced branches have different length profiles.

`measure_natural_length.py` (initial traces, n=100, MATH-L4, 1.7B, ceiling=3000):
94% natural EOS, 0% repetition-guard stops, 6% hit the 3000 ceiling.
Percentiles among naturally-concluded: p50=477, p75=743, p90=1045,
**p95=1488**, p99=2233, max=2701.

`measure_natural_branch_length.py` (forced branches, n=300 forced from 100
traces generated at a 1500-token budget, ceiling=3000): 91.3% natural EOS,
0% repetition-guard stops, 8.7% hit ceiling. Percentiles among
naturally-concluded: p50=200, p75=362, p90=766, **p95=1268**, p99=2417,
max=2930.

0% repetition-guard stops in both measurements, at every budget tested in
this whole project (500/650/1500/3000) — repetition loops are not and never
were a meaningful contributor to cap-hitting. Every earlier "truncation
error" concern was genuine long/hard reasoning being cut off, not degenerate
loops; the guard is a no-op safety net here, not doing real work.

Tried running the tournament with a 3000-token ceiling on both trace and
branch generation ("non-truncated") first, per initial request. Confirmed
impractical: smoke test (n=5) ran 19+ minutes and was still on problem #1.
Root cause: a 3000 ceiling forces `MAX_BRANCH_BATCH` down to ~6 (KV-cache
OOM otherwise), and each problem has 25-40 unique candidate points across
the 8 strategies, so each problem needs 5-7 sequential micro-batches. With
~6-9% of completions genuinely needing near-ceiling length, nearly every
micro-batch contains at least one straggler holding up the whole batch —
classic static-batching tail latency, massively amplified by a high
ceiling. Settled on p95 instead: GEN_MAX_NEW_TOKENS=1500,
CONT_MAX_NEW_TOKENS=1300, MAX_BRANCH_BATCH=15, GEN_BATCH_SIZE=32.

## TRUE FINAL run (2026-10-24, p95-calibrated budgets, trust this one over everything above)

Full 829-problem MATH-L4 test set, 3 shards, all completed (~8-9 hours each,
overnight). Same shared-trace, no-resampling-confound design as the
2026-10-23 run above, but with budgets set from the real p95 data instead of
guessed (GEN_MAX_NEW_TOKENS=1500, CONT_MAX_NEW_TOKENS=1300). 364 of 829
problems scored (greedy got the other 465 right already and those are
skipped by design — `greedy_correct` filter, not a bug).

```
strategy                rescued  avg_cut_pos  cost/rescue
noise_alpha0.75          42.9%        376.1        877.7
hybrid_spaced_softmax    42.3%        390.7        923.5
uniform                  42.9%        403.1        940.6
noise_alpha1.0           40.9%        394.2        962.9
noise_alpha0.5           40.7%        394.5        970.3
noise_alpha0.25          40.1%        420.2       1047.6
random                   38.5%        403.3       1048.7
noise_alpha0.0 (pure probe) 39.8%     441.0       1107.1
```
(cost/rescue = avg cut position in tokens / rescue rate, lower is better — matches article's methodology)

Trace length at the new 1500 cap: mean=808, p50=635, p90=1500 (!),
cap-hit-rate=24.5% — down from 55.7% at the old 650 cap, so the bigger
budget genuinely helped most traces finish, even though the raw p90 sitting
right at the cap confirms the tail is still real and wasn't fully absorbed.

Branch continuation length, correct vs incorrect (cap=1300):
correct: n=4434, mean=589, cap-hit-rate=7.0%. incorrect: n=45006, mean=531,
cap-hit-rate=20.9%. Same "wrong answers are measurably more expensive"
pattern as the 2026-10-23 run holds up at the new budgets too.

**Verdict, now on genuinely clean data: the spread across all 8 strategies
is 38.5%-42.9%, a ~4.4-point band with no clean winner.** uniform and
noise_alpha0.75 tie for the top rescue rate; hybrid_spaced_softmax (the
probe-informed spacing strategy) is statistically indistinguishable from
both. noise_alpha0.0 (pure probe, no noise at all) sits mid-pack at 39.8%,
not meaningfully different from random's 38.5%. This is the clean,
deconfounded, un-truncated version of the conclusion the article already
draws — if anything it's a slightly tighter race than the article's current
table shows, since that table's numbers predate this calibration and should
be replaced with the above.

## Recursive/adaptive multi-pivot search (2026-10-24/25): depth loses to breadth

Tested whether the tournament's structural limitation — fork once, run that
fork straight to completion, never check whether the branch itself later
hits another high-entropy spot — was costing real rescues. Two scripts:

`recursive_pivot_search.py`: blind fixed-depth recursion. Fork into 3
lineages at one point, each independently re-pivots up to 3 total times,
forcing a rank-2 token each time (rank-1 would be a no-op since completions
are greedy-decoded). Smoke test (n=19) showed a real design flaw: final-depth
rescue rate came back identical to first-pivot-only rescue rate, because
blind re-pivoting has no way to know it already landed correct and can
clobber a right answer with a wrong one. Killed before a full run.

`adaptive_diversity_search.py`: fixed that by gating recursion on *observed*
outcome agreement instead of predicting it. At every pivot, force
BRANCH_FACTOR=3 alternatives, run each to completion, check if they agree
(Shannon entropy over distinct answers, same ground-truth metric as Part 1,
just read off live branches instead of a trained probe). Agreement -> settle,
don't recurse (nothing to gain, real risk of clobbering). Disagreement ->
recurse into each disagreeing branch, up to MAX_DEPTH=3 pivot events total.
Pivot-point *selection* still uses the diversity probe; only the
stop-vs-recurse decision stopped trusting it.

Full run, all 829 MATH-L4 problems' 364 known-wrong subset, 1.7B model:

```
true rescue rate (of genuinely-wrong-on-regen): 19.4% (60/309)
tournament's best strategy for comparison:       42.9% (uniform)
already-correct answers clobbered anyway:        11/55 (20%)
avg forced completions per problem:               21.7 (max 39, the depth-3 ceiling)
hit the full depth-3 ceiling:                     221/364 (60.7%)
agreement on the very first pivot:                31.1% (matches Part 1's ~35.5% GSM8K baseline)
```

**Verdict: depth from one starting point loses badly to breadth across
several independent starting points (19.4% vs 42.9%).** The adaptive gating
successfully avoids blind clobbering (confirmed: the earlier flaw doesn't
reproduce here) but still underperforms by a wide margin, because recursing
deeper only ever explores the neighborhood of wherever the first pivot
landed — if that neighborhood is a dead end, no amount of depth escapes it.
The tournament's 5-points-x-5-branches design gets to try genuinely
different regions of the trace, and that breadth matters more than the
depth this experiment added. Consistent with the rest of the project: the
engineered, "smarter" approach loses to the simpler one again, just along a
new axis (depth vs breadth instead of probe vs random/uniform).

Caveat: this used freshly-regenerated traces (do_sample=True, unseeded) for
each of the 364 problems, not the exact same traces as the FINAL p95-budget
tournament run above -- so it's a directionally clear result, not a
perfectly paired comparison. Re-running on the identical saved traces would
tighten this but wasn't judged worth the compute given how large the gap is.

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

## CORRECTED run (2026-10-26, after the off-by-one fix -- supersedes every forced-branch number above)

Ground truth re-collected (900 MATH-L4 train problems, 6300 points, 1500/1300
budgets). All-5-branches-agree rose 44.2% -> 54.3%, mean diversity 0.770 ->
0.616 (the misplaced token was manufacturing disagreement). Diversity probe,
held-out r by layer: 6=.49 10=.49 14=.51 **18=.56** 22=.53 26=.53 28=.53
(best layer moved 14 -> 18). Concatenating all 7 recorded layers: r=.560, no
gain over layer 18 alone (layers are redundant for this target).

Tournament (829 test problems, 377 greedy-wrong scored, layer-18 probe):
uniform 45.4%, random 44.8%, hybrid_spaced_softmax 43.2%, alpha0.25 43.0%,
0.5 42.4%, 0.75 42.2%, pure probe 40.8%, pure noise 40.6%. Branch cap-hit
17.8% (7.6% of correct branches, 18.7% of wrong). Trace cap-hit 20.8%.

Why the probe loses (debug_strategy_trace.py + analysis):
- It's a digit detector: digits are 31% of cuts but 81% of its top-10% scores.
- Global top-5 picks skew late (35% in the last fifth) and clump (median gap
  33 tokens, 25% within 5 tokens).
- Per-cut rescue falls with position: 28% (first fifth) -> 13.6% (last).
- BUT within a position band, diversity does predict rescue (oracle high vs
  low: 25/10, 23/14, 18/6 %; probe: 20/15, 20/14, 14/8 %) -- target is fine,
  selection is the problem.
- Uniform's edge is structural: its cut at token 0 is a 5-way restart that
  alone rescues 32.1%; uniform without it = 36.3%. Its last cut is EOS (wasted).
- "Rescue" mixes restarts, final-answer digit lottery (e.g. forcing "1"->"2" in
  "Final Answer: 18"), and genuine repair. Next tournament should save trace
  ids so rescues can be classified post hoc.

## vLLM advanced tournament (2026-10-26, 56 min on 3 GPUs vs ~7h on HF)

382 scored greedy-wrong MATH-L4 problems, 9 arms, all paired on identical
traces (advanced_tournament_vllm.py, analyze_tournament.py). Rescue rate:
sampled_restarts x25 52.4% (vs uniform +38/-13, p=0.001, but 18.4k tok/problem);
restart+{local_only, product, stratified_probe} 48.2%; zsum 47.9%;
resample_uniform_cuts 47.9% (cheapest: 11.8k tok); restart+uniform_interior 46.6%;
restart+product_guarded 46.3%; uniform 45.8%. No forced-branch arm beats
uniform significantly (best p~0.15).

Combo test (does the outcome probe add anything on top of local entropy?):
product vs local_only +7/-7 p=1.0; zsum vs local_only +6/-7; guarded vs
local_only +12/-19 p=0.28. -> No. Consistent with "Beyond the 80/20 Rule"
(arXiv 2506.01939): local entropy already marks the forks.

Compute-matched best-of-N (sampled restarts up to the arm's own token budget):
49.0% at uniform's budget (vs 45.8%), 50.5% at guarded's (vs 46.3%), 47.9% at
resample_uniform_cuts' (tie). Nothing beat plain resampling at equal compute.

Mechanism: local-entropy arms' rescuing cuts are 56-58% word-level decisions
vs 24-31% for uniform/probe (which rescue largely via formatting-token
"restarts in disguise", 30-35%). Picking meaningful forks does not rescue more
problems than perturbing anywhere -- for small-model math, branch-point choice
barely matters.

Also (validate_vllm.py / diag_vllm.py): ~35% of greedy forced branches change
final answer between HF and vLLM from bf16 kernel differences alone -- branch
outcomes are fragile; treat single-branch labels as noisy.

## Fork-vote tournament (2026-10-26): branching loses decisively once an answer must be chosen

382 scored MATH-L4 problems (fork_vote_tournament.py / analyze_fork_vote.py).
Cuts = token 0 + top local entropy per segment; original-token branch dropped.
Each arm vs best-of-N sampled restarts at the SAME token budget on the same problem:

  arm            any    majority | matched best-of-N any / maj | maj paired
  fork_greedy    46.9%  14.4%    | 45.8% / 22.3%               | +9  -39 p<0.001
  fork_sampled   48.2%  14.7%    | 46.3% / 22.3%               | +12 -41 p<0.001
  fork_resample  47.6%  13.1%    | 45.5% / 20.9%               | +11 -41 p<0.001

Branches landing back on the trace's own wrong answer: 43.3% / 42.4% / 45.8%
vs 32.7% for independent restarts. Sampled continuations / unforced resampling
do NOT decorrelate: the correlation comes from the shared wrong prefix, not
from greedy decoding. Under oracle selection branching is ~1-2 pts ahead;
under majority vote it is ~8 pts behind independent sampling at equal compute.

Also from the vLLM advanced tournament data: forced-token probability rank
doesn't matter (ranks 2-5 each ~10% correct); the original-token branch is
worst (8.6%) and reproduces the trace's wrong answer 51% of the time; spreading
a fixed budget over more cuts with fewer alternatives does not help.

## Fork pruning + 2x2 ablation (2026-10-26) -- earlier "positive result" did NOT survive the ablation

fork_prune_tournament.py (lookahead pruning: 32-token greedy lookahead per
candidate branch, drop branches that rejoin the original path or duplicate a
sibling, complete the best 20) first looked like a win: semprod_pruned 51.6%
"any" vs equal-compute best-of-N 45.8% (+34/-12, p=0.002), and beat both
no-prune arms. BUT the pruned arm also searched 12 candidate cuts vs 5.

fork_prune_ablation.py: 2x2 {semantic-product cuts, evenly spaced cuts} x
{prune, no prune}, every cell 12 cuts + token 0 and 20 completed branches.
382 scored problems:
  semprod_pruned 49.2% any / 18.6% maj; semprod_noprune 48.7 / 14.7;
  uniform_pruned 48.2 / 17.8; uniform_noprune 47.6 / 14.9.
  Pruning effect: any n.s. (p~0.87); MAJORITY +22/-7 p=0.008 (semprod),
    +19/-8 p=0.052 (uniform). Pruning removes re-votes for the wrong answer.
  Heuristic effect (semprod vs uniform cuts): none, p=0.6-1.0 either way.
  vs equal-compute best-of-N: any +32/-24 p=0.35 (the p=0.002 did not
    replicate: wider search + run-to-run noise of 2-3 pts); majority 18.6 vs
    21.7, p=0.10 (gap narrowed from ~8 to ~3 pts).
  Rejoin rate: 35.7% of alternatives at semantic-product cuts vs 45.4% at
    evenly spaced cuts -> the heuristic's cuts do produce more genuinely
    divergent alternatives; it just doesn't change outcomes.

Rejoin detector: exact tail-string match missed whitespace/markdown/LaTeX
variants; normalized match fixed it (verified by reading lookahead dumps).

## Fork-detection benchmark (2026-10-26, 59 wrong traces; extension running)

Ground truth: TV distance of final-answer distributions sampled (16/side)
from just before vs just after a token. Real fork = TV above null 95th pct
(0.56): 10% of random positions. Top-3 pick lift over random: probe 1.01x,
local 1.30x, sem_local 1.07x, product 1.46x, sem_product 1.24x, earliness
1.52x. Every entropy heuristic correlates +0.36-0.41 with position while fork
strength falls with position; position-controlled rank correlations are all
~0 (-0.002..+0.065). Product is the best content heuristic but no better than
"branch early".

## Fork-detection benchmark, final (2026-10-26, 155 wrong traces = 59 + 96 extension)

Extension stopped early for a cooling break (in-flight chunks lost, ~10-15 traces).
Real fork (TV > null 95th pct 0.56) at 8.7% of random positions. Top-3 lift:
probe 0.94x, local 1.44x, sem_local 1.27x, product 1.47x (59/465 picks, ~3 SD
over random), sem_product 1.29x, earliness 1.54x. Position-controlled rho all
+0.02..+0.04. With more data product's edge over local entropy alone vanished
(was 1.58x vs 1.35x at 95 traces): local entropy carries the fork signal, the
outcome probe adds ~nothing, and "branch early" is as good as any of them.

## Divergence gate (2026-10-26, post-hoc on the vLLM tournament's saved traces, 250 integer-answer problems)

gate_addon.py / analyze_gate.py. Candidates per quarter: top-8 by product and by
local entropy; gates over the non-original top alternatives:
  next-step divergence = prob-weighted JS divergence of next-token distributions
  lookahead divergence = prob-weighted share of 16-token greedy lookaheads that
                         do NOT rejoin the original continuation
Results (rescued / majority / solved within 4k tokens):
  product x lookdiv 73.2% / 22.4% / 49.6%  (best branching rescue rate tested)
  local x lookdiv   70.8% / 23.2% / 48.8%
  product x nextdiv 69.6% / 20.8% / 49.2%  (next-step gate doesn't help: paraphrases
                                             differ on the next token, converge later)
  product 71.6 / 20.8 / 49.2; local 71.2 / 19.2 / 48.0; restart+even 68.8 / 20.8 / 48.4
Paired: product x lookdiv vs restart+even rescue +23/-12 p=0.09; local x lookdiv vs
local majority +12/-2 p=0.01 (borderline after ~9 comparisons). Fresh samples still
79.2% / 35.6%. Lookahead divergence is the useful third term; nothing beats resampling.
