# legacy/

Earlier versions of the pipeline, kept for the research trail in NOTES.md:
the HF-generate tournaments and their variants, the pre-fix monotonic-noise
run, the first collection script, probe exports for earlier targets, and
one-off pilots and benchmarks.

These aren't maintained. Their imports point at `forkprobe`, but several still
expect globals from the old token-tree app (a preloaded GSM8K split, a probe
loaded at import) or data files from the old flat layout, so most won't run
as-is. The current versions of everything the article reports are in
`experiments/`.
