# Brill local telegrapher-scale investigation

This is the compact, Git-sized review bundle for the four matched AthenaK
Figure-3 telegrapher experiments.  It accompanies source commit
`2a8ad80e02279769a99fe279b7a33516bc6c8d0d` on branch
`codex/cartoon-allbulk-brill-scaleinv-20260813`.

Read in this order:

1. `RESULTS.md`
2. `REMOTE_REVIEW_PROMPT.md`
3. `data/analysis_summary.json`
4. the five comparison plots under `figures/`
5. the implementation and tests named by the prompt

The exact IrisK coefficient payload, AthenaK input, rendered published curves,
campaign result JSON, reduced history curves, selected AthenaK mu profiles,
analysis scripts, PNG/PDF plots, and the preceding project-status report are
included.  Native AthenaK
binary dumps and build products are intentionally excluded; their hashes and
terminal disposition are bound in the included summaries.

The central scaling convention is

```text
Kstar      = max_domain |K|
Q          = mu / Kstar
tau_eff    = tau / Kstar
kappa_eff  = kappa / Kstar
Q/tau_eff  = mu/tau
kappa_eff/tau_eff = kappa/tau
```

The implementation evaluates the two cancelled ratios, avoiding division by
zero at time symmetry.  It does not add an extra `Kstar` multiplier to a local
field.  No run used coefficient tuning, floors, clipping, smoothing, or a
weakened chi gate.

The prior PDF under `prior_context/` predates this four-mode experiment.  It is
included only for broader Cartoon/Brill context; `RESULTS.md` and
`data/analysis_summary.json` are authoritative for this comparison.
