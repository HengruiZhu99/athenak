# Brill local telegrapher-scale investigation

This is the compact, Git-sized review bundle for four matched AthenaK
Figure-3 telegrapher-scale experiments and one controlled Gamma-driver shift
damping diagnostic.  It accompanies source commit
`2a8ad80e02279769a99fe279b7a33516bc6c8d0d` on branch
`codex/cartoon-allbulk-brill-scaleinv-20260813`.

Read in this order:

1. `figure3_shift_damping_report.pdf`
2. `RESULTS.md`
3. `REMOTE_REVIEW_PROMPT.md`
4. `data/analysis_summary.json` and `data/fixed_shift_eta2_comparison.json`
5. the original five telegrapher plots and four fixed-shift plots under
   `figures/`
6. the implementation and tests named by the prompt

The exact IrisK coefficient payload, AthenaK input, rendered published curves,
campaign result JSON, reduced history curves, selected AthenaK mu profiles,
analysis scripts, PNG/PDF plots, the new seven-page report, and the preceding
project-status report are included.  Native AthenaK
binary dumps and build products are intentionally excluded; their hashes and
terminal disposition are bound in the included summaries.

The fixed-shift control changes exactly
`z4c/shift_eta_max_K=true -> false`: the baseline uses
`eta_shift=2 max|K|`, while the control holds `eta_shift=2`.  It remains on the
same Figure-3 initial data, telegrapher lapse, N128/O6/RK4 grid, AMR, KO,
damping, and strict-chi settings.  The control avoids the baseline chi gate and
runs to `t=16.73892M`, but later fails a distinct axis-central diagnostic.  It
is evidence about the baseline stiffness, not a successful reproduction or a
recommended dimensionful gauge.

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

The prior PDF under `prior_context/` predates both experiments.  It is included
only for broader Cartoon/Brill context.  `RESULTS.md`,
`data/analysis_summary.json`, and `data/fixed_shift_eta2_comparison.json` are
authoritative for the current comparison.
