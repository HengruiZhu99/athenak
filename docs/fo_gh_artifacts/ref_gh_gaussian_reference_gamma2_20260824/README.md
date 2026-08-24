# Figure-backed Ref-GH report artifacts

This directory contains generated figures and compact provenance for
`docs/ref_gh_gaussian_reference_gamma2_20260824.tex`.

The three T4/T5 figures are derived only from committed histories under
`../ref_gh_feedback_continuation_20260823/`.  The Gaussian profile figure is an
analytic design illustration, not simulation evidence.  Regenerate all plots
with:

```bash
python3 scripts/ref_gh/plot_ref_gh_gaussian_gamma2_report.py
```

Large Aurora field dumps and restart files are intentionally not copied here.
Their locations are recorded in the report and the parent campaign handoff.
