# Remote read-only review prompt

Repository: `https://github.com/HengruiZhu99/athenak`

Branch: `codex/cartoon-allbulk-brill-scaleinv-20260813`

Review directory:
`docs/investigations/brill_r16_resolution_isolation_20260815/`

Numerical source commit:
`2a8ad80e02279769a99fe279b7a33516bc6c8d0d`

Please perform a read-only mathematical and source review of the R=16 Brill
`A=-0.047` N128/N256 resolution-isolation experiment.  Read `README.md`,
`data/analysis_summary.json`, both reduced histories, the exact input deck,
terminal result/log evidence, and the three comparison figures.  Inspect the
numerical source at commit `2a8ad80e` rather than assuming the documentation
commit was the executable source.

Assess whether the near-common collapse times and shared sequence of growing
curvature/constraints, deeper AMR, and timestep collapse genuinely favor a
formulation or gauge-scale instability over inadequate resolution.  Check the
max-|K|-scaled telegrapher lapse (`tau=kappa=1`), fixed-eta Gamma-driver
coupling, dynamic timestep and AMR criteria for a plausible feedback or stiff
mode.  Explain whether the different final fatal lines can reasonably be two
manifestations of the same preceding instability.

Please clearly separate observations, mathematical inferences, hypotheses,
and recommended tests.  Do not propose floors, clipping, threshold relaxation,
or retrospective sample selection.  Recommend the single highest-information
next experiment or bounded diagnostic, preferably one that distinguishes a
telegrapher/formulation mode from AMR-trigger feedback without a large sweep.
