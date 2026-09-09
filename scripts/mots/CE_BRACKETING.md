# Native VC Cartoon constant-expansion checks

`search_checkpoint.py --ce-bracket` adds a three-surface check to the ordinary
zero-target search. This is analysis of a saved slice, not an evolution. It
retains the existing ordinary candidate policy and outermost selection.

For each completed full-angular-order central trial with RMS at most the
existing promotion bound (0.5), its independently measured area A0 fixes
Rref=sqrt(A0/(4*pi)). Starting from that same center and harmonic shape, two
continuations solve Theta_plus-c with q=c*Rref reaching exactly -0.05 and +0.05.
Intermediate q steps use a fixed Rref. L is capped at 64 for bracket checks.
Nonzero targets never set ordinary horizon or restart-success flags.

The direct solver, flow projection, numerical Newton Jacobian and merit checks
all use target error. Physical outgoing and ingoing expansion are preserved.
CE convergence retains the strict solver tolerances (defaults 1e-6 RMS and
1e-5 maximum dimensionless target error). A stalled CE solve is reported as
such, even if the resulting surfaces supply a numerically signed sandwich.

Separate outputs mean:

- `ordinary_candidate`: existing zero-target policy.
- `supported`: positive smooth radial graphs with verified geometric nesting,
  sampled physical signs on independent dense grids including poles, and sign
  margins larger than the reported angular uncertainty estimates.
- `narrow`: maximum sampled proper radial-connector width, with quadrature
  sensitivity allowance, is at most 0.25 Rref. This is a localization diagnostic,
  not a condition for existence of the numerically signed sandwich. Broad
  sandwiches set `localization_warning`; neither q nor tolerances are adjusted.
- `ce_converged`: strict CE target-error convergence, independent of `supported`.
- `stability_operator_verified` and `spatially_validated`: false unless separate
  qualification is implemented. A sandwich alone does not establish these.

Nesting uses radial ordering at the shared center with harmonic derivative
bounds between angular samples, rather than area ordering. Floating-point
bounds and angular sampling are numerical evidence, not rigorous proofs.
Proper separation is measured along radial connectors in the spatial metric,
not shortest geodesic/normal distances. Refining connector quadrature estimates
integration error. Neither this nor angular oversampling bounds spatial errors.

`*.ce_brackets.jsonl` records every family, all target steps, extrema, norms,
areas, coefficients, signs, nesting, uncertainty and separation diagnostics.
`*.ce_surface_*_{inner,central,outer}.csv` contains physical expansions versus
theta, including poles. `*.ce_events.csv` records ordinary and bracket detection
independently, their first times, and the outermost bracket-supported family.
The frozen command manifest records checkpoint/executable hashes and source SHA.

Standalone `--ce-target C --ce-reference-radius R` is restricted to frozen mode
and can use `--seed FILE --seed-only`; it verifies the ordinary restart carrier
is unchanged. Target C is dimensional; q=C*R is recorded separately.

For bounded live tests, `restart_runtime.py --outer05 --ce-bracket` preserves the
restart's physics and authenticated live-AMR ledger prefix. It disables stopping
on ordinary horizons/dispersion and enables `stop_on_mots_bracket`. Add
`--cadence 32 --checkpoint-cadence 32` to save every scheduled finder slice;
verify the actual coordinate-time spacing against the declared resolution target.
`--tlim` and `--cycle-limit` bound the test. Do not use failed CE searches as a
classification of dispersion. A current detection always runs the bracket before
its separate stopping condition is considered.
