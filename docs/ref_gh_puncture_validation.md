# Reference-frame FO-GH puncture validation report

## Gate decision

**REFERENCE-COVARIANT GH NOT ESTABLISHED.**

The repaired stationary t=0 source gate now passes, but the post-correction
three-resolution stationary evolution through `t=20` and the required
wormhole-to-trumpet transition ladder have not been run.  The Perlmutter
production campaign remains paused.  No long-time puncture-stability or
transition-success claim is made here.

## 1. Original baseline

At the original coordinate-source baseline, the `dx=1/16,1/24,1/32`
initial-RHS ladder was `4.786e-9`, `8.442e-8`, and `1.357e-7`; its maxima moved
inward and were dominated by `Pi00`.  The accompanying coordinate reference-
Ricci diagnostic was `1.372e-7`, `7.210e-7`, and `2.316e-6`.  This is the
immutable control in
`fo_gh_artifacts/reference_covariant_repair_20260818/baseline_control.tsv`.

Two rejected experiments—transformed-background subtraction and an added
coordinate background-wave identity—were not retained.  No mask, floor,
excision, reset, clipping, or weakened threshold has been introduced.

## 2. Jet-only repair

The provider now interpolates the regular primitives `alpha(y)`, `R(y)`, and
`q(y)=beta^r/r` with a single quintic Hermite polynomial per interval, where
`y=ln(r/M)`.  Values and first/second derivatives therefore originate from one
numerical function; `psi^2=R/r` and Cartesian-shift derivatives are then
reconstructed analytically.  The table/interpolation audit and the coordinate-
source-only ablation are retained as `hermite_jet_audit.json` and
`jet_only_coordinate_ablation.tsv`.  The coordinate path remained
conditioning-sensitive, so the jet fix alone was not treated as the final
source repair.

## 3. High-precision coordinate source

The 80-decimal direct-implicit oracle evaluates the coordinate/product-rule
source at radii `1/8` through `1/128` without using the binary64 table.  Its
residual tends to arbitrary-precision zero even while the individual coordinate
intermediates grow from `3.33e4` to `4.22e9`.  Thus the continuum coordinate
algebra is consistent but numerically ill-conditioned near the puncture; see
`high_precision_coordinate_oracle.json`.

## 4. Covariant-source oracle

The production source uses frame-native `Q=bar-nabla g`, connection difference
`Delta`, the reference Riemann, quadratic sectors, GH damping, and the scalar
frame correction.  The high-precision covariant trumpet oracle gives
`Q=Delta=0`, maximum scalar-source residual `9.68e-73`, and maximum frame-
Ricci residual `4.84e-73`.  The independent random-state flat/non-flat source
oracle has maximum frame-vs-coordinate mismatch `1.11e-15`.  These results
are in `high_precision_covariant_trumpet_oracle.json` and are documented in
`ref_gh_covariant_source_repair.md`.

After that audit, a compiled generic-state test exposed an initialization-order
defect in the production construction of raised `Delta^A_BC`: rows of
`Delta_DBC` could be read before they were filled. The lower tensor and the
raised tensor are now constructed in separate passes. The new 1,000-jet flat
source unit test changes from a pre-fix mismatch of `1.14718e-4` to
`6.93889e-17`. Its evidence and a focused remote-review request are in
`fo_gh_artifacts/reference_covariant_repair_20260818/delta_upper_source_repair/`
and `ref_gh_remote_review_delta_upper.md`. This makes all prior evolution
evidence pre-delta-upper-fix; it is not a new stability result.

## 5. Stationary t=0

The generic Cartan curvature trace was found to be the remaining binary64
forcing sector.  The stationary reference now supplies the full analytic
Schwarzschild Weyl tensor in its Eulerian Cartesian orthonormal frame, while
retaining generic spin/spin-derivative data for the scalar correction.  This
is an exact prescribed-reference curvature, not a background-source
subtraction.

The post-correction executable `64^3/96^3/128^3` ladder has RHS Linf
`1.17e-16`, `1.59e-16`, `1.46e-16`; frame-Ricci Linf
`3.33e-16`, `4.44e-16`, `5.00e-16`; and zero final regular-field/native-
constraint norms.  The coordinate Ricci grows inward (`3.76e-10`, `9.14e-09`,
`5.33e-08`) but is secondary only.  Exact commands, hashes, and compact logs
are in `fo_gh_artifacts/reference_covariant_repair_20260818/`
`stationary_analytic_vacuum_t0*`.

## 6. Stationary evolution

Pre-curvature-correction four-A100 stationary histories reached `t=0.1` and
`t=1` at all three resolutions, and the coarse/medium cases later reached
`t=20`; the fine case was deliberately stopped at `t=4.00185`.  Because that
campaign predates the exact-curvature correction and its long-time common
norms were not resolution-improving, it is **not** a post-correction pass.
The campaign is paused, the three-resolution post-correction `t=20` gate is
incomplete, and no stability classification follows.  See the `perlmutter/`
pause handoff for the exact historical rows and checkpoint provenance.

## 7. Transition evolution

The time-dependent wormhole-to-trumpet reference transition has not been
implemented or run after the stationary source repair.  It remains explicitly
out of scope until the post-correction stationary `t=20` gate passes.

## 8. Aurora corrected stationary gate

The attempted frozen-source Aurora `t=1` gate did not supply stationary
evolution evidence. Two corrected 64-cell runs on distinct Aurora nodes each
completed source/diagnostic initialization with a roundoff-scale stationary
RHS and a byte-identical clean t=0 history, then the Level Zero runtime aborted
on a GPU write fault at cycle 0. No positive-time sample, medium/fine run, or
`t=20` continuation exists. Thus convergent trumpet evolution remains **not
established**; the evidence cannot distinguish any formulation behavior because
execution stopped before an evolved step. See
`fo_gh_artifacts/reference_covariant_repair_20260818/aurora_stationary_frozen/`
for the scheduler, mapping, history, and failure records.
