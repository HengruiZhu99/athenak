# Reference-covariant FO-GH source repair

## Current diagnosis

The original stationary trumpet residual was a lower-order coordinate-source
cancellation problem, compounded by an interpolation provider that returned
independently interpolated value, first-derivative, and second-derivative
tables.  The repair uses one quintic Hermite polynomial per interval for the
regular primitives `alpha(y)`, `R(y)`, and `q(y)=beta^r/r`, with `y=ln(r/M)`.
The physical-looking `psi^2=R/r` and Cartesian shift derivatives are rebuilt
analytically from that one jet.

The production lower-order source is now the reference-frame covariant
connection-difference form: `Q=bar-nabla g`, `Delta=Gamma[g]-bar-Gamma`, the
frame Riemann term, quadratic Q and Delta sectors, GH damping, and the scalar
frame-connection correction.  The legacy coordinate source remains selectable
only as `ref_gh/source=coordinate_oracle`; `covariant` is the default.

## Hard algebra evidence

`high_precision_covariant_trumpet_oracle.json` evaluates the direct implicit
n=2 trumpet at 80 decimal digits, without the binary64 table, at all prescribed
radii from `1/8` through `1/128`.  It finds exact-arithmetic `Q=Delta=0` and
maximum scalar covariant-source and frame-Ricci residuals of `9.68e-73` and
`4.84e-73`.  In contrast, the legacy coordinate source intermediate grows from
`3.33e4` to `4.22e9` over the same radii while its arbitrary-precision residual
vanishes.  This establishes correct continuum algebra but severe coordinate
conditioning.

The independent random-state oracle covers 1000 flat Lorentzian samples and
64 curved diagonal/off-diagonal/shift/generic references.  Its largest
frame-vs-coordinate source mismatch is `1.11e-15`.

## Stationary t=0 result

The clean-build three-resolution covariant ladder is recorded in
`stationary_covariant_t0.tsv`.  The regular-state RHS is
`4.57e-12`, `1.58e-11`, and `9.29e-12` for `dx=1/16,1/24,1/32`: it is below the
`1e-10` target and has no inward resolution reversal.  The frame Ricci remains
bounded below `7e-10`; the coordinate Ricci grows inward and is retained only
as the predicted conditioning diagnostic.  All final field and native
constraint norms are zero for the exact constant regular state.

This establishes only the algebra and stationary-initial-data gates.  Flat
regressions now also pass: exact Minkowski remains exactly zero, linear-wave
L1 orders are 3.913 and 3.946, and robust-Minkowski perturbations decay at
8, 16, and 32 cells.  See `flat_regression_covariant.tsv`.

## Short multi-GPU stationary and restart gates

On Perlmutter allocation 57256611, the CUDA/MPI build ran one rank on each of
four distinct A100 GPUs.  The exact stationary trumpet stayed finite through
`t=0.1` and `t=1.0` at `dx=1/16,1/24,1/32`.  At `t=1`, the maximum regular
field errors were `2.52e-12`, `5.37e-12`, and `5.74e-12`; native constraint
Linf values were `6.60e-12`, `1.48e-11`, and `2.56e-11`.  The bad-state flag
remained zero in every history.  These are bounded, roundoff-scale stationary
results, not a long-time or puncture-stability claim.

The first restart attempt exposed a concrete plumbing defect: the Ref-GH
50-field state was absent from restart-file serialization and deserialization.
Commit `e162917c` repairs both paths.  A fresh four-rank checkpoint at
`t=0.1023608699` now restarts to `t=0.25` with the same history values as the
uninterrupted run, to the printed precision.  Pre-repair checkpoints contain
no Ref-GH state and are invalid; they were not used as evidence after the
repair.

Compact Perlmutter histories and rank-to-GPU evidence are under
`docs/fo_gh_artifacts/reference_covariant_repair_20260818/perlmutter/`.  Raw
restart files (about 205 MB each) are intentionally excluded.

The `t=20` stationary gate and the time-dependent wormhole-to-trumpet reference
remain required before a formulation-success claim.  In particular, the
successful short stationary and restart checks do **not** establish puncture
stability.
