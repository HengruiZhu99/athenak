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

## Paused stationary `t=20` ladder (not a gate pass)

The subsequent four-A100 campaign used the same CUDA-aware-MPI build and was
intentionally stopped at the user's request.  The `dx=1/16` and `dx=1/24`
stationary cases completed `t=20` with bad-state flag zero and effective CFL
0.05; their final square-root-of-integrated GH/reduction norms were
`4.19e-12`/`3.27e-10` and `8.68e-12`/`6.03e-10`, respectively.  The `dx=1/32`
case was cleanly cancelled after checkpoint `.00002`; its final written history
row is at `t=4.00184777`, also with bad-state flag zero and CFL 0.05, but has
GH/reduction norms `8.29e-11`/`1.81e-10`.

Thus all observed runs remained finite, but these common norms do not show a
resolution-improving trend.  The fine case has not reached `t=20`, so the
three-resolution stationary gate has **not** passed and no conclusion about
long-time puncture stability or formulation success is justified.  This is an
observation requiring source/formulation review, not a diagnosed cause: no
parameter, threshold, clipping, floor, or source term was changed to improve
it.  The compact histories, an exact table of the final rows, and reviewer
questions are in the artifact directory.

The time-dependent wormhole-to-trumpet reference remains out of scope until
the stationary gate is resolved.

## Curvature-conditioning diagnosis and correction

The pause review isolated the non-improving stationary histories to a
table-faithful binary64 reference-curvature error, not to nonzero `Q`,
`Delta`, damping, or a source-sector ambiguity.  At the closest off-grid cells
of the 64/96/128 ladder, the old generic coordinate-2-jet/Cartan construction
produced frame-Ricci Linf values `6.54e-11`, `2.54e-10`, and `6.88e-10`.
For the exact regular state its curvature sector was the complete source
residual; the other four source sectors vanish.

The stationary reference is exactly Schwarzschild, so its Riemann tensor in
the Eulerian Cartesian orthonormal frame is known analytically.  Commit
`b5594291` supplies that full vacuum Weyl tensor from the interpolated areal
radius, with electric part
`E_IJ = M/R^3 (delta_IJ - 3 n_I n_J)` and zero magnetic part.  This is neither
background-source subtraction nor projection of an approximate curvature:
the full analytic Riemann remains available to the covariant source for
perturbed states.  The generic Cartan spin and spin-derivative construction is
retained for the scalar-frame correction.

The new table-faithful audit records a roundoff-scale analytic Ricci trace and
source at all three closest cells (source Linf at most `5.68e-16`), while
preserving the independently measured raw-curvature discrepancy.  A clean
serial C++ 64^3 initialization also gives RHS Linf `1.17e-16` and frame-Ricci
Linf `3.33e-16`; its coordinate-Ricci diagnostic remains `3.76e-10`, as
expected from the deliberately cancellation-prone coordinate calculation.
The 96^3/128^3 executable ladder and all post-change flat/long-time gates are
still required; this correction does not retroactively pass them.
