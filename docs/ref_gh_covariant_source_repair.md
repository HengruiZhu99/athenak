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
regressions, stationary evolution, and the time-dependent wormhole-to-trumpet
reference are still required before a formulation-success claim.
