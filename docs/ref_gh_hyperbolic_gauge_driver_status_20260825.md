# Ref-GH ordinary hyperbolic gauge-driver status — 2026-08-25

## Implemented

The Ref-GH state now contains 61 fields: the original 50 Einstein fields plus
ordinary, reference-independent `Hhat_A` (4), `theta_A` (4), and coordinate
`Upsilon^i` (3). The implementation follows the improved
Lindblom--Szilagyi driver, includes the exact moving-frame terms, drives toward
true advective 1+log and the first non-advective conformal Gamma-driver target,
and couples the evolved ordinary source into the covariant Ref-GH wave source.

The standard `gamma0` gauge-constraint damping and `gamma2` first-order
reduction damping are both active in the controlling stationary and perturbed
trumpet inputs. Combined Einstein/gauge characteristic fields and their
inverse are device tested. Existing array-driven MPI, AMR, output, and restart
paths now carry all 61 fields.

The review also corrected a pre-existing inverse-coframe derivative index
transpose. A nonsymmetric-frame identity oracle now prevents the diagonal or
radial reference frames from hiding that error.

## Puncture-stencil policy

Convergence/error samples are discarded when the puncture lies inside the
axis-aligned support box of the complete local centered stencil. For spacing
`h_i` and stencil radius `s`, a point is excluded exactly when

```
abs(x_i - x_puncture_i) <= s h_i
```

for all three coordinates. This policy is used by the stationary-trumpet
field/RHS/native-constraint diagnostics, the Ref-GH history constraint sums,
and the common ADM `H/M` regional sums and maxima. Finite-state and physical
extrema still inspect every active cell. The primary common ADM history uses
no evolving lapse or chi mask; native formulation histories remain secondary.

## Local results

Exact gauge-enabled Minkowski remains exact through `t=0.2`. The source,
driver, target, characteristic, and inverse-coframe device oracles pass at
roundoff. A direct two-cycle run and a one-cycle/checkpoint/restart run have
byte-identical final Ref-GH history rows.

For the unperturbed stationary trumpet at common time `t=0.01`, fixed shells
away from the puncture improve toward fourth order. Between `32^3` and `48^3`,
the observed orders are:

| Quantity | `0.5 <= r < 1` | `1 <= r < 1.5` | `1.5 <= r < 2` |
|---|---:|---:|---:|
| Initial RHS | 3.468 | 3.724 | 3.778 |
| Einstein field error | 3.452 | 3.725 | 3.778 |
| Gauge-state error | 3.329 | 3.648 | 3.722 |
| Native constraint | 3.311 | 3.756 | 3.784 |

These are short local discriminators, not a `t=1` convergence result.

## Preserved failure and interpretation

The production first-order-state puncture exponent estimator passes. The
independent direct-FD estimator converges at fixed physical coordinates, but
does not converge toward the first-order estimator on the prescribed
`2h <= r < 8h` shell even after removing every stencil whose support contains
the puncture. At fixed `r/h`, the stencil samples the same nonsmooth
similarity profile as the mesh is refined; this is not a conventional
fixed-coordinate truncation sequence.

The same-shell gate therefore remains red. It has not been weakened, and the
direct-FD result is not used to repair or replace the first-order state.

## Claims not established

No Aurora run was launched. This work does not establish the required `t=1`
stationary/perturbed convergence, generic-reference evolution, wormhole to
trumpet evolution, long-time stability, SMR behavior, production performance,
or convergent trumpet evolution. The next scientific decision is whether the
independent direct-FD qualification must remain on a fixed-`r/h` shell (which
the present evidence shows is nonconvergent) or may use a separate fixed
physical shell while the production first-order estimator retains the required
near-puncture `2h <= r < 8h` sampling.

Compact numerical evidence and reproduction commands are in
`docs/fo_gh_artifacts/ref_gh_hyperbolic_gauge_driver_20260825/`.
