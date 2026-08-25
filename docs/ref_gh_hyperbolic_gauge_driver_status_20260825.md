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

Directly storing the raw ordinary-gauge variables was not puncture regular.
Across `16^3` to `48^3`, `|Hhat_A|_inf` grew from `7.48325` to `125.126` and
`|theta_A|_inf` from `19.3337` to `1786.14`; exact `24^3` and `32^3` runs then
failed at approximately `t=0.828` and `t=0.629`. The failure therefore moved
earlier with refinement.

For a static reference, the evolved arrays now store

```
delta_Hhat_A = Hhat_A - Href_A
delta_theta_A = theta_A - theta_ref_A.
```

`Href_A`, `theta_ref_A`, and coordinate derivatives of `Href_A` are
reconstructed analytically from the reference two-jet. The physical driver,
GH source, gauge constraint, and boundary data reconstruct the unchanged raw
variables; finite differencing, KO dissipation, communication, and restart act
on the regular differences. Enabling this representation for a time-dependent
reference fails closed until the required analytic time derivative of
`theta_ref_A` is available.

The review also corrected a pre-existing inverse-coframe derivative index
transpose. A nonsymmetric-frame identity oracle now prevents diagonal or
radial reference frames from hiding that error.

## Puncture-stencil policy

Convergence/error samples are discarded when the puncture lies inside the
axis-aligned support box of the complete relevant stencil. For spacing `h_i`
and stencil radius `s`, a point is excluded exactly when

```
abs(x_i - x_puncture_i) <= s h_i
```

for all three coordinates. Fourth-order centered derivatives have `s=2`, but
the enabled matching KO operator has `s=3`; histories and evolved-state
diagnostics therefore use

```
s = fd_order/2 + (dissipation > 0 ? 1 : 0).
```

The previous radius-two results did not satisfy the stricter full evolution-
stencil interpretation and have been superseded. This policy is used by the stationary-trumpet
field/RHS/native-constraint diagnostics, the Ref-GH history constraint sums,
and the common ADM `H/M` regional sums and maxima. Finite-state and physical
extrema still inspect every active cell. The primary common ADM history uses
no evolving lapse or chi mask; native formulation histories remain secondary.

For interpolated three-resolution comparisons, a target is accepted only if
every tensor-product interpolation source cell has a clear finite-difference
support box. Thus a point is rejected whenever any contributing source stencil
is puncture contaminated, even if the target point itself is outside the
near-puncture region.

## Local results

Exact gauge-enabled Minkowski remains exact through `t=0.2`. The source,
driver, target, `gamma2`, characteristic, and inverse-coframe oracles pass at
roundoff in local Serial/OpenMP builds. A direct two-cycle run and a
one-cycle/checkpoint/restart run have byte-identical final Ref-GH history rows.

With static-reference gauge subtraction, the exact stationary trumpet remains
finite through `t=1` and is a roundoff fixed point:

| Resolution | Initial RHS Linf | Field Linf at `t=1` | Native constraint Linf |
|---:|---:|---:|---:|
| `16^3` | `1.639e-14` | `7.745e-14` | `3.621e-14` |
| `24^3` | `2.345e-13` | `5.477e-12` | `2.172e-12` |
| `32^3` | `3.189e-13` | `4.762e-11` | `1.851e-11` |

These values test preservation of the analytic fixed point. They are too close
to floating-point roundoff to define a truncation-order measurement.

For historical context, before gauge-reference subtraction the unperturbed
stationary trumpet at `t=0.01` showed near-fourth-order fixed-shell behavior.
Between `32^3` and `48^3`, the observed orders were:

| Quantity | `0.5 <= r < 1` | `1 <= r < 1.5` | `1.5 <= r < 2` |
|---|---:|---:|---:|
| Initial RHS | 3.468 | 3.724 | 3.778 |
| Einstein field error | 3.452 | 3.725 | 3.778 |
| Gauge-state error | 3.329 | 3.648 | 3.722 |
| Native constraint | 3.311 | 3.756 | 3.784 |

The perturbed convergence analyzer applies the same conservative maximum
stencil footprint to every contributing interpolation source cell. For a
leading-class-preserving `r^8 exp(-r^2/w^2)` perturbation, the masked
`24^3/32^3/48^3` ladder gives field/native-constraint L2 orders
`4.9122/5.1657` at `t=0.2` and `4.7002/3.9484` at `t=1`. The corresponding
`t=1` Linf orders are `4.7421/3.0151`. At exact common times
`t=0.4,0.6,0.8,1.0`, field L2 orders are `4.438,3.613,3.553,4.700` and
native-constraint L2 orders are `3.163,3.808,3.741,3.948`. This is finite,
resolution improving, and approximately fourth order in L2, but not uniformly
fourth order at every checkpoint.

A current-build `24^3` replay has elementwise-identical final field and
constraint cbin payloads to the pre-correction run. Only diagnostic inclusion
changed: the included history volume decreased from `320.878` to `295.688` and
the final reported native-constraint Linf from `1.7211e-4` to `1.3168e-4`.
The absolute constraints approach the nonzero gauge-violation profile planted
by the perturbation, so only inter-resolution differences are used for the
convergence statement.

The outer faces are at `2M`; measured maximum characteristic speeds are below
`0.610`, giving earliest face-to-`r<1M` arrival estimates later than `1.64M`.
The `t=1` comparison is therefore earlier than this causal estimate. The grids
are uniform and have no SMR interface.

## Preserved failures and interpretation

The production first-order-state puncture exponent estimator passes. The
independent direct-FD estimator converges at fixed physical coordinates, but
does not converge toward the first-order estimator on the prescribed
`2h <= r < 8h` shell even after removing every stencil whose support contains
the puncture. At fixed `r/h`, the stencil samples the same nonsmooth
similarity profile as the mesh is refined; this is not a conventional
fixed-coordinate truncation sequence.

The same-shell gate therefore remains red. It has not been weakened, and the
direct-FD result is not used to repair or replace the first-order state.

The original centered `r^0` Gaussian changes the leading anisotropic puncture
class and is not the controlling physical perturbation. With the corrected
full-stencil mask, however, its `32^3/48^3/64^3` ladder is numerically
resolution improving at `t=0.2`: field and native-constraint L2 orders are
`3.9188` and `4.0296`. The earlier `1.4212` constraint order came from the
incomplete radius-two mask and is superseded.

## Claims not established

No Aurora run was launched. This work does not establish time-dependent or
generic-reference evolution, wormhole-to-trumpet evolution, q control, SMR,
long-time stability, production performance, or a broad convergent-trumpet
claim. The supported claims are narrower: a local static-reference exact fixed
point through `t=1`, and approximately fourth-order masked L2 convergence for
the stated regular uniform-grid perturbation through `t=1`.

The remaining near-puncture estimator decision is whether the independent
direct-FD qualification must remain on a fixed-`r/h` shell (which the present
evidence shows is nonconvergent) or may use a separate fixed physical shell
while the production first-order estimator retains the prescribed
`2h <= r < 8h` sampling.

Compact numerical evidence and reproduction commands are in
`docs/fo_gh_artifacts/ref_gh_hyperbolic_gauge_driver_20260825/`.
