# Composed constraint-diagnostic attribution

The earlier private composed-derivative experiment changed the evolution's
diagonal second derivative from Dxx4 to Dx4 followed by Dx4, while deliberately
retaining the original constraint diagnostic. This audit reads the same saved
binary64 states with both diagnostics. It performs no evolution or projection
and changes no accepted source, stored state, original history or receipt.

Two fresh standalone Release utilities differ by one private Diagnose loader
statement. Both retain the archived radius-four symmetric quadratic continuation
and analytic Minkowski reference jets. Only diagonal second derivatives of
deviations change; values, first derivatives and mixed derivatives are identical.
The composed operator is not applied to the full analytic reference field.

## Independent identity and snapshot checks

Let g be the twice-conformal metric, chi the conformal factor and b=g/chi the
Penrose spatial metric. With delta denoting composed minus original Hessians,

```
delta(b_ij,dd) = delta(g_ij,dd)/chi - g_ij*delta(chi,dd)/chi^2,
delta R[b] = sum_dij (b^di*b^dj-b^dd*b^ij)*delta(b_ij,dd),
delta H = Omega^2*delta R[b].
```

All connections and lower-derivative Ricci terms agree. Momentum depends only
on first derivatives of the spatial metric, A, chi and P+2Theta; Z depends on
Lambda and the metric connection, and Theta is stored. Hence all three must
agree pointwise under this diagnostic change.

Eighteen utility executions cover nine stored states. The original diagnostic
reproduces every selected HST H/M/Z/Theta exactly. The independent delta-H
identity agrees within 1.177e-14 at every active cell, and M/Z/Theta agree
pointwise exactly.

| Stored state | Exact time | Original H RMS | Composed H RMS |
| --- | ---: | ---: | ---: |
| N24 pulse | .02 | .003536276640 | .001836759768 |
| N24 pulse near .2 | .20017968749999435 | .02733392267 | .01680304227 |
| N24 pulse | 2 | 1.154724072 | .7826542322 |
| N36 pulse | .2 | .01355636738 | .003553582506 |
| N24 reference | .05 | 8.57096636e-14 | 3.60253541e-14 |

Initial H remains about 6e-15 under both loaders. The diagonal Hessian choice
therefore explains part of the previously reported H magnitude. It cannot
explain the unchanged N24 t2 M/Z/Theta values
1.401967977 / .3650457565 / .02886699640. These are functional comparisons on
identical states, not evidence of improved dynamics or an energy bound.

## Matched diagnostic-time comparison

N24 diagnostic RMS values are interpolated between saved times
.17537109374999502 and .20017968749999435 to t=.2. No field is interpolated.
N36 uses its exact t=.2 state.

| Composed functional | N24 interpolated | N36 exact | N24/N36 | log-ratio / log(1.5) |
| --- | ---: | ---: | ---: | ---: |
| H | .01678042991 | .003553582506 | 4.722116 | 3.828337 |
| M | .03206624257 | .007134943120 | 4.494253 | 3.706360 |
| Z | .01332279235 | .002090460982 | 6.373136 | 4.567820 |

Near that time the actual timesteps are .000427734375 and .000115104166667.
Only two resolutions are available. These descriptive ratios are consistent
with improved early-time discrete consistency, but do not establish pure
spatial order or long-duration convergence. Flat linear closure requires both
composed RHS and composed constraint functionals; this audit proves no nonlinear
or variable-coefficient discrete product rule, boundary energy estimate, exact
scri closure or black-hole transition.

## Provenance

The original private evolution executable remains
`e337453cae3553d05393478204f91131c620cb84c469e5c1a6fce3db073cfa16`,
with original build receipt
`f2c884662b04b68b4efef0cb1dedb2e11b26f242683427642d233afbf433c886`.
Its recorded repository dependencies, original link inputs, private objects,
headers and executable are verified unchanged before and after the audit.
Production remains implementation27c19d20 with the Minkowski hyperboloidal
reference throughout.

The [immutable archive](validation/hyperboloidal-lapse-and-composed-diagnostic-experiments-20261009/README.md)
contains exact paired sources/commands/logs, selected original inputs/history,
the independent identity and both eight-state and final nine-state receipts.
The diagnostic sub-index SHA256 is
`45e12a0beba75a3b58de595d6b7843c447f4170b856a11ffb0107857379b5c0e`;
large state/raw/cell arrays and executables are indexed by hash/size only.
The same archive separately records the rejected
[lapse-advection candidate](hyperboloidal-lapse-advection-audit.md).
Use its verifier read-only and preserve all original frozen evidence.
