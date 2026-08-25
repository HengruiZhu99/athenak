# Ref-GH hyperbolic gauge-driver local evidence (2026-08-25)

This compact bundle records local CPU qualification of the 61-field Ref-GH
state through base commit `6019e77f` and the full-stencil diagnostic correction
at `a1a0c928`. It intentionally contains no restart file or field dump.

Completed evidence:

- inverse-coframe derivative, gauge-driver frame, ordinary-gauge source,
  physical-target, `gamma2`, and combined-characteristic oracles pass at
  binary64 roundoff;
- exact gauge-enabled Minkowski remains exact through `t=0.2`;
- exact stationary trumpet is a roundoff fixed point through `t=1` at `16^3`,
  `24^3`, and `32^3`;
- direct versus checkpoint/restart evolution has identical final Ref-GH history
  rows in the focused 61-field restart gate;
- regular `r^8 exp(-r^2/w^2)` perturbed-trumpet runs at `24^3/32^3/48^3`
  remain finite through `t=1` and are approximately fourth-order in masked L2
  self-differences.

The exact fixed point required an equation-preserving storage change for the
ordinary gauge state. Raw `Hhat_A` and `theta_A` are singular at the puncture;
the evolved arrays store differences from the analytic static-reference
values. Physical sources and constraints reconstruct the unchanged raw fields.
Time-dependent-reference subtraction still fails closed because the needed
analytic time derivative has not been implemented.

## Full evolution-stencil puncture mask

For fourth-order Ref-GH, centered derivatives reach two cells while enabled KO
dissipation reaches three. The original radius-two mask therefore omitted part
of the evolution footprint. Histories, problem diagnostics, and offline
convergence now use radius three when KO is enabled:

```
stencil_radius = fd_order/2 + (dissipation > 0 ? 1 : 0).
```

The offline analyzer rejects a target when any tensor-interpolation source cell
has a support box containing the puncture. This is deliberately conservative.
A current-build replay changed only diagnostic inclusion: final field and
constraint cbin arrays were elementwise identical to the earlier run, while
the included history volume decreased as expected.

| Case | Samples | Field L2 order | Field Linf order | Constraint L2 order | Constraint Linf order |
|---|---:|---:|---:|---:|---:|
| `r^0`, `32/48/64`, `t=0.2` | 1833 | 3.919 | 3.330 | 4.030 | 3.621 |
| `r^8`, `24/32/48`, `t=0.2` | 252 | 4.912 | 5.321 | 5.166 | 5.680 |
| `r^8`, `24/32/48`, `t=1.0` | 252 | 4.700 | 4.742 | 3.948 | 3.015 |

At exact common times `t=0.4,0.6,0.8,1.0`, the regular perturbation's field L2
orders are `4.438, 3.613, 3.553, 4.700`; native-constraint L2 orders are
`3.163, 3.808, 3.741, 3.948`. The sequence is resolution improving and
approximately fourth order, but not uniformly fourth order at every time.

The outer faces are at `2M`. With measured characteristic speeds below `0.610`,
the earliest face-to-`r<1M` arrival is later than `1.64M`; the `t=1` result is
not attributed to the boundary. These are uniform grids, so no SMR interface
is present.

## Open red gates and limitations

The first-order-state puncture exponent estimator passes its pointwise analytic
checks. The independent direct-FD estimator still cannot converge toward it on
the prescribed fixed-`r/h`, `2h <= r < 8h` shell: its singular-power stencil
bias is scale invariant. This hard gate has not been weakened, so closed-loop
`q` control has not started.

The centered `r^0` perturbation changes the leading anisotropic puncture class.
Its corrected masked numerical convergence is recorded, but it is not promoted
as the physically appropriate regular perturbation; `r^8` remains controlling.

No Aurora/PVC execution, time-dependent/generic-reference gauge subtraction,
q control, wormhole evolution, SMR, long-time stability, or broad trumpet-
stability claim is included. The strongest result is a local static-reference
fixed point through `t=1` plus approximately fourth-order masked uniform-grid
L2 self-convergence for the regular perturbation through `t=1`.
