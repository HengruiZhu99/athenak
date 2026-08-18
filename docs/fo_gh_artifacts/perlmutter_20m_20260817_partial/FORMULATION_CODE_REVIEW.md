# FO-GH puncture failure: formulation and code-review handoff

## Findings first

### Critical: the FO-GH system fails the resolution ladder

All resolutions made valid 2M restarts, then the characteristic timestep
collapsed at 3.431611M, 3.024995M, and 2.658676M as `dx_min` decreased from
1/16 to 1/32. Native masked Hamiltonian growth precedes each collapse. The
reversed failure-time ordering contradicts numerical stability and makes long
evolution unjustified. It does not alone prove continuum ill-posedness.

### High: independently qualify the gauge-driver subsystem

`src/fo_gh/fo_gh_rhs.hpp` evolves

```
dt h = beta^i d_i h - mu_H (h-f) + vartheta
dt vartheta = -eta_H (beta^i d_i h + vartheta).
```

The second equation uses advection of `h`, not `vartheta`. A direct audit found
that the core regularized 3+1 equations and moving-puncture targets map to Brown,
*Generalized Harmonic Equations in 3+1 Form* (arXiv:1109.1707), but this separate
relaxation driver is neither Brown's algebraic gauge prescription nor the full
Lindblom et al. wave-driver system (arXiv:0711.2084). Its frozen, fixed-target
Fourier subsystem is stable, but the coupled principal symbol and constraint
propagation remain unqualified. Derive those independently and add a regression.

### High: constraint additions and principal symbol are unqualified

The regularized `K`, `Atilde`, `Lambda`, and `pi` equations contain
sign-sensitive `C_perp`, `c^i`, Hamiltonian, and divergence-of-`c` additions.
Pointwise tests reproduce documented equations but do not establish equivalence
to a strongly hyperbolic damped FO-GH system. Derive the principal symbol and
constraint propagation independently, especially the `+kappa alpha c^i`
Lambda term with `c^i=-Lambda^i+Gamma^i`.

### High, confirmed and repaired: common ADM histories are invalid

All common fixed-region momentum histories are exactly zero, while native
FO-GH momentum is nonzero. The cause was confirmed in
`src/coordinates/adm_constraints.cpp`: `Gamma_udd(c,a,b)` read
`Gamma_ddd(d,a,b)` before all `d` components were initialized, and `DK_udd`
made the same one-pass error with `DK_ddd`. This can corrupt both common H and
M2. The history reduction additionally used `fmax(0,M2)`, which can turn a NaN
into zero. The repair splits both operations into fill-then-raise passes,
initializes point tensors, preserves invalid M2 as a visible NaN, and adds two
mesh-backed manufactured tests: a non-diagonal constant flat metric with
`K_yy=z` (`H=0`, `M2=25/24`) and an exactly flat curvilinear metric
(`H=M2=0`). The original ordering fails all 64 active cells; the repaired
Release/Serial test passes. The committed Perlmutter common-ADM histories
predate the repair and must not be used or reanalyzed scientifically.

### Medium: invalid metric states become an endless timestep stall

`src/fo_gh/fo_gh_tasks.cpp::NewTimeStep` inverts `gtilde` without checking a
finite positive determinant. Exploding fields can yield zero/subnormal `dt`
and continue at fixed physical time. Add fail-closed metric/finite checks and
record extrema at the first collapse. This is a diagnostic deficiency, not the
established cause.

### Medium: finalization-only summaries miss cancelled failures

`src/pgen/fo_gh/puncture.cpp::CheckFoGhPuncture` writes its compact summary at
finalization, so stalled/cancelled runs lack first-bad-state extrema. Emit a
small diagnostic record at every restart/checkpoint time.

## Evidence-based exclusions and next gate

Boundary arrival is excluded for onset on the 32M half-domain. GPU mapping,
CUDA-aware MPI, restart continuity, and one/eight-rank agreement passed.
Memory stayed below 40GB. No Z4c production result exists.

Independently qualify the remaining driver, characteristic fields, and
constraint propagation; add Fourier tests and first-invalid-state telemetry.
Then rerun only a bounded FO-GH ladder and regenerate the common-ADM evidence.
The repaired diagnostic did not feed the evolution, so it cannot explain the
timestep collapse. Do not tune damping or resume long/Z4c runs before the
remaining formulation gate is addressed.
