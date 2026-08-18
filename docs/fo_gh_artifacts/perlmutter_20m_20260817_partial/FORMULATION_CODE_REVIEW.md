# FO-GH puncture failure: formulation and code-review handoff

## Findings first

### Critical: the FO-GH system fails the resolution ladder

All resolutions made valid 2M restarts, then the characteristic timestep
collapsed at 3.431611M, 3.024995M, and 2.658676M as `dx_min` decreased from
1/16 to 1/32. Native masked Hamiltonian growth precedes each collapse. The
reversed failure-time ordering contradicts numerical stability and makes long
evolution unjustified. It does not alone prove continuum ill-posedness.

### High: independently derive the gauge-driver subsystem

`src/fo_gh/fo_gh_rhs.hpp` evolves

```
dt h = beta^i d_i h - mu_H (h-f) + vartheta
dt vartheta = -eta_H (beta^i d_i h + vartheta).
```

The second equation uses advection of `h`, not `vartheta`. Documentation and
unit tests encode the same expression, so they cannot validate its derivation.
Derive the intended covariant driver, rescalings, and principal symbol, then add
an independent Fourier-mode regression.

### High: constraint additions and principal symbol are unqualified

The regularized `K`, `Atilde`, `Lambda`, and `pi` equations contain
sign-sensitive `C_perp`, `c^i`, Hamiltonian, and divergence-of-`c` additions.
Pointwise tests reproduce documented equations but do not establish equivalence
to a strongly hyperbolic damped FO-GH system. Derive the principal symbol and
constraint propagation independently, especially the `+kappa alpha c^i`
Lambda term with `c^i=-Lambda^i+Gamma^i`.

### High: common ADM momentum evidence is unusable

All common fixed-region momentum histories are exactly zero, while native
FO-GH momentum is nonzero. Audit `src/coordinates/adm_constraints.cpp`, the
FO-GH ADM adapter/update cadence, tensor symmetries, and ghost data using a
manufactured nonzero-momentum ADM field.

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

Independently derive the regularized evolution, driver, characteristic fields,
and constraint propagation; add symbolic/manufactured and Fourier tests;
diagnose common ADM momentum; and add first-invalid-state telemetry. Only then
rerun the bounded FO-GH 32/48/64 ladder. Do not tune damping or resume long/Z4c
runs before confirmed defects are addressed.

