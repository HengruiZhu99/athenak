# Independent pointwise constraint-rate oracle

This source-only analytical preparation supports the finite-rb C0/spatial-norm
control. Production and frozen helper bytes are unchanged. The actual caller
must pass its separately owned source/binding gate before numerical readback.

The exact flat-core commuting-operator proof is `prove_flat_core.py`. It derives
the eight physical constraint rates from the full22 core action and restricts
the algebraic metric/A normals through the complete20 chart. Its displayed
raw22 extension is only the formal extension of those tracefree formulas;
no identification with the actual off-normal constraint functional is made.
The physical metric decomposition gives
`H=divdiv(h_STF)-2 Delta(tau)/sqrt(3)`,
`M=div(A)-2 grad(P+2Theta)/3`, and
`Z=(Lambda-div(h_STF))/2`. The resulting rates are

```
H_t     = -2 div M
M_i,t   = -.5 partial_i H + Delta Z_i - partial_i div Z
          + 2 kappa_input partial_i Theta
Z_i,t   = M_i + partial_i Theta - kappa_input Z_i
Theta_t = H/2 + div Z - 2 kappa_input Theta.
```

The separately owned driver is
`boundary/total-j-finite-rb-control-20261009/run_constraint_rates.py`; its API is
the neighboring `constraint_rate_api.hpp`, included after the actual bridge.
The driver evaluates complete physical polynomial fields directly in Cartesian
coordinates, forms exact polynomial source/constraint jets in the core, and
compares those with the actual source, actual constraint derivative and frozen
subsidiary. No `r^L` division or finite differences across the `.05` transition
are used in that exact core gate.

At the fixed transition/collar points, the driver separately samples the actual
full22 source and physical eight constraints, forms their fourth-order Cartesian
jets at five frozen h values, and applies the respective actual constraint
derivative and coefficient-aware subsidiary. It retains every sequence,
increment, extrapolation, absolute/scaled discrepancy and failed attempt. Sample
RMS/peaks are not integrated Penrose norms or a conserved energy. Boundary owns
the independent radial assembly, moving-frame contractions and volume norms.

This checks a stationary-reference linear continuum action. It does not identify
a generic nonlinear closure, assert discrete Bianchi closure, classify a frozen
primitive root, or establish boundary stability.
