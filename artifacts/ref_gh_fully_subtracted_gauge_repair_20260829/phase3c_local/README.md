# Phase-3c compact analytic residual-source checkpoint

The analytic radial-q path now has a compact, cancellation-free evaluator for
the ordinary-GH Einstein gauge increment. It uses the generated contraction's
linearity in `g_inverse` and `d_g_inverse` to evaluate the upper-index
`Delta B` directly, then lowers and differentiates it with exact residual
product identities. It does not materialize reference Christoffel, spin, or
Riemann tensors and is not yet dispatched from production `CalcRHS`.

The source-unit matrix compares this compact path with the independent generic
residual implementation. Exact matched `q=1` sources are bitwise zero at every
expanded radius. Perturbed states pass the unchanged conditioned-radius gate;
the all-radius compact-versus-generic diagnostic is `3.05105e-12`, located at
`r=0.03M`. This inner discrepancy is preserved for the Phase-4 coefficient
conditioning/asymptotics audit rather than hidden by a larger tolerance.

No evolution or production-readiness claim follows from this checkpoint.
