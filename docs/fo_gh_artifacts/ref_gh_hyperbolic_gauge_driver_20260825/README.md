# Ref-GH hyperbolic gauge-driver local evidence (2026-08-25)

This compact bundle records local CPU qualification of the 61-field Ref-GH
state at source commits `c6181505`, `693cd000`, `1ff3321b`, and `2579e5d5`.
It intentionally contains no restart file or field dump.

Completed evidence:

- inverse-coframe derivative identity on a nonsymmetric frame: maximum error
  `2.77556e-17`;
- improved gauge-driver frame oracle: `5.55112e-17`;
- ordinary-gauge source product-rule oracle: `1.38778e-16`;
- physical target oracle: `1.38778e-17`;
- combined Einstein/gauge characteristic map and inverse: `1.66533e-16`;
- exact gauge-enabled Minkowski through `t=0.2`: zero maximum error;
- exact stationary-trumpet fixed point through `t=1` at `16^3`, `24^3`, and
  `32^3`, with field and native-constraint errors at roundoff;
- 61-field two-cycle direct versus one-cycle/checkpoint/restart equivalence:
  byte-identical final Ref-GH history rows. The omitted checkpoint SHA-256 was
  `c3eab8109bf65f456e636954564678b339472904cc1492831affddbea13d0b60`;
- a leading-class-preserving `r^8 exp(-r^2/w^2)` perturbation through `t=0.2`
  at `24^3`, `32^3`, and `48^3`: field L2 order `3.9425` and native-constraint
  L2 order `4.6395` after the puncture-overlap mask. The corresponding Linf
  orders are only `3.3504` and `1.4868` and remain a limitation.

The exact fixed point required an equation-preserving storage change for the
ordinary gauge state. Raw `Hhat_A` and `theta_A` are singular at the puncture;
the evolved arrays now store their differences from the analytic static
reference values. Physical sources and constraints reconstruct the same raw
fields. Time-dependent-reference subtraction fails closed because the needed
analytic time derivative has not yet been implemented.

Puncture-overlap policy:

- every source diagnostic discards a cell when the complete centered
  finite-difference support box contains the puncture;
- the offline convergence analyzer also discards a target when any tensor
  interpolation source cell has such a contaminated support box;
- this is deliberately more conservative than masking only by target radius.

Open red gates and limitations:

- The first-order-state puncture exponent estimator passes, and the independent
  direct-FD estimator converges at fixed physical coordinates. The direct-FD
  estimator does **not** converge toward the first-order estimator on the
  prescribed `2h <= r < 8h` shell. Its wormhole difference changes from
  `0.236831` to `0.443482`, and its trumpet difference from `0.0175538` to
  `0.0200649`, between `h=1/16` and `h=1/64`.
- The older centered `r^0` Gaussian changes the leading anisotropic puncture
  class and yields only `1.4212` native-constraint L2 order on the masked
  `32^3/48^3/64^3` ladder.

The red results are not hidden or used to alter the evolved state. At fixed
`r/h`, a stencil samples the same nonsmooth similarity profile at every
resolution; removing stencils whose support contains the puncture does not
make this diagnostic a fixed-coordinate truncation test.

No Aurora execution, generic/time-dependent-reference evolution, wormhole
evolution, SMR, long-time stability, or broad trumpet-stability claim is
included. The exact `t=1` result is a static-reference fixed-point test, while
the regular perturbed result establishes fourth-order L2 convergence only for
the stated local uniform-grid experiment through `t=0.2`.
