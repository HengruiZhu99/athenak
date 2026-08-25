# Ref-GH hyperbolic gauge-driver local evidence (2026-08-25)

This compact bundle records the first local CPU qualification of the 61-field
Ref-GH state at source commits `c6181505` and `693cd000`. It intentionally
contains no restart file or field dump.

Completed evidence:

- inverse-coframe derivative identity on a nonsymmetric frame: maximum error
  `2.77556e-17`;
- improved gauge-driver frame oracle: `5.55112e-17`;
- ordinary-gauge source product-rule oracle: `1.38778e-16`;
- physical target oracle: `1.38778e-17`;
- combined Einstein/gauge characteristic map and inverse: `1.66533e-16`;
- exact gauge-enabled Minkowski through `t=0.2`: zero maximum error;
- exact stationary-trumpet fixed-shell discriminator through `t=0.01` at
  `16^3`, `24^3`, `32^3`, and `48^3`;
- 61-field two-cycle direct versus one-cycle/checkpoint/restart equivalence:
  byte-identical final Ref-GH history rows. The omitted checkpoint SHA-256 was
  `c3eab8109bf65f456e636954564678b339472904cc1492831affddbea13d0b60`.

Open red gate:

- The first-order-state puncture exponent estimator passes, and the independent
  direct-FD estimator converges at fixed physical coordinates. The direct-FD
  estimator does **not** converge toward the first-order estimator on the
  prescribed `2h <= r < 8h` shell. Its wormhole difference changes from
  `0.236831` to `0.443482`, and its trumpet difference from `0.0175538` to
  `0.0200649`, between `h=1/16` and `h=1/64`.

The red result is not hidden or used to alter the evolved state. At fixed
`r/h`, a stencil samples the same nonsmooth similarity profile at every
resolution; removing only stencils whose support contains the puncture does
not make this diagnostic a fixed-coordinate truncation test.

No Aurora execution, `t=1` convergence, generic-reference evolution, wormhole
evolution, long-time stability, or trumpet-stability claim is included.
