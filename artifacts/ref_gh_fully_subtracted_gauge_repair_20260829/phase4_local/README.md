# Phase-4 stationary-trumpet coefficient audit

This directory preserves the arbitrary-precision audit of every gauge-sector
coefficient family required by the fully subtracted q=1 formulation.  The
calculation uses the implicit trumpet solution rather than the generated
binary64 table and differentiates with respect to the stored frame variables
`Psi`, `Pi`, `Phi`, `delta_H`, and `Upsilon`.

The exact lapse exponent is

`p = 1.091297104795417177142734699770404899977`.

All 26 fitted coefficient envelopes agree with their analytic powers to a
maximum error of `1.3239722512508953e-4`, below the hard `5e-3` gate.  The
identity checks use a `1e-45` threshold; their largest error is
`2.3738919364399497e-65`.  Repeating all target Jacobians with centered
difference epsilons `1e-30` and `1e-24` gives identical TSV output and
identical powers at all 40 JSON digits.

The audit distinguishes cancellation from genuine coefficients.  Pure
reference forcing is zero to the arbitrary-precision identity gate, but the
continuum residual system still contains nonuniform lower-order maps.  In
particular, `delta(beta)^i Kref_iA` per stored `Psi` grows as `r^(-2p)`, and
the complete same-stage Einstein gauge-source maps grow as

* `Psi -> source`: `r^(-3p)`;
* `Pi -> source`: `r^(-2p)`;
* `Phi -> source`: `r^(-(3p+2))`;
* `Upsilon -> source`: `r^(-(3p+1))`;
* `delta_H -> source`: `r^(-2p)`.

Thus the standard principal symbol remains unchanged at every `r>0`, but the
ordinary unweighted lower-order energy bound is not uniform as the puncture is
approached.  This audit does not prove a uniformly equivalent weighted
symmetrizer; any such claim requires a separate construction.

No production dispatch, fixed-point evolution, robustness, or performance
claim follows from this artifact.
