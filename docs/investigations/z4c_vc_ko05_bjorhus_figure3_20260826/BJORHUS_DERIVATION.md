# Full-field Z4c Bjorhus boundary closure

The implementation and complete derivation are maintained in
[`../z4c_full_bjorhus_cpbc_20260826/BJORHUS_DERIVATION.md`](../z4c_full_bjorhus_cpbc_20260826/BJORHUS_DERIVATION.md).
This file records the interpretation used by the Brill campaign.

The default-off option is

```text
<z4c>/boundary_rhs = full_constraint_bjorhus
```

It leaves the volume RHS intact except at physical outer-boundary points.  In
the local conformal-normal frame, it forms four principal incoming
constraint-rate rows associated with `Theta` and the three components of
`Z_i`, solves the nonsingular four-by-four map for corrections to the
`Theta`/`Gamma^i` RHS entries, and applies those sparse corrections.  The
Cartoon axis is excluded and the existing low-order one-sided boundary-normal
derivative policy is retained.

This is a zero-incoming-rate compatibility projection, not a proof of a fully
well-posed CPBC for the nonlinear discretized system.  With only four sparse
RHS corrections it cannot, for generic incoming data, simultaneously cancel
four incoming rows and preserve all four paired outgoing rates.  The
manufactured incoming test measures a nonzero induced outgoing-rate projection
of `0.686111111111...`; the report therefore calls this implementation
"experimental full-constraint Bjorhus compatibility" and does not describe it
as exact outgoing preservation.

The production definitions use the full evolved fields, including

```text
DeltaGamma^i = Gamma_evolved^i - Gamma_metric^i
Z_i = 0.5 gtilde_ij DeltaGamma^j
```

with no analytic background, residual variables, SAT term, or residual gauge
evolution.
