# Bjorhus CPBC boundary discriminator

## Qualification status

The planned four-way comparison is incomplete and is not a qualified CPBC
comparison. Job 57636959 was intentionally cancelled at user request and no
further work was submitted.

Actual scheduler state at cancellation was:

| case | boundary/domain | retained result |
|---|---|---|
| A | Sommerfeld, Rout16 | reached t=6.5 |
| B | experimental CPBC, Rout16 | failed closed at t=3.244461 |
| C | Sommerfeld, Rout128 | reached t=6.5 before cancellation |
| D | experimental CPBC, Rout128 | intentionally terminated at t=1.508791; excluded |

C completed before the cancellation request could take effect. It is preserved
for provenance but does not make the A/B/C/D comparison complete. D has partial
history only and is excluded from scientific comparison.

## Implementation defect found and repaired

The first nonlinear CPBC replay exposed a concrete frame-construction bug:
`MakeFullConstraintBjorhusFrame` raised each normal component before all
covariant components were initialized. Cartesian tests masked the defect. The
fix in commit `d39822c6522688749fe5ead8025907bc055f02f8` separates initialization
from index raising and adds a non-diagonal-metric regression test. The strict
incoming-residual gate was not weakened. A corrected CUDA replay crossed the
previous fatal events and reached t=0.05 with zero residual-gate hits.

## Retained observations

- All three CUDA manufactured/policy tests passed on the final executable.
- A remained healthy to t=6.5.
- B developed a physical-boundary corner instability. At its last history row,
  C was about 3.63e3 and the maximum was at `(rho,z)=(16,-16)`.
- B then failed the strict characteristic-speed gate because the boundary no
  longer had one incoming and one outgoing physical-speed member.
- Over the common early central interval, A and B differed in
  `log10(abs(axisKret))` by at most `8.4e-7`, despite the boundary-local runaway.
- C was healthy to t=6.5. D was cancelled too early for a boundary signal or a
  stability conclusion; its apparent zero central deviation is not evidence.

The sparse Theta/Gamma compatibility projector cancels four incoming principal
constraint rates but cannot also preserve all paired outgoing rates. The
manufactured case records a maximum induced outgoing-rate change of 0.686111;
this is not a reflection coefficient.

## Claim boundary

This evidence does not show that CPBC reduces boundary contamination and does
not mathematically rule out boundary effects in the main collapse. It does show
that the present experimental CPBC is not production-stable at Rout16.

For the main Rout128 KO=0.5 campaign, the early localization and resolution
behavior strongly deprioritize the physical outer boundary relative to a
bulk/AMR rho approximately 5M instability. “Deprioritize” is not “exclude.”

Plots and compact data are under `analysis/cpbc/`.
