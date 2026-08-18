# Reference-frame FO-GH puncture formulation status

This branch implements a separate vacuum 50-field first-order GH system in a
fixed frame: ten symmetric `Psi_ab`, ten `Pi_ab`, and thirty `Phi_iab`. The
frame/coframe and reference connection are prescribed rather than evolved; the
regular stationary trumpet state is `Psi=eta`, `Pi=Phi=0`.

With `Pi=-n^a partial_a Psi` and `Box Psi=S`, the lower-order `Pi` terms are
`-Phi_i D^i(alpha) + alpha S`. The implemented principal symbol has a complete
50-vector analytic eigenbasis at all audited radii from 1/8 through 1/64: rank
50, condition number one, and roundoff residual.

The n=2 Schwarzschild trumpet provider directly interpolates tabulated value,
first-, and second-derivative profiles. This improved the interpolation audit's
first- and second-derivative errors to `7.13e-11` and `4.46e-8`.

## Blocking defect

The exact regular state does not approach zero semidiscrete RHS near the
puncture:

| dx | closest radius | initial RHS Linf | reference Ricci Linf |
|---:|---:|---:|---:|
| 1/16 | 0.05413 | 4.786e-9 | 1.372e-7 |
| 1/24 | 0.03608 | 8.442e-8 | 7.210e-7 |
| 1/32 | 0.02706 | 1.357e-7 | 2.316e-6 |

The maximum is always `Pi00`. The trend localizes the failure to near-puncture
coordinate/reference-jet cancellation rather than the characteristic basis or
table interpolation floor. Two source-balancing experiments worsened it and
were reverted, with their commits retained for review.

Therefore **REFERENCE-GH FORMULATION NOT ESTABLISHED**. The wormhole transition
was intentionally not implemented or run after this hard stop.
