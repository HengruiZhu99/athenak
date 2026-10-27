# Continuum equation versus discrete metric-connection freezing

This qualification applies to every earlier solver-residual claim in this
folder, including native, MPI-reference, manufactured, restart and coupled
reports. Their archived numbers remain unchanged. A small reported "original
vector residual" certifies the declared expanded continuum-equation
finite-difference operator, independently of its multigrid preconditioner.
It does **not** certify exact freezing of the connection computed by the actual
metric difference operator. No coupled wormhole-to-trumpet scientific milestone
has passed.

Let the actual metric connection evaluator be

    V_h(g) = F(g, D_h g).

Its instantaneous variation under the actual metric RHS is

    R_freeze = DV_h(g)[C_h + H_h(beta)] + eta V_h(g).

This derivative applies the same global metric halos and actual evolved scri
metric rates. The implemented expanded continuum operator instead expands
products and uses direct mapped first/second metric and shift derivatives.
Discrete products do not obey the continuum Leibniz rule; direct D2 is not
D1 composed with D1. Positive log reconstruction can also differ from first
differencing reconstructed nodal physical values. Consequently its reported
residual R_exp and R_freeze differ by spatial/interface truncation terms that
must be measured, not assumed small. Full variable-metric coefficients and the
independent evolved Z4 connection are retained in both audits.

The final accepted N6 failed affine wormhole state at t=.0002 has expanded
RMS6.13349e-11 and point9.70e-10. Direct evaluation of the actual discrete
metric-built V_t gives physical-volume RMS2.20200, conformal RMS2.84287 and
point15.2275. A finite-difference metric variation at epsilon1e-4 agrees with
this actual Frechet rate within3.46e-10 RMS. The decomposition closes to
1.79e-14 RMS. Its largest contribution is the direct-second versus composed-first
shift derivative (RMS2.31192); product-rule RMS0.147876 and physical-lapse-chain
RMS0.003894 are recorded separately. The leading points lie in the thin shell
r=.65--.70, not at the puncture. Complete region and arbitrary nonspherical
finite-variation controls are being qualified separately.

A fresh default affine-OFF solve on exactly the same geometry converges in16
iterations to expanded RMS5.384e-11, but its discrete V_t RMS is2.20317. Thus the
dominant bulk mismatch is inherited from the prior discretization. The affine
candidate also substantially worsens the surface discrete rate, and remains off.
These are fixed-state controls, not additional evolved trajectories.

A separate unadopted prototype constructs actual nodal H_h(beta), extends its
six tensor components through the same global/surface halos, and applies the
exact metric-connection variation with D_h H_h. It leaves the physical problem
and evolved fields unchanged. Its first N6 curved-metric affine and nonlinear
manufactured solves fail the unchanged residual targets after120 iterations,
with or without the affine derivative representation. Final RMS ranges from
7.0e-5 to6.3e-4 rather than1e-10. This exposes preconditioning/conditioning work;
it is not a replacement for the passing expanded operator.

Centered first derivatives vanish at a periodic Nyquist mode, so simply
composing them can introduce a checkerboard kernel. Actual nonperiodic mapped
halos can alter that kernel and cannot be presumed to cure near-null modes.
The initial mapped checkerboard probe remains nonzero, which proves neither a
spectral gap nor safe conditioning. Low/high-frequency and near-null controls,
independent arbitrary nonspherical metric-rate variations, refinement and
coarse/fine interface consistency are required before adopting any replacement.
No gauge change, Lambda projection, metric projection, floor or physical RHS
subtraction is authorized as a numerical fix.

Evidence: `discrete-connection-first-20261027`,
`discrete-connection-baseline-20261027`, and `composed-metric-first-20261027`
under `evidence/`. The first two are independent no-evolution audits of a saved
accepted state; the last is a standalone manufactured candidate and explicitly
fails. The38 passing engineering CTests in
`affine-integration-regression-20261027` do not override this scientific failure.

The completed [independent audit](discrete-connection-consistency-audit.md)
includes two arbitrary nonspherical metric-rate directions, nonunit determinant,
N6/N8/N10 and refined-core controls, and an exact zero-trace affine surface
falsifier. [Composed operator controls](composed-vector-discretization.md) retain
periodic kernels, actual mapped trial-space refinement, negative physical-energy
witnesses and rejected preconditioner attempts. No replacement is qualified.
