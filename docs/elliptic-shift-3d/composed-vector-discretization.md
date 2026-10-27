# Unadopted discrete-gauge operator: consistency and conditioning

This standalone prototype differentiates the actual global metric rate rather
than expanding continuum derivatives before discretization. It changes no
production default, field, gauge, boundary value, or evolved Z4 connection.
See [the qualification of all prior residual claims](discrete-gauge-qualification.md).

For fixed stage metric jets, construct all six nodal tensor components
H_h(delta beta), with the actual vector Dirichlet trace and Hermite scri jets.
Extend those rates through the same active-only global metric halos and surface
interpolation. Apply the exact metric-connection Frechet variation using D_h H_h.
The known source uses the same procedure on -2 alpha A + H_h(beta_ref), where
beta_ref=-x and H_ref=-x dot D_h g. It is not zero for a nonconstant metric.
No physical RHS residual is subtracted. A saved accepted-state comparison with
the independent audit agrees in every cell: RMS3.60e-14, max5.90e-13.
The actual discrete rate remains2.202, so this agreement confirms the failure.

N6/two-shell, curved-metric affine and nonlinear continuum MMS solves using the
existing scalar multigrid preconditioner all fail after120 iterations. Their
final declared residual RMS ranges6.998e-5--6.301e-4. The affine derivative option
also has a separately demonstrated surface angular cancellation artifact and
remains disabled. A single eight-parity-conjugated scalar-MG preconditioner
control worsens nonlinear RMS to0.06651 after80 iterations; it is rejected.

Periodic fourth-order centered D1 has symbol

    i sin(theta) (4-cos(theta)) / (3 h).

Thus -D1 squared has a Nyquist kernel. Relative to a second-order scalar-Poisson
inverse, its symbol is cos(theta/2)^2 (4-cos(theta))^2/9, whereas direct
fourth-order -D2 gives1+sin(theta/2)^2/3. Real alternating-vector tests at
N32/64/128 give exactly zero D1 and D1-squared action, while direct D2 remains
nonzero. This is an independent periodic control, not a claim of an exact null
on the nonperiodic mapped domain.

A42-mode weighted orthonormal trial space includes smooth fields, global
logical parity, Cartesian-localized parity and thin-shell modes in three
Cartesian polarizations. Actual full operator actions are used for its normal
Gram matrix. Seven-layer N6/N8/N10 right-MG restricted condition estimates are
15.16/14.12/13.28 for the expanded operator and26.60/257.73/871.39 for the composed
candidate. Their smallest composed singular values are.24479/.11512/.07333;
the weak trial modes are predominantly Cartesian parity. These are restricted
estimates, not full spectra, spectral-gap bounds or proofs of global nullity.
One/two-shell and inner-refined N6/N8 controls are retained separately; the
refined N10 case would exceed the50,000-cell budget and was not run.

Independent80-step double-MGS Arnoldi controls go beyond the initial trial
space. Expanded AP reaches roundoff by32 directions, while composed AP remains
at relative residual.167 after80 unrestarted directions. A fresh physical
vector z has Rayleigh quotient <z,Az>/<z,z>=-857.3184. A separate full-action
reconstruction verifies the sign; this is not merely a projected-matrix result.
The independent [energy audit](composed-energy-audit.md) finds E=-.00361952
against positive strain energy Q=+.00157119. Cartesian cells and the first
transition layer account for99.71% of their global difference. Regional
contributions include legitimate cross-region fluxes and do not individually
prove an interface identity failure. The global negative energy is decisive
for noncoercivity in this norm. It does not alone establish singularity or a
mathematical obstruction to an alternative qualified discretization.

The current priority is interface differentiation/transfer consistency and
compatible preconditioning. Radial interpolation that omits interior Cartesian
donors and one-face angular extrapolation have separately identified large
amplification defects. New geometry-only transfer prototypes must pass all-row
value/derivative/noise guards before their energy, spectrum, MMS and coupled
checks. No simply composed centered derivative is being silently adopted.

Evidence directories: `composed-metric-first-20261027`,
`composed-metric-accepted-20261027`, `composed-masked-first-20261027`,
`composed-spectrum-{first,rightmg,family,analysis}-20261027` and
`composed-arnoldi-20261027`. Source capsules preserve exact numerical source,
commands and child exit codes. Early generic runner scope labels described
manufactured solves even for no-solve spectrum/accepted-state controls; a
separate provenance note corrects those labels without changing archived data.
The original wrappers returned zero after recording failed child cases; current
wrappers propagate those failures. Child exit1 and the failed residual targets
remain explicit in every scientific report.
