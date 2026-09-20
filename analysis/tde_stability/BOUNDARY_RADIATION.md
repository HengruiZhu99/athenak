# Experimental constraint-radiation boundary: vacuum gate failed

The suggested differential boundary is implemented, but **it is not a stability fix**. Controlled perturbed-trumpet runs develop a new rapid outer-face instability. The option remains off by default; no TDE executable, input, job or production queue was changed. The original campaign's slower residual mode also remains unresolved. No spinning evolution has been started.

## Implementation and cancellation guarantee

`<z4c>/boundary_rhs=characteristic_cpbc` with `characteristic_bc_source=physical_constraint_radiation` enables the diagnostic candidate. `characteristic_radiation_areal_shift=1` uses the reference radiation weight R=r+1 for the centered M=1 trumpet. It is an outgoing face-normal approximation, not a generic exact Kerr absorbing condition.

The helper forms physical Theta and covariant Z differences using identical full/reference evaluations, including the metric-defined contracted connection and its exact discrete time derivative. The boundary changes incoming characteristic rates in the momentum-like variables; metric RHS neighbors are immutable, RHS ghosts are never read, and face/edge/corner output ownership is disjoint. No threshold resets, clipping or global synchronization was added.

An initial composed-derivative accuracy defect was corrected: fourth-order inner metric derivatives restore second-order consistency of the composed boundary operator. This accuracy repair does not remove the instability. Default-path outputs were bitwise identical to the baseline in matched regressions. Exactly zero residual and all recorded ghost cells remain exactly zero through the tested stages. Perturbed outputs are bitwise repeatable with OpenMP1/2 and MPI1/4 on the fixed eight-block mesh. This establishes the tested equilibrium/repeatability properties, not perturbation stability, GPU equivalence or new refinement-interface coverage.

The reusable stage regression is `tst/regression/z4c_constraint_radiation.py`; it requires a candidate executable, immutable baseline, sixth-order trumpet input and a new output directory. Add `--launcher mpiexec --ranks 1 4 --threads 1` for the MPI comparison. The helper's 15 compiled manufactured/Jacobian/poisoned-ghost checks and convergence data are archived under [cartesian-design](evidence/boundary-radiation/cartesian-design/DESIGN.md).

## Why the new candidate is rejected

With dx = 0.25M, dt = 0.0375M, a 1e-8 lapse pulse, G = 2, kappa1 = 0.1 and kappa2 = 0, and no sponge:

| Candidate | First invalid ghost metric | Evolution outcome |
|---|---:|---|
| v2 cubic ghosts |13.725M|boundary validity abort after last progress18.75M|
| v2 linear ghosts, eight blocks |19.8375M|active invalid state22.0125M|
| v2 linear ghosts, one block |19.4625M|active invalid state21M|
| v2 quadratic ghosts, one block |16.575M|active invalid state20.025M|
| v2 second-order volume control |44.0625M|active invalid state45M|
| v2 flat background, cubic ghosts |none|reached50M, maxTheta1.56e-10|
| v2 flat background, linear ghosts |none|reached50M, maxTheta1.74e-8 and growing|

The second-order case uses PLM without FOFC to satisfy its ghost-zone requirements; it is a discretization discriminator, not production validation. Earlier input rejections are recorded separately. See [comparison plot](evidence/boundary-radiation/pilot-comparison.png), [full run summary](evidence/boundary-radiation/RESULTS.md), and exact first-event records in [pilot-summary.json](evidence/boundary-radiation/pilot-summary.json).

At the first RK stage in the initial implementation, the pre-boundary Theta RHS maximum is 3.57e-11 at (0.625, 0.125, -0.125)M, inside the horizon. The boundary raises the maximum to 3.45e-10 at (1.875, -0.125, -0.125)M, on the outer face. These are initial truncation injection and immediate boundary amplification of a nonzero perturbation. Later invalid outer ghosts are a separate event.

The complete v2 RK map amplifies a face-dominated direction by 1.46338 over 0.3M. Physical H/M/Q amplify by 1.46471/1.45952/1.45710; this is not only a Theta or ghost-norm artifact. The 24-vector Ritz direction is **not a converged eigenvector**, so its estimated growth rate is not reported as a measured eigenvalue. [Full-map analysis](evidence/boundary-radiation/modes-v2/README.md).

An independent frozen Cartesian model reproduces a shifted, oblique face mode with gamma = 1.2128/M at the actual boundary lapse/shift; its eigenpair residual is 6.1e-14. About 99.94% of its state norm is in the outermost three cells. Removing normal shift nearly removes this fast branch, and the old zero-rate closure lacks this branch in the same reduced model. Growth increases roughly as 1/h at fixed grid tangential frequency. This implicates the new boundary closure's shifted multidimensional compatibility. The finite-grid result alone neither establishes continuum ill-posedness nor identifies the original production mode's cause; the separate continuum test below addresses the new prototype. [Frozen-model report](evidence/boundary-radiation/fd-symbol/RESULTS.md).

Linear ghost extrapolation has a second, independently verified defect: applying the unchanged sixth-order second derivative to q=x^2 at the first active cell gives -7/30 instead of 2, independent of h. Quadratic ghosts fix that consistency defect but not the actual trumpet failure.


A subsequent continuum half-space calculation validates a positive mode at 1.2307116367/M for transverse k = 2pi/M, using an ordered-Schur decaying subspace. The reconstructed 20-field profile satisfies the bulk equations to 7.84e-14 and the boundary conditions to 1.66e-16. Without damping, its growth scales linearly with transverse frequency. Theta, H, M and Q are near 1e-14, but electric/magnetic Weyl norms are 0.145/0.0103: this is a constraint-satisfying mixed physical/gauge boundary mode, not a pure coordinate mode. Thus its frozen principal continuum boundary limit admits arbitrarily fast growing modes; ghost discretization is not the only problem. This calculation omits background gradients and the lower-order areal radiation weight. The finite-grid counterpart also excites measurable constraints. This result concerns the new prototype, not the old slower campaign mode. [Continuum validation](evidence/boundary-radiation/continuum-halfspace/RESULTS.md).

## Rejected alternatives and remaining work

Reduced correction strength, direct Theta/A constraint allocation, tangential gauge/tensor terms, higher-order polynomial ghosts, matched D6 derivatives and a more complete frozen G2 gauge treatment all retain growing modes. The published formulations and gauge conditions differ; these tests are explicitly labeled adaptations rather than evidence against the published methods. [Derivation and audit](evidence/boundary-radiation/PUBLISHED_BOUNDARY_PLAN.md).

The next step is a compatible complete gauge/constraint/radiation boundary closure, checked first with the continuum symbol and then with a discrete estimate. The radial constraint-only decay result is insufficient. No additional Aurora allocation was spent on a candidate already known to fail locally. A cure must still pass long perturbed vacuum, resolution, MPI/GPU and refinement tests before matter or spin promotion.

## Liu wormhole preparation

`src/coordinates/kerr_liu.hpp` provides axis-regular Cartesian Liu-Etienne-Shapiro Kerr wormhole geometry with analytic derivatives. The standalone tests cover 105 points, both sheets, the throat, axes and spins 0, 0.5 and 0.9. Hamiltonian/momentum errors are about 3e-15/3e-16; independent sixth-order finite differences converge as expected. At a/M = 0.9, the coordinate horizon diameter is 0.717944947M, so dx = 0.0625M gives 11.49 cells across it. [Geometry tests and model distinction](../../tst/unit/kerr_liu/README.md); [measured evidence](evidence/liu-geometry/geometry-results.json).

Identical full/reference RHS evaluations cancel algebraically at the chosen reference. Full-stage equilibrium preservation for the unregistered Liu provider remains untested. With the positive precollapsed Liu lapse, however, the physical continuum geometry is nonstationary. Subtracting that finite RHS changes nearby constraint propagation; it is a reference-forced model, distinct from physical moving-puncture relaxation. Both are useful future controls once the vacuum boundary issue is addressed. The provider is not registered in the problem generator, and no high-spin evolution is claimed.

A separate nonspinning isotropic-wormhole vacuum control using the existing problem generator was tried as a zero-background-shift discriminator. It first invalidated a fourth corner ghost at 67.3875M and aborted at 77.025M. This was the same coarse dx = 0.25M grid (only four cells across its horizon), with the unchanged r+1 leading radiation model rather than the exact wormhole areal radius. It is a rejected coarse control, not a resolved wormhole test or a vacuum pass. [Failure record](evidence/boundary-radiation/wormhole-control/failure-summary.json).
