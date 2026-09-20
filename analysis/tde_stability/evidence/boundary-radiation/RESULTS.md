# Constraint-radiation boundary investigation

The requested boundary prototype is implemented in the isolated `athenak-boundary-radiation` worktree at base74b7691e. It has **not passed perturbed vacuum stability** and is not in the TDE executable. No Aurora job was submitted or changed by this investigation. High-spin evolution remains gated; the independent Liu geometry provider has passed geometry tests but is not registered for evolution.

## Exact equilibrium and implementation checks

The opt-in `characteristic_bc_source=physical_constraint_radiation` computes outgoing radiation residuals of physical Theta and Z, then lifts them to incoming characteristic rates. Full and background evaluations use the same operations. All stencil inputs are immutable; only metric RHS neighbors are read, and the boundary never changes metric RHS. Disjoint face ownership prevents multiple writers. RHS ghosts are not read.

- Original defaults remain bitwise-identical in matched outputs to the base executable.
- Version1 preserves exactly zero residual through every recorded RK boundary/recast stage, including ghosts, with OpenMP1/2 and MPI1/4 on a fixed eight-block mesh. Nonzero pulse responses are bitwise repeatable across those partitions.
- Version2 preserves the same stage properties with OpenMP1/2 and MPI1/4, including bitwise nonzero pulse repeatability on the fixed eight-block mesh.
- Exact-zero version1 vacuum reaches20M. The version2 complete-map probe also preserves zero and bitwise timestep composition. These checks do not establish perturbation stability or new refinement-interface coverage.
- Fifteen compiled helper tests pass with poisoned RHS/state ghosts, including exact cancellation, an analytic Jacobian check, and manufactured constraints. A composed derivative consistency flaw was repaired: version1 used nested low-order derivatives with only first-order boundary consistency; version2 uses a fourth-order inner metric derivative and is second-order consistent for the composed operator.

## Actual evolution results

All cases use a lapse pulse of1e-8 unless stated otherwise. The sixth-order controls use RK3, dx=.25M and dt=.0375M. The original Z4c constraint damping remains kappa1=.1,kappa2=0; G=2; no sponge or clipping. See `pilot-summary.json` for exact events and manifests, and `pilot-comparison.png` for histories.

| Version / case | First invalid C2P metric | Active abort / outcome |
|---|---:|---:|
| v1 trumpet cubic |9.7125M|14.025M|
| v1 trumpet linear |22.35M|24M|
| v1 trumpet wider box, cubic |12.975M|17.025M|
| v2 trumpet cubic |13.725M|boundary validity abort after last progress18.75M|
| v2 trumpet linear, eight blocks |19.8375M|22.0125M|
| v2 trumpet linear, one block |19.4625M|21M|
| v2 trumpet quadratic, one block |16.575M|20.025M|
| v2 second-order trumpet, PLM/noFOFC |44.0625M|45M|
| v2 flat cubic |none|reached50M; maxTheta1.56e-10|
| v2 flat linear |none|reached50M; maxTheta1.74e-8 and growing|

The earlier second-order attempts with WENOZ or PLM+FOFC were rejected at input validation for insufficient ghost zones. They are retained as setup rejections, not counted as evolution failures.

## Locations and distinct mechanisms

At cycle0/stage1 in v1, the pre-boundary Theta RHS maximum is3.57347e-11 at(.625,.125,-.125)M, inside the horizon. The boundary update produces a larger3.45140e-10 maximum at(1.875,-.125,-.125)M, on the physical outer face. This is local truncation injection and boundary amplification of an already nonzero perturbation, not loss of exact zero equilibrium. Later invalid outer ghosts and final active-cell collapse are separate diagnostics.

The v2 complete-timestep analysis finds a strongly amplified direction: active norm gain1.46338 in0.3M, with97.779% of active squared norm and99.849% of Theta squared norm on face cells. Theta peaks at(-.125,-.125,1.875)M. Physical H/M/Q amplify by1.46471/1.45952/1.45710. The24-vector Ritz direction is not a converged eigenvector (6.45% active residual); the associated growth-rate estimate must not be called a converged eigenvalue. See `modes-v2/README.md`.

An independent frozen-coefficient Cartesian/Fourier-tangential matrix reproduces a fast oblique boundary mode with the actual trumpet face lapse, conformal factor and outward shift. Its rate1.2128/M is close to the complete-map amplification scale. Old zero-rate boundaries do not have this fast branch in the same model. Removing normal shift nearly removes it. At fixed grid tangential frequency, growth increases roughly as1/h. This is evidence for an additional compatibility defect in the **new boundary prototype**, not a claim that it explains the original campaign's slower residual mode. Model refinements and alternative closures are recorded in `fd-symbol/`.

Separately, linear ghost extrapolation is inconsistent with the unchanged sixth-order second derivative at the first active cell: on q=x^2 it gives-7/30 instead of2, independent of h. Quadratic restores consistency but did not cure the actual trumpet failure. Reducing timestep, block partitioning, derivative order, or extrapolation order alone is not an established cure.


A subsequent continuum half-space calculation validates a positive mode at1.2307116367/M for transverse k=2pi/M, using an ordered-Schur decaying subspace. The reconstructed20-field profile satisfies the bulk equations to7.84e-14 and the boundary conditions to1.66e-16. Without damping, its growth scales linearly with transverse frequency. Theta, H, M and Q are near1e-14, but electric/magnetic Weyl norms are.145/.0103: this is a constraint-satisfying mixed physical/gauge boundary mode, not a pure coordinate mode. Thus the implemented combination itself is unstable at the continuum boundary level; ghost discretization is not the only problem. The finite-grid counterpart also excites measurable constraints. This result concerns the new prototype, not the old slower campaign mode.

## Current next step and spin gate

Use the frozen oblique reproducer to test consistent coupled gauge/constraint/tensor boundary conditions before another expensive evolution. Direct Theta/A allocation alone reduces but retains the fast branch; it is not a fix. The published gauge conditions contain transverse terms absent from a zero-incoming-rate closure, and the2016 formulation also differs from the actual Z4c Gamma equation. Formulas must be translated rather than copied between formulations.

The standalone Liu-Etienne-Shapiro Kerr wormhole provider is prepared at a/M=.9. Its horizon diameter is.717944947M; dx=.0625M resolves it with11.49 cells. Geometry tests verify constraints, axis/throat regularity and sixth-order finite-difference convergence. Exact full-minus-background cancellation makes it a fixed point, as requested. With positive precollapsed lapse its unsubtracted continuum RHS is nonzero, however; the frozen-reference model and physical moving-puncture evolution must be distinguished. No spin timesteps, matter run, GPU or AMR test are claimed.

A separate nonspinning isotropic-wormhole vacuum control was then tested as a zero-background-shift discriminator. On the same coarse dx=.25M grid it first produced an invalid fourth corner ghost at67.3875M and aborted with active invalid state at77.025M. It uses the unchanged r+1 leading radiation weight rather than the exact wormhole areal radius, and resolves the horizon with only four cells across; this is a rejected discriminator, not a resolved wormhole or spin stability claim. See wormhole-control/failure-summary.json.
