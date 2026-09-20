# Peer review: lapse-adjusted damping prototype

**No blocking source defect found for the planned isolated residual-vacuum pilots.** Review covers the uncommitted three-file Z4c patch and inherited `main.cpp` mode hook in `/Users/hz0693/research/TDE/athenak-lapse-damping`, plus `test_sources.py` and completed `regression-v2` evidence. This is a source/operator review, not a stability result or approval to modify production.

The default-off option consistently replaces the damping product `alpha*kappa1_eff` by `kappa1_eff` in Khat, Theta, and Gamma. It preserves the current Gamma coefficient 2 and the Khat/Theta kappa2 factors; reconstructing K therefore retains the consistent damping `-3(1+kappa2)sigma*Theta`. No numerical inverse lapse is formed. The optional time-dependent kappa roll is retained through the same `kappa1_eff` parameter.

All three relevant paths were inspected: `BuildStandardPointwiseRHS`, direct full-minus-background residual assembly, and `ComputeResidualTerms`. Full and background use the same selected product. The geometric Hamiltonian, matter sources, lapse/shift equations, stencils, dissipation, projections, boundaries, and task dependencies are unchanged. On `Theta=Q=0` the added source vanishes; the change is lower order in the existing second-order spatial system. The disabled branch retains the original expressions/operation order; the runtime comparison below verifies this for the tested control.

The mode import hook is restricted to a serial, one-block, fixed-mesh, double-precision, pure-vacuum Schwarzschild trumpet. It imports the full residual allocation including ghosts, checks byte count/finite entries, and refreshes ADM through `Z4cToADM`, which reconstructs the full state. Stage-1 `CopyU` subsequently copies imported u0 into u1, so no stale RK companion is used. With the hook disabled, it changes no field array. The hook is a diagnostic facility, not a general restart or matter-data import path.

## Completed checks

- Corrected v2 regression passes. The original v1 script incorrectly called field index 1 Khat; the enum places Khat at **7**. The implementation uses named fields and was not affected. Original failed-test artifacts remain preserved.
- Three-stage zero residual, including saved ghosts, remains exactly zero.
- Mixed Theta/Gamma pulse source differences match the analytical change with maximum absolute error **8.94e-23**. Other first-stage RHS fields remain bitwise unchanged, and initial OFF/ON states match.
- The default-off binary matches the saved pre-prototype executable bitwise in pre-RHS state, volume RHS, and post-recast snapshots across all three RK stages.
- Initial pure-atmosphere matter RHS at zero residual is bitwise unchanged OFF/ON.
- Independent read-only postprocessing checks local Khat/Theta forensic sums and Khat/Theta/Gamma damping entries against saved snapshots within printed precision. Results are in `peer-review-forensics.json`. No evolution was rerun by the reviewer.
- `git diff --check` passes. The tested executable SHA256 reported by the regression is `5be3094897d5edd5106aa6cadc1faa5a5d25959d1a78ab2cc3774013686374c8`.

## Limits and nonblocking observations

Runtime tests exercise the direct analytic-background residual path with kappa1=0.1, kappa2=0, fixed kappa and unchanged initial lapse. The standard/full branch is source-audited but not independently executed by this script; nonzero kappa2, rolling kappa, a simultaneous finite lapse perturbation, GPU, MPI and refinement interfaces are outside this regression. Those limits do not block matched local residual pilots, but must remain explicit in any broader validation claim.

The forensic geometry recomputation already has tiny arithmetic discrepancies: for exactly zero state it can print Khat_alg around `-3.8e-19` and Theta_Ht around `-2.5e-24` while actual volume RHS is zero. New damping entries are exactly zero. This pre-existing diagnostic behavior is not introduced by the patch; do not describe all forensic terms as bitwise identical to evolution.

No new general source-timestep safeguard is added. The proposed coordinate products 0.1/M and 0.3/M are safely nonstiff at dt=0.075M (`2*sigma*dt=0.015,0.045` for kappa2=0). Larger rates would require a fresh timestep assessment. None of these checks demonstrates long-time bounded constraint growth or permits calling a finite target stop stable.
