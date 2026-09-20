# Vacuum mode isolation, then Liu Kerr follow-up

The user authorized these tests on2026-09-19, with the spinning Kerr follow-up conditional on passing vacuum. All runs here are independent diagnostics. Do not change, cancel or restart the TDE campaign; debug-scaling remains reserved for it.

## Completed tests and remaining limitation

- `mode-analysis/`: complete discrete timestep linearization in an isolated worktree, including RK, ghost updates and algebraic projection. A restart alone is not assumed to be the exact timestep map, because initialization and timestep-end ghost/projection order differ. Check composition against uninterrupted evolution before interpreting eigenvalues.
- `discrete-symbol/`: frozen-coefficient principal symbol using actual sixth-order stencils and a compatible tensor-derivative comparison. These checks cannot establish global variable-coefficient or boundary stability.
- `evolution/`: GPU job8840650 finished in debug/MHDTidal. Original and wider controls aborted at672M and476.025M; the refined control stopped cleanly at333.3375M on its application walltime, with all eight checkpoints valid but growing constraints. The independent CPU half-timestep comparison stopped cleanly at244.275M and reproduced the baseline growth. All binary and restart files are per-rank. No diagnostic job remains active.
- `covariant-sources/`: coupled-source sigma1 control reached1000M with valid active/ghost metrics, but its exterior norm grew2.924 times over750–1000M. An independent complete-map test verifies a weak oscillatory growing physical constraint mode, gamma about.0018/M. Exact equilibrium cancellation is not perturbation stability. These controls have matter feedback disabled; passive fluid behavior is not atmosphere validation.
- `constraint-lower-order/`: independently validated spherical constraint and full-state volume operators identify sensitivity to physical constraint radiation at the boundary. The reduced constraint spectrum can decay, but the full-state boundary discretization is not yet stable. No boundary cure has been promoted to AthenaK.
- `liu-plan/`: primary-paper geometry checks and implementation plan only. No spinning evolution is authorized before the vacuum gate below passes.

## Vacuum acceptance gate

An exact zero equilibrium by itself is insufficient. Before proceeding to spin, require:

1. Exact residual zero through all stages, boundary/projection operations and a long1000M vacuum control, including MPI and the refinement transfers used by the candidate configuration.
2. Controlled nonzero perturbations through1000M with finite active and ghost metrics, no recovery or invalid-state errors, and complete matching checkpoints. A walltime stop is incomplete.
3. Successive post-transient peak amplitudes and fitted windows must support bounded/decaying constraint and gauge perturbations; simply remaining finite or briefly plateauing is not a pass. Compare at least two perturbation amplitudes and resolutions to distinguish linear growth, numerical noise and nonlinear saturation.
4. Check timestep, domain and boundary dependence, and repeat the successful candidate on MPI/GPU and with refinement. Do not promote a small-box or frozen-symbol pass to a stellar stability claim.

A failed test stops the spin progression, not the diagnosis. Never threshold/reset tiny residuals, clip constraints or suppress all physical evolution to manufacture a pass.

## Liu et al. caveat

The paper1001.4077 constructs two-ended Kerr puncture initial data, not the positive-lapse stationary trumpet used by the current residual scheme. The geometry, extrinsic curvature and gauge must be kept consistent. A fixed analytic reference with a gauge that has a nonzero continuum geometric RHS must not be advertised as cancellation of truncation error only. See the independent geometry/stationarity audit before implementing this extension.
