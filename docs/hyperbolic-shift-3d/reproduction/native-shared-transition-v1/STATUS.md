# Active convergence investigation

Branch: `codex/native-initializer-shared-transition`, based on preserved SG1 commit3681f42ac. The active full objective is [goal-specification.txt](goal-specification.txt); it adds the confirmed dissipation-closure defect to the earlier user request. Prior turn produced the completed three-resolution histories (progress); this turn starts implementation. All SG1 grids reached10M; N32 checkpoint verification passed remotely, but large checkpoint downloads/common-cutoff radial reconstruction remain pending. No new HPC job has been launched.

## Evidence and remaining work

1. Shared restoration: implemented as an opt-in separate target, original SG1 targets unchanged. `OuterHyperbolicGauge` removes C only from the restoring pole; `HyperbolicGauge` applies W once. Independent nonradial oracle checks the requested equation, exact inner plateau and identical near-scri sources; CPU/device and partition parity passed. Separate ST1 checkpoint magic prevents SG1 restarts. Auxiliary D evolution and mu unchanged. A native production initializer is still required before the controlled evolutions.
2. Native initializer: not implemented. Port the isotropic wormhole construction from `tst/elliptic_shift_1d/wormhole_isotropic.py`; integrate initial beta deviation directly, with no fixed table. Actual volume/scri initialization and tolerance tests remain.
3. Initial-data convergence: not done. Must separate continuum, FD, exchange/projection/scri initialization and norm/cutoff effects, using N16/N24/N32 and finer bounded tests if necessary. Shift is absent from H/M constraints.
4. Dissipation consistency: not yet fixed or verified on actual patches. Existing B4-transpose-weight-B4 block truncation is third order at an edge. An energy-negative B5 version is a candidate, but its mapped-grid consistency/stability and CPU/device parity must be measured before adopting it.
5. Scri operator consistency: not yet established. Prior null closure and device parity do not satisfy this requirement. Prioritize sampled normal derivatives of numerically differentiated pole numerators, then penalties and interior pole/Omega, with nontrivial regular manufactured data and cross-face measurements.
6. Controlled native-initialization N16 A/B to10M: not started; consistency gates precede long runs. No new long N24/N32 authorized.

## Why restoration is retained

Removing the redundant collar is distinct from removing restoration. For a regular near-scri deviation beta+x = Omega*b1 + O(Omega²), the existing source tends to -4*b1. W=C=1 in a neighborhood of scri, so the shared transition preserves this finite limit and its normal derivatives exactly. Removing restoration would change both tangential shift evolution and the dependent normal D: with `a = a_base - 4*b1` and `D_normal = (normal_rate - a_normal)/mu`, the zero-restoration closure would need `D_normal_new = D_normal_old - 4*b1_normal/mu`. No evidence presently establishes that this different boundary system preserves accuracy/stability; restoration remains. This is a limiting-equation comparison, not proof that restoration is mathematically indispensable.

## Local validation

Fresh build directory `build-native-shared-transition` preserves earlier local builds. Configure with MPI and `Athena_ENABLE_HYPERBOLIC_SHIFT_3D_TESTS=ON`. Built the shared-transition oracle, shared CPU/device parity and original SG1 oracle. All three passed; logs in `checks/shared-transition-tests.log`. These tests establish the source change and parity only, not evolution stability or truncation order. Production shared-transition target and checkpoint diagnostic are registered separately; not yet run.
