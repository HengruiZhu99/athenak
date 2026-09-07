# Initial intrinsic mesh integration

`pc_gh/formulation=intrinsic_clean` now allocates and evolves the actual 50-field
layout through AthenaK's mesh, task queue, same-level boundary communication,
RK3 registers and restart I/O. No legacy tensor slices are bound to this state.
The compiled `FiniteDifferenceRHS` consumes synchronized same-stage stencil
values. Both algebraic and auxiliary resets are absent; requesting either GH
or reduction projection is rejected. Legacy equations and bindings are unchanged.

The mode currently accepts only uniform periodic meshes and RK3. Initial data
must be `intrinsic_minkowski` or `intrinsic_smooth`; the latter is a small periodic
trigonometric perturbation in all 50 fields, including independent auxiliary,
curvature and GH errors. It is an integration fixture, not an Einstein solution.
The constructor rejects unsupported PC-GH settings, nonperiodic/refined meshes,
legacy history/constraint output and unrelated initial data. Complete state
output uses 50 distinct `pcghi_*` names and `variable=pcgh`. Restarts carry
`intrinsic_pcgh50`, version 1, fields 50; no 55-field restart conversion is implied.
The generic state binary output remains float32; oracle checks read float64
restart payloads instead.

The mesh wrapper implements the prescribed lapse-scaled default
`lambda=alpha*gamma_R`, with `reduction_profile=lapse_scaled` and
`reduction_rate=gamma_R=1`. `reduction_profile=constant` explicitly selects the
constant-lambda control. Default eta=2, kappa=1 and KO=0.3 are recorded in the
input. The RHS equations and the fixed C-infinity gauge switch are unchanged.
The timestep candidate includes all characteristic speed families, additive
coordinate-direction frequencies, KO and relaxation/damping rates. It is a
conservative scalar estimate, not a proof of coupled nonlinear stability;
these oracle runs independently enforce dt=1e-4.

Health validation checks all state/stencil cells for finiteness, positive w/rho,
0<alpha<2, alpha^2*w^2<4, finite geometry/inverse, and a positive numerical minimum
metric eigenvalue. Each validation writes rank-local active-cell bounds for w,
rho, alpha, domain margins, minimum metric eigenvalue, Frobenius condition number
and all 50 absolute component maxima. The Frobenius condition of identity is 3,
not 1. The first bad cell retains surrounding stencil values and RHS values,
with RHS ghost validity explicitly limited to active cells. These initial host
mirror checks prioritize auditability, not throughput. They do not yet supply
all required volume norms, signed operation budgets, RK stage numbers, independent
physical H/M, reduction/curl fields, or puncture-layer diagnostics. Legacy
constraint output is rejected rather than filled with misleading zeros.

## Executed controls

Evidence: `qualification-runs-20260907/pcgh-clean-reduction/intrinsic-mesh-001`.
Raw arrays, inputs, restart files, binaries and source snapshots are external,
with hashes in the manifest.

- Independent Python candidate RHS, separately assembled centered FD/KO and
  RK3: FD2/4/6 x 2D/3D x constant/lapse-scaled damping, all 12 cases PASS.
  All 50 final fields, including periodic ghost faces/edges/corners, agree within
  1.111e-16 normalized error, against a frozen 2e-12 tolerance.
- Initial recorded metric eigenvalue and Frobenius condition bounds match NumPy
  eigvalsh/inverse to 4.441e-16 absolute error.
- Actual two-step versus one-step + intrinsic restart: bitwise equal in 2D/3D
  for both rate laws, including all ghosts. Two-step exact Minkowski unchanged.
- Thirteen controls PASS for unsupported paths, invalid-state/stencil retention,
  intrinsic-to-legacy restart rejection and the actual 50-name state output header.
- Legacy: six FD2/4/6 2D/3D one-step comparisons remain bitwise equal to the
  collision-source executable; all 19 legacy restart controls remain PASS.

An initial compile failed by attempting to call a private ParameterInput helper;
it was corrected to inspect the public input-block list. Its log is preserved.
Early constant-rate runs preceded the explicit damping-profile selector and health
recording; final runs use the retained `intrinsic-mesh-final-source-002/athena`.
A setup attempt used the system Python without NumPy; it did not execute an
intrinsic case. The final tests use the recorded virtual environment.

## Reproduction and remaining scope

Build normally with the existing configuration (the intrinsic built-in routing
also works when the qualification legacy user pgen is configured). Run:

```
python analysis/pc_gh_clean_reduction/check_intrinsic_mesh.py \
  --binary /absolute/path/athena --reference /absolute/path/candidate.py \
  --rate-profile lapse_scaled --output /new/output/directory
```

Repeat with `--rate-profile constant`. The controls script takes the generated
FD6 2D input and restart as `--input` and `--restart`.

These are real mesh integration checks, not Gate 2 physical/convergence tests.
CPU serial single-block periodic communication is covered. Multiple blocks,
MPI/CUDA execution of the integrated task graph, coherent intrinsic refinement,
physical boundaries, physical/reduction/curl diagnostics and all black-hole
qualification gates remain open. Next is multiple-block/backend verification and
independent intrinsic diagnostics before any physical promotion.
