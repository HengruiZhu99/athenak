# Residual transfer implementation checkpoint

`coherent_transfer=residual_shifted` is an opt-in legacy 55-field experiment.
The default is `none`. It presently rejects nonperiodic boundaries, adaptive
regridding and nonlegacy layouts. These are unfinished parts of Gate 1.

Each ordinary and post-projection transfer first completes the existing primary
restriction/exchange/prolongation. On this synchronized state, form E=G-R(u) in
all 33 auxiliary slots of a private scratch array. R uses centered FD2/4/6 on
active cells and a degree-order shifted polynomial derivative inside the
available halo at ghost cells. The stencil never reads a sixth primary ghost.
The table generator uses exact rational weights; the actual mesh oracle computes
its reference weights independently using long-double Lagrange products.

Exchange this scratch array through a separate boundary communicator. Its
coarse auxiliary slots mean restricted E, not G. Thus the reverse leaf transfer
is G_coarse_ghost=R_coarse(actual primaries)+R_E(E_fine), while fine ghosts use
R_fine(actual primaries)+P_E(E_coarse). Same-level exchange uses the same formula.
The coarse scratch is an intermediate interpolation buffer, not an evolved
coarse state. Existing coarse_u0 and all primary components remain untouched by
the correction. Every active auxiliary remains bitwise unchanged. The two
correction brackets are 11 (ordinary) and 12 (post-projection).

For the legacy lapse target, R is either the explicit collision expression
2(w D rho+rho D w) or the newer product target 2D(rho*w), as selected separately.
There is no independent interpolation of alpha. Metric Q still uses the six
legacy metric components. This is not yet the intrinsic 50-field chart transfer.
The historical GH/algebraic/auxiliary projections and their timing are unchanged.

The constant-residual fixture calls the existing centered projection, adds a
separate nonzero constant in every auxiliary component, then performs three
ordinary/repaired exchanges. It checks each face, edge, corner and ghost layer
against an independent derivative target and checks primary/active invariance.
Run `run_transfer_mesh.py --binary PATH --output NEW_DIRECTORY` and add `--smr`
for a single refined octant, or `--ranks 2` for the MPI build. Compile with
`PROBLEM=../../analysis/pc_gh_clean_reduction/transfer_mesh_oracle`.

Uniform 2D/3D FD2/4/6 and corrected 7/15-leaf refinement fixtures pass the frozen
2e-12 absolute bound. The two-rank MPI build also enables Kokkos bounds checking.
All these are constant-residual transfer tests, not variable-residual convergence,
curl preservation, coupled RK/KO stability or a physical evolution. No Gate 1
promotion follows from them alone.

The first SMR input used obsolete `<refinement1>` syntax and was rejected; this
fixture failure is preserved. The corrected input exposed the inherited 2D
high-order prolongation's unconditional 3D reads/writes (SIGBUS/SIGSEGV), and the
inherited FD4 refinement parser rejection. The dimensional fix collapses inactive
axes as constant extensions and writes only existing child cells, preserving 3D
point ordering/arithmetic. Dispatch now checks actual nghost=2 or 4 rather than
rejecting FD4 even with nghost=4. Historical 2D restriction remains averaging,
so it is still second order. The failure outputs precede the corrected results.

Remaining before promotion: spatially varying and finite residual injection,
complete curl/tangency/operation budgets, boundary parity/outflow, regrid/restart,
CUDA, coupled amplification/stability, stage-policy effects and full convergence.

## Physical boundary completion and retained failures

The opt-in legacy residual transfer now admits periodic, outflow and reflecting
faces on fixed meshes. It completes physical boundary values after ordinary
prolongation (operation 13), then transfers residuals with their tensor parity,
completes residual physical corners after prolongation, and reconstructs auxiliary
ghosts. Operation 13 may change ghost primaries; operations 11/12 preserve all
primaries and all active entries. No bulk equation or projection policy changed.
Reflection reconstruction folds coordinates to the interior mirror and applies
the auxiliary tensor parity. Centered target arithmetic matches the existing
projection; shifted targets use differences to reduce cancellation.

The first physical-boundary matrix failed because prolongation left physical
corners stale. Both pre-fix and post-fix results remain archived. The corrected
operator passes all FD2/4/6, 2D/3D default outflow and compatible mixed reflection/
outflow fixtures on uniform and refined meshes. The latter seeds all 33 residual
components with independently defined parity-compatible multiaffine polynomials.
Its serial refined maximum residual error is 2.24e-13 and parity error 3.47e-18.
Two-rank MPI with bounds checking passes the same compatible fixtures and default
outflow refinement. Periodic uniform/refined serial regression also passes.

Two negative controls remain FAIL at their frozen 2e-12 tolerance. Nonzero
constant odd residuals have discontinuous reflected extensions; their refined
interpolation errors cannot be interpreted as smooth exact-preservation errors.
The separate compatible fixture addresses that ambiguity without deleting the
failure. Cubic outflow extrapolation in 3D amplifies encoded residual roundoff
above tolerance (up to 6.16e-11); it is not qualified. Default extrapolation order
2 and order 3 passed the tested constant residual checks. No tolerance was raised.

The original CUDA transfer build failed: NVCC requires the member enclosing an
extended lambda to be public. The access-only correction is rebuilding in the
same owned remote directory, with original failure logs and source manifest
preserved. New controllers are build PID 605504 and test PID 625235; the latter
waits for success and checks GPU occupancy. This snapshot is 40e0bc1f plus the
recorded access patch, not this newer boundary implementation. GPU transfer tests
remain NOT_RUN until actual results are collected. Full Gate 1 and all physical
gates remain unpassed; dynamic regrid/restart, complete causal budgets and coupled
amplification still require work. The intrinsic 50-field kernel is not yet present.
