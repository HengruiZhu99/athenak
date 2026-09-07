# Clean reduction implementation and qualification plan

The supplied goal is the scope authority, preserved verbatim in the evidence
directory. Ordered promotion gates are 0 provenance/legacy equivalence,
1 operators/interfaces, 2 smooth forced-constraint dynamics, 3 fixed-layout
single puncture through 20M with spatial/temporal convergence, 4 original M/256
stress through 6M at two timesteps, 5 moving puncture then matched head-on binary.
No expensive evolution precedes the mathematical and compiled oracle checks.
Failed gates prevent promotion, not affordable diagnosis or corrective work.

Keep A (collision legacy) and B (coherent transfer only) matched in equations,
gauge, damping and reset policy. C is the explicit clean 50-field system with
free GH and reductions. D adds only a separately qualified stagewise auxiliary
core. No reset of C/Z belongs to D. Default proposed C physical parameters are
M gamma_R=1, M kappa=1, M eta=2, lambda=alpha gamma_R; constant lambda is a
separate algebra/control option. The switch is the pinned C-infinity plateau
at z=(0.1,0.5), and its gauge coefficient is one.

Intrinsic mode and restart ABI must explicitly distinguish 50 fields from the
legacy 55. The map uses s=(a,c,b,d,e), five trace-free Ahat coordinates, and
l=d(alpha), half the old L. True primary derivatives must be used throughout
the complete source differentiation. Never replace d(s), d(beta), or d(alpha)
with independent auxiliaries in those derivatives. Keep the advective curl term.

Before each new fixture runs, append its exact operators, halo-valid regions,
orders, normalized tolerances, health criteria, resources and commands to PLAN.md
and gates.json in the evidence directory. Freeze thresholds before examining
results. Do not label FD6+q5 interfaces sixth order. Derivative ghost reconstruction
requires six-cell primary reach for FD6, beyond the existing four ghost layers.
No unsupported nghost increment is an implementation.

Diagnostics must preserve full and masked coordinate-volume norms, independent
physical Ricci/Hamiltonian/Codazzi constraints, signed same-cell operation
increments and ghost validity, separate intrinsic/raw curls and reductions,
semidiscrete tangency defect including KO and RK effects, domain margins and
metric conditioning. Preserve the first invalid stage before projection.

Existing common-symmetrizer and puncture-limit obstructions are retained limits,
not gates to redesign the gauge or add cleaning fields. A passed candidate
suite alone does not prove compiled correctness or discrete stability.

## Legacy compiled equivalence, frozen before execution
Seven smooth nonlinear periodic off-constraint seeds, all 55 rows and active cells; FD2/4/6, 2D/3D, anisotropic lengths 1,1.3,1.7. Compare source b81b44d6 and current source using an identical isolated test adapter. Use the explicit collision_factorized L projection and unchanged legacy gauge switch and equations. Require normalized component error <=2e-12 (normalization in gates.json), and compare one actual RK3 step. The direct_product default is a negative projection control, not an unchanged baseline. No Einstein-solution claim from these arbitrary data. CPU zero-step controls first; CUDA/one-step controls follow.

## CUDA legacy controls resource reservation
Build independent source snapshots on della-vis1 with four CPU build jobs, sequential builds. Test one 8^3 or 8^2 block at a time, one A100, estimated <1GiB GPU memory, maximum one RK3 step per fixture. Maximum 600 seconds per launched test group. Occupancy sampled 3496MiB/40960MiB, 0% utilization; recheck immediately before execution. No performance claim and no unrelated process changes. Serial equivalence first; MPI/transfer/restart controls remain separate.

## Signed state-budget writer oracle (before execution)
Test every field and ghost cell on 2D and 3D one-block fixtures against independently indexed signed increments, tolerance 2e-15 absolute; require stored same-cell subtraction exactly, and an exactly zero no-op increment. Test truncated payload rejection. Explicitly label both ghost-validity flags unasserted. Output is chronological native-endian binary with an endian marker and explicit active ranges; no memory-layout assumption. A state increment does not by itself supply a reduction/curl causal budget.

## Actual mesh residual-transfer fixture (frozen before execution)
Use the new residual_shifted option on CPU Serial, periodic root 16 per active
axis, blocks 8, four ghosts, FD2/4/6. Seed nonconstant positive primaries and
independent constant residuals in all 33 legacy auxiliary components using the
existing centered projection kernel. Compare ordinary and repaired ghosts using
independent long-double Lagrange-product derivative weights. Require exactly
unchanged primaries and active auxiliaries, and absolute constant-residual error
<=2e-12 in every ghost component (faces, edges, corners, all layers). Repeat three
ordinary/repaired exchanges. First uniform 2D/3D, then one-octant static refinement;
record failures separately, not average across blocks. Each CPU fixture has a
120-second limit. These are operator checks, not physical evolution. No CUDA/MPI
or general boundary/regridding claim from this fixture. The historical 2D
restriction is retained and remains second order.
The same frozen component thresholds also apply to a two-rank CPU MPI build
with Kokkos bounds checks enabled. Retain the original parser/2D crash results
and rerun corrected fixtures in fresh directories. MPI runs remain tiny operator
checks on the local CPU, with the same 120-second per-process bound.

## Task-path integration smoke check (before execution)
One 2D periodic block, FD6/RK3, one step of the existing smooth arbitrary legacy
seed, coherent transfer on and collision lapse target. Enable raw state budgets.
Require a finite successful step, ordinary correction records at initialization
and stages 1--3, exactly one post-projection correction at the final stage, and
bitwise unchanged primary/active entries within each correction bracket. This
checks task integration only; the arbitrary seed is not a physical solution and
one step cannot establish stability or convergence.

## CUDA transfer build/resource plan
Build an isolated snapshot of production commit 40e0bc1f with CUDA AMPERE80,
MPI enabled and Kokkos bounds checks, using four CPU compilation jobs on
Della-vis1. No GPU evolution in the build controller. Next operator tests use
one A100, <=15 blocks and estimated <1 GiB GPU memory per rank, first one rank,
then two ranks sharing the authorized GPU if the single-rank checks pass.
Recheck free memory immediately before those tests. Last occupancy before build:
3559/40960 MiB, 0% utilization. Do not alter unrelated processes. Use the same
120-second process bound and frozen 2e-12 component tolerance; no long evolution.
