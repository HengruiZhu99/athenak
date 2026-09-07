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
