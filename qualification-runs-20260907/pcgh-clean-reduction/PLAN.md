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

## Varying-residual transfer/curl ladder (frozen before execution)
Use the same periodic topology (4/8 uniform leaves; 7/15 static refined leaves),
but double cells per block 8,16,32 at fixed block boundaries and physical domain.
FD6, four ghosts, three exchanges, no evolution. Seed each legacy auxiliary with
E_n=0.001*(n-I_P1+1)*sin(2*pi*(x+y/1.3+z/1.7)+0.13*n), with z omitted in 2D.
The primaries remain the previous nonconstant fixture. Compare residual ghosts
against their analytic E and curls on all active cells against analytic dE.
The factorized legacy lapse curl includes its discrete product-rule error and
must remain separately identifiable as family 7; do not silently subtract it.
Require exact primary/active invariance. Uniform ghost residual errors must be
<=2e-12; uniform curl errors should converge at FD6 order (minimum rate 5.5).
For static interfaces expect second-order residual interpolation in 2D and
fifth-order in 3D; minimum successive max-error rates 1.7 and 4.5, respectively.
One derivative can lower interface max curl order: minimum rates 0.8 (2D) and
3.5 (3D), unless the component reaches the 2e-12 roundoff floor. Compute rates
for each repeated exchange and retain every component/failure. Ghost norms count
overlapping block halos and are not physical-volume constraint norms. Active
curl coordinate-volume norms cover every leaf without excision. At the largest
3D mesh, estimated working memory <8 GiB; each fixture remains bounded to 120s.

## Full ghost-consumer TT/KO falsification (before execution)
The earlier toy added a global KO matrix but omitted reconstruction at KO's
auxiliary ghost consumers. For the same-level two-block periodic TT subsystem,
include K_D*h in v_t and K_KO*h in q_t, obtained from the actual shifted ghost
target minus the owner's centered target at every consumed ghost. E=q-Dh then
has E_t=-lambda*E+KO*E+K_KO*h. This is a semidiscrete tangency defect, not the
candidate's continuum source mixing. Test blocks 8,16,32,64; FD2/4/6; KO 0 and
0.3; lambda=1; SSPRK3 dt=0.2*h. Retain the full-width control. Flag normalized
positive generator rates >1e-10 or RK eigenvalue modulus >1+1e-9. Near-neutral
floating flags require independent resolution, as before. This is a necessary
TT subsystem test, not a full Einstein/AMR stability proof. A failed result
blocks downstream promotion but not correction of the transfer/KO construction.

## Physical-boundary residual transfer (frozen before execution)
Add outflow extrapolation of E and reflection via the mirrored-primary derivative
with full auxiliary tensor parity. Periodic transfer is unchanged. Preserve all
primaries and active auxiliaries exactly. Test FD2/4/6, 2D/3D, uniform and the
same 7/15-leaf refinement maps, with all-outflow, all-reflecting, and mixed faces.
Use the constant residual fixture and independent index-count parity plus
long-double derivative oracle. Require max ghost residual error <=2e-12 and
reflection parity error <=2e-12 for every state component, including corners;
repeat three exchanges. Constant odd-parity residuals are a discrete boundary
fixture, not smooth physical data: do not claim physical curl convergence from
this test. Start extrap_order=2, then independently exercise 3/4 for outflow.
The parser must reject user/inflow/unsupported boundaries, adaptive topology,
and nonlegacy layouts for this option. Existing one-step periodic controls must
still pass. CPU/MPI operator runs retain 120-second per-process limits.

## Compatible reflecting residual fixture (before execution)
The nonzero constant odd-parity residual has a jump at a reflecting plane.
Near an intersecting refinement interface, polynomial interpolation of that
odd extension does not reproduce a constant; retain this as finite injection,
not an exact-preservation success. Add a separate compatible multi-affine seed
for mixed faces: for each odd tensor axis multiply E_n by (x-x_ref)/L, where
x_ref is that axis's reflecting plane; even axes contribute one. Each axis has
at most one reflecting face, so this polynomial obeys reflection exactly and
linear outflow extrapolation. Require the unchanged 2e-12 absolute residual
and parity limits, all fields and ghosts, FD2/4/6 and uniform/refined 2D/3D.
This does not erase the failed discontinuous constant-residual fixture.

### Boundary one-step task bracket regression (frozen before run)

Use the existing arbitrary smooth legacy one-block adapter, FD6/RK3, dt<=1e-4,
default extrapolation order 2, 2D/3D outflow and mixed reflecting/outflow faces.
Require every recorded value finite, exact same-cell after-before equality,
operation 13 at stages [0,1,2,3,3], operations 11 at [0,1,2,3], and operation 12
at [3]. Require zero active increments for all three operations and zero primary
increments for operations 11/12. Operation 13 may change ghost primaries. This
is a timing/invariance regression on non-solution data, not a smooth physical
boundary or stability test. Run a periodic no-transfer one-step legacy control
against the collision binary with the already frozen equivalence tolerance.

### Intrinsic geometry and finite-radius map (frozen before compiled test)

Use independent symbolic metric entries and their first/second chart derivatives,
random non-diagonal determinant-one metrics, trace-free curvature and independent
chart gradients. Check all compiled geometry/J/H entries and all 50/55 map entries
at scale-aware absolute error |a-b|/(1+|b|)<=2e-12 for charts in [-0.7,0.7],
100 seeded states. Include independent Cholesky-based reverse states, nonzero
GH fields and reduction errors. Require determinant, inverse, curvature and Q
trace identities <=2e-12 after scaling. Reject explicit non-SPD, non-unit-det,
traceful A/Q and nonpositive w/rho states; rejected conversion leaves output
unchanged. Conversion is not restart integration, and no evolution mode is added.

### Complete intrinsic point kernel (frozen before compiled comparison)

Compare all 50 RHS rows, ten complete configuration sources, lower-triangular
chart rate, and their three true directional derivatives to the pinned nonlinear
Python oracle with its identical C-infinity switch. Use 24 seeded non-diagonal
off-reduction states with nonzero K/A/C/Z/B, independent jets, switch z values
0.05,0.1,0.100001,0.2,0.3,0.499999,0.5,0.8, and multiple positive alpha/w.
Use normalized |compiled-reference|/(1+|reference|)<=2e-11 for all outputs.
Also compare full 50x50 derivative matrices for eight oblique normals by applying
all 50 basis jets and subtracting the zero-jet source, at the same tolerance.
Record individual row/column discrepancies; no eigenspectrum substitutes for
entrywise comparison. Parameters lambda, eta, kappa vary independently. This is
a kernel oracle, not grid evolution, hyperbolicity proof, or gate promotion.

### Compiled physical GH oracle (frozen before run)

Use the pinned independent physical-metric Ricci/Hessian/Codazzi oracle on its
128 seeded smooth finite-radius reduction-satisfying jets with nonzero C/Z.
Compare compiled K,C,Ahat[5],Z[3] after subtracting primary advection at normalized
error |a-b|/(1+|a|+|b|)<=2e-12, with matched kappa=1, eta=2, lambda=1.
This checks ten physical primary rows; configuration gauge and auxiliary rows
remain covered by the complete off-reduction point-jet oracle, not this test.
