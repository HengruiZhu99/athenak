# Thin core overlap with paired trace penalties (experimental V12)

Base: `codex/hyperbolic-core-shell` at
`e48a8c43f56d079225c12ec2e2901e5e461e3f1c`.
Experiment: `codex/hyperbolic-thin-overlap`. This is an additive research
variant, not a replacement for the admitted V11 controls. The original task
continues sparse communication validation independently; its inherited dirty
snapshot is preserved locally at `38a793225` on
`codex/hyperbolic-sparse-stage`, not treated as validated source.

## Geometry and centering

The volume remains cell centered with common Cartesian tensor components.
The cube has nominal half width .18, 2 blocks per axis and n cells/block.
Pure spherical shells begin at .16, .45, .75, ending at scri=1.
Core sampling is unchanged: even 2n samples across the origin, no r=0 vertex.
Only the extra evolved Cartesian extension is varied (1, 2, or 3 cells).
Shell angular extensions and the first shell's outer radial extension remain
three cells. Ordinary FD ghost width stays three. The negative-square KO
operator retains its existing active-donor reach of four.

There are TWO artificial interfaces: the outermost evolved Cartesian layer
and the first evolved spherical radial layer. Each supplies one trace layer.
At each trace location BOTH independent solutions are sampled. These are
interpolated coincident evaluations, not topologically shared vertices. No
Cartesian cube is claimed to coincide with a sphere. The geometric overlap
remains: on the nominal axis .16 < r < .18, and a much larger overlap toward
cube corners. Neither masking nor cube deformation is introduced.

The trace lift acts on the full six-point interpolation support. Thus 'one
layer' describes the trace locations and extra Cartesian evolution; it does
not restrict the donor or penalty support to one point. No claim of one-layer
angular shell seams is made. Nodal interface traces are the selective reuse
of a nodal viewpoint, not a conversion of the whole evolution to native VC.

A centered six-point donor is unavailable for some thin cases. V12 explicitly
allows a second pass that shifts all six donor points within the evolved
sample hull. It tries this only after the unchanged centered pass fails.
Outside-hull extrapolation remains forbidden. The nearest three-layer
zero-penalty/shifted-off experiment recovers V11. The shifted donor policy
is a changed discretization and must be compared separately.

## Penalty and the precise energy claim

Let q contain log(chi), log(alpha/A_ref), and the remaining native deviations
in the common Cartesian frame. The Minkowski reference is exactly q=0.
For each interface trace use six-point tensor-product interpolation T_C,T_S
at exactly the same physical x. N_C,N_S are derivatives of those same
polynomials along a common physical normal. The shell normal is +r, the
Cartesian normal is the outward face normal; the two domain traces use the
SAME normal at a given point, not opposing normals.

Define L0=[T_C,-T_S], L1=h_C[N_C,-N_S], W=diag(V_i)>0 over all independent
volume copies, and A=diag(surface_quadrature*h_C)>0. W deliberately is NOT
the diagnostic partition of unity, whose weights vanish on some copies.
For one field the extra chart rate is

    Pq = -kappa W^-1 (L0^T A L0 + eta L1^T A L1) q.

The actual transpose, not a separately interpolated reverse transfer, is
used for the paired lift. Therefore, in E=1/2 q^T W q,

    dE/dt|penalty = -kappa ||L0 q||_A^2 - kappa eta ||L1 q||_A^2 <= 0.

Both penalties vanish on their discrete agreement subspace. Agreement of
smooth continuum data leaves interpolation truncation error; it is not
exact agreement of two different interpolants. Constant reproduction,
oblique smooth data and refinement are measured independently.

Value mismatch is penalized for all 31 variables. The derivative term acts
on chi, six metric components, lapse and three shifts (11 fields), whose
second derivatives enter the system. Other fields use eta=0. Default eta=.25,
kappa is explicit and defaults to ZERO. This is a component norm in the
transformed variables, not a proven hyperbolic symmetrizer or physical
constraint energy. It supplies no proof of a stable second-order boundary
closure. Characteristic flux/SBP boundary terms are NOT implemented.

The physical added rates are chi*Pq for chi and alpha*Pq for lapse, and Pq
otherwise. Existing log RK conversion, counted physical lapse floor 1e-10,
reference functions, accepted algebraic projection and scri closure remain.
The proof is for the additive semidiscrete chart term alone; algebraic
projection and the full bulk/ghost/time integrator do not inherit it.

A conservative infinity-norm bound of the symmetric matrix
W^-1/2 L^T A L W^-1/2 is precomputed. The isolated RK4 condition
`dt*kappa*(upper_value + eta*upper_normal) <= 2.7` is enforced.
It does not replace the bulk CFL or establish stability of their sum.

## Cached implementation and diagnostic ownership

All trace locations, interpolation/normal coefficients, transpose-lift
indices, cell volumes and reference lapse samples are built once. Device
execution uses batched chart conversion, trace residual and deterministic
per-target transposed lift. No atomics enter evolution sums. A rank retains
only rows whose lifts contribute to its owned points; dependencies are
exposed for integration with the sparse stage exchange. There is no
per-cell kernel launch. The initial V12 comparison uses the preserved full gather control. The
separate sparse target adds trace dependencies to the union and has passed
local two-rank poisoned-buffer equivalence. Short twelve-tile L1/kappa1
oblique and puncture communication/restart comparisons now pass exactly;
longer stability and accuracy qualification remain open (see README).

Trace diagnostic rows belong to the rank owning the first donor of that
row, independent of the number of ranks storing it. Norms include both
artificial interfaces separately and normalize value/normal sums by 31/11
fields and trace weight. Normal mismatch is h-scaled. Failed horizon
acceptance remains unavailable for exterior constraints/mass; no tolerances
were relaxed. A distinct V12 checkpoint appends 32 bytes binding layer count,
shifted-donor policy, kappa and eta; restart rejects mismatches.

## Infrastructure audit

Remote heads independently checked on 2026-10-29:

- VC performance: `6bfa5c11c3b10775294f5f2d95a196d84d6718cb`.
- Sparse synchronization: `62993e7bac8fbaed13f592834282ca09142a5c2d`.
- Original VC: `7e0690e46766cdb101f45aceefe8cc5f2a66ae4f`.
- Hardening: `34cd6db227e81aed91064f3684733e720c9fac7e`.

The donor topology plan builds authority/participant lists once per topology
generation, retains device pack/unpack buffers, and hard-averages truly
coincident vertices deterministically. Its exact endpoint/ghost index ranges
and midpoint AMR rules are tied to structured VC identities. Copying that
averaging to these two independent traces would erase the discrepancy before
measurement. The mapped-shell research driver is not a MeshBlockPack VC
client, so its Cartesian neighbor class is not directly usable unchanged.
Reuse here is the cached sparse-plan/explicit ownership pattern; gauge, RHS,
subcycling and puncture treatment are not imported. Actual sparse exchange
integration is now incorporated as a separately testable V12 target, with
the original performance task's controls retained.

The donor's `docs/investigations/z4c_vc_amr_correctness_performance_20260822/REPORT.md`
explicitly records rejected r=0 puncture initialization (divergent analytic
ADM carrier), with a regular true-vertex black-hole fixture still open.
Keeping the present puncture sampling avoids inventing that analytic limit.
Current continuous-core layout still requires a uniform Cartesian level.
This experiment does not establish dynamic AMR or prolongation support.

## Literature and limits of transfer

[Kopriva–Nordström–Gassner (2021)](https://arxiv.org/abs/2105.04664)
derive overlap-weighted energy estimates and conservation under specified
coupling-matrix conditions for linear hyperbolic systems. Their paired
interior penalties motivate using matched weighted reactions here; their
conditions are not a theorem for our nonlinear second-spatial-order system.

[Kopriva–Winters–Nordström (2024)](https://arxiv.org/abs/2405.04668)
show that a well-posed overset continuum formulation plus a stable individual
DG method need not give a uniformly stable coupled approximation. Their
additional disagreement penalties motivate measuring both spectral and
finite-time behavior, rather than inferring stability from the energy sign
of one added term.

[Sharan–Pantano–Bodony (2018)](https://doi.org/10.1016/j.jcp.2018.01.049)
use actual diagonal-norm SBP derivative operators and SAT interface closures,
with incoming-characteristic interpolation. Their one-dimensional stability
proof and multidimensional experiments depend on those operators and norms.
Our centered mapped differences, ghost filling and transposed mismatch
penalty are honestly described as experimental ghost-plus-penalty, not
SBP–SAT. A later characteristic closure must explicitly address the
second-derivative boundary terms before borrowing that stability claim.
