# Clean intrinsic reduction implementation status

2026-09-07. Work in progress. Only one-step legacy controls have evolved; no new-formulation qualification claim.

The remote source was fetched at `62945657b4f2828a481abb5e7708e0e3e06dbd8d`,
exactly the independently verified SHA. The requested new branch
`codex/pcgh-clean-reduction-transfer-20260907` was created there, then fast-forwarded
to the originating checkout's committed descendant `249929fd` after inspecting
the intervening commit inventory and production diff. This retains the direct
lapse-gradient helper, tracker and CUDA fixes, and negative qualification evidence.
The originating checkout has dirty analysis and extensive untracked raw evidence;
none of those files was modified or merged. The parent main checkout is untouched.

Separate detached legacy control: `../athenak-pcgh-legacy-control-20260907`,
commit `b81b44d658f3b81584e94ce79b92656c112ff908`. It is an evidence-only checkout.
The mathematical reference is detached at
`7ef9c61c0c2bd12a46b28f334a53a1aabcd1842e` in
`../pcgh-candidate-reference-20260907`; its files are not changed by tests.
The candidate suite executes in `../pcgh-clean-reduction-tests-20260907/candidate-suite`
using a separate virtual environment. All eight original checks returned zero.
This is reproduction of candidate algebra/point jets, not independent compiled
AthenaK verification, a puncture theorem, or an evolution result.

The direct-lapse report is recovered at
`qualification-runs-20260905/direct-lapse-gradient/REPORT.md`. Preserve its failure
at 5.167923M versus 5.187818M, unavailable first-invalid intermediate state,
and scalar-max/stale-ghost limitations. The R16 20M ladder has anti-aligned field
differences and does not pass the binary gate. Existing transfer brackets do not
establish a causal projection-feedback explanation of late growth.

Code inspection confirms GH projection every stage, reduction projection only
on the final stage, followed by restriction/exchange/BC/prolongation. Both ordinary
and postprojection transfers require explicit coverage in the new operator work.
The inherited direct lapse helper returns **twice** the gradient: intrinsic l
must account for that convention and cannot reinterpret legacy L.

Della access works using the preserved multiplexed ControlPath. Initial occupancy:
A100-PCIE-40GB, 40960 MiB total, 3496 MiB occupied, 0% sampled utilization; no
user Slurm jobs. This does not imply the occupied memory is available. Scheduler
gputest reports a 15-day limit and heterogeneous GPU node types. Resource use is recorded separately for the completed small CUDA controls. Inspect
occupancy and exact requested node resources again before each new allocation. User authorization permits direct vis1 testing or up to
eight 80GB A100s through gputest; request only resources needed by each staged test.

## Legacy equivalence checkpoint

The inherited direct-lapse patch also changed the optional global L projection
in legacy mode. Therefore merely selecting `reduction_system=legacy` does not
reproduce the collision source. Added explicit
`lapse_projection_target=collision_factorized` for legacy controls; the default
remains `direct_product`. Unknown targets and collision targets in advective
mode fail at input validation. No bulk equation or projection timing changed.

Two independently built full CPU executables use identical test adapters against
the collision source and current source. Seven smooth non-diagonal off-constraint
seeds, all active cells and all 55 fields, FD2/4/6, and 2D/3D anisotropic fixtures
have exactly matching saved RHS, algebraic/GH projection and auxiliary projection
values with the collision target. These are sampled nonlinear equivalence checks,
not a proof for arbitrary jets. The direct-product negative control differs by
0.0010--0.0049 in the projection fixtures, confirming the target distinction.
Both invalid-input tests pass. CPU raw outputs/builds are preserved outside Git
under `../pcgh-clean-reduction-tests-20260907`; compact results and complete source
manifests are in the dated evidence directory.

Independent CUDA builds and controls have completed in
`/scratch/gpfs/FPRETORI/hz0693/pcgh-clean-reduction-20260907-legacy`.
Six zero-step fixtures match to normalized error at most 5.53e-17; six actual
RK3 one-step fixtures match to at most 3.62e-17 (frozen tolerance 2e-12).
The CPU one-step counterparts match exactly. The old-source CUDA one-step
executable includes only the view-capture repair required to avoid host-this
access; its patch, complete source manifest, executable hashes, inputs and raw
output hashes are retained. The exact unpatched CUDA executable is preserved.
The CUDA build and test controller exit files both contain zero. Raw CSVs remain
outside Git; compact records are in `cuda-legacy-001` in the evidence directory.
These comparisons do not exercise MPI, AMR, restart, or a physical solution.

An opt-in `state_budget` writer now records before/after/signed increments for
all fine-array state components, including ghost cells, at existing operation
brackets and before post-RK validation. CPU 2D/3D independent-index fixtures,
no-op records and truncated-record rejection pass. The writer explicitly makes
no ghost-validity assertion. It does not yet record the coarse buffer, separate
GH from algebraic correction, or compute independent constraint increments;
these limitations prevent calling it a complete causal budget.

A proposed shifted derivative-ghost stencil was screened in a periodic TT toy
subsystem. Floating-point eigenvalues flagged tiny positive roots near neutral
modes, including the uniform control. Exact rational FD6 spatial matrices for
two blocks of 8 or 16 cells have distinct real nonpositive eigenvalues and a
constant kernel. This resolves those no-KO toy flags without changing a numerical
tolerance. It does not establish a uniform energy bound, KO/RK stability,
multidimensional/AMR stability, or a complete transfer repair. Both raw flags and
exact polynomials are preserved; no production halo operator changed.

Next: complete the archived transfer discriminator and implement full coherent
transfer with valid stencil support and signed operation diagnostics, then the
intrinsic map and complete kernel oracles. Gates 1--5 remain unpassed.
