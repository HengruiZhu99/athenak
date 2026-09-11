# Execution log

User approved implementation and Perlmutter shared_interactive tests on a separate branch/worktree. The review PLAN.md is the original proposal; its pre-approval status paragraph is historical.

- Existing local worktree vc-corner-fix was clean at c930074d, matching the latest relevant origin branch after fetch. SSH push reported everything up to date, so no empty commit was manufactured.
- Local new branch: codex/vc-cartoon-subcycling-20260911; worktree: /Users/hz0693/research/collapse/vc-subcycling.
- Remote separate bare repository and worktree: /pscratch/sd/h/hzhu/vc-subcycling-20260911/{repository.git,source}. Uses the existing qualified Kokkos dependency via symlink.
- Added default-off time/execution_profile. Timings fence Kokkos operations; hierarchy distinguishes inclusive from exclusive work and initialization from evolution. Fences can perturb execution overlap; paired uninstrumented measurements are required.
- Paired production/profile restarts prepared at t62.03460401194272 (3434 blocks, cycle118310) and t62.20556032298022 (8924 blocks, cycle133489). Twelve live-AMR cycles each, original physics and output cadence, separate output/AMR files. Full checkpoint payload and history comparisons are required.
- Source inspection found a quadratic contributor scan in shared-node topology construction. It is a candidate performance bottleneck; await measured evidence before changing direction.
- Initial remote configure failed because git's empty submodule directory received a nested symlink. Replaced only that newly created empty directory with the intended dependency symlink; restarted configure/build.

No asynchronous level evolution or production campaign changes have been made at this point. Gate results will be recorded here.

## Classical RK4 reference implementation (in progress)

Added opt-in `time/integrator=rk4_classical` for vacuum Z4c. CopyU retains
beginning-of-step u1; ExpRKUpdate accumulates the four weighted RHS evaluations
in active-block-sized scratch and retains the existing axis regularity,
zero-shift, admissibility and AMR lifecycle hooks. The source timestep contract
uses the classical stability polynomial and its2.7852935634 negative-real radius.
The old `rk4` path is unchanged. This is synchronous stepping only, not subcycling.

Local CPU full executable build passed. `athena_classical_rk4_test` measures
nonautonomous nonlinear ODE error ratios15.5164,15.7631,15.8829 under successive
halvings and checks the source stability radius. A four-step8^3 Z4c linear-wave
smoke test with the actual executable terminated on its requested cycle limit.
These checks do not qualify VC Cartoon, PDE temporal convergence, temporal
boundary conditions, or production gauge behavior. Those remain required.

Recovered the earlier remote build: it completed, but its post-build workflow
failed because kokkos was a symlink that Git rejected. Preserved pre-sync.patch,
replaced only that symlink with a shared clone at the identical gitlink SHA,
fast-forwarded remote source to aa41ea73 and restarted the incremental profiling
build. Production files and jobs were not modified. The new local classical-RK4
changes are deliberately not mixed into the baseline profiling executable.

Goal remains incomplete: level-local scheduling, parent states, stage-consistent
boundaries, global gauge qualification, dynamic AMR and single-A100 matched-time
reproduction through the provisional run's end time all remain outstanding.

## Native VC Cartoon temporal reference gate

Added an optional Gaussian lapse pulse to the exact-flat-geometry Minkowski
carrier (amplitude zero retains the default). Nonzero amplitude is explicitly
restricted to the native VC Cartoon coordinate map and validated for positive
lapse/width. Initial spatial geometry and extrinsic curvature remain flat;
subsequent gauge evolution is nontrivial.

`test_classical_cartoon.py` uses the production Z4c task graph on a fixed16x32
meridional grid, CFL0.4,0.2,0.1,0.05 to t0.5. All25 evolved fields on the radial
slice z0.5 are compared using double-precision text output. Successive RMS
errors7.7640e-7,5.2873e-8,3.3109e-9 give ratios14.6843 and15.9696. Local full build
passed. This is temporal self-convergence on a slice, not spatial convergence,
full-domain qualification, or asynchronous-boundary qualification.

Two measurement problems encountered were retained in /tmp/vc-classical-cartoon-1
through4: binary output stores float32 and obscures the finest differences;
VC table output exactly at z0 (a shared block face) had no rows because selected
block and canonical diagnostic owner differ. The final test samples z0.5,
inside a block, and does not claim to fix or test that table-output issue.

Reproduce with the bundled Python (NumPy required):
`python scripts/subcycling/test_classical_cartoon.py /path/to/athena /new/output/path`.
The output directory must not already exist; failures and raw outputs persist.

## Scheduler ordering and stage-time groundwork

Corrected classical-RK4 RHS evaluation of explicit time-dependent damping ramps
to use stage time. Existing low-storage behavior is unchanged. Native pulse
self-convergence with roll_kappa enabled gives ratios14.6862 and15.9700. This
check retains the scope limitations of the preceding slice test.

Added a recursive, callback-based interval scheduler with integer tick identifiers,
coarse-level grouping and power-of-two ratio caps. Tests cover ratios1,2,4,16,32,
parent prediction before children, synchronization only after child completion,
shallow trees, invalid ratios and immediate failure propagation. Callbacks are
responsible for numerical predictors, transfers and rollback. This scheduler is
NOT yet connected to Z4c evolution; there is no functional subcycling runtime
option yet. Next integration work must supply those numerical callbacks, with
stage-consistent boundaries and qualified global gauge behavior.

Remote continuation: finish_profiles.sh is running as PID1635650, protected by
profile-launch.lock. It waits for the verified build PID1579887 to exit, requires
the successful build marker and clean aa41ea73 source, verifies the executable
hash, builds the profile unit test, then requests one shared_interactive A100
for the four prepared checkpoint comparisons. It writes profile-launch-status,
allocation.log and comparison.log. Do not start a duplicate launch if observation
times out; inspect PID1635650, its child allocation, and terminal status first.

## Synchronous per-level RHS execution

Added default-off time/level_batch_rhs. Persistent device block lists are grouped
by logical level and rebuilt when the local level sequence changes. The main,
Gamma, gauge and KO RHS kernels use these batches. Global gauge reductions,
axis regularity and boundary tasks still run on the common-time whole hierarchy.
This is synchronous level-local kernel execution, NOT asynchronous evolution.

Full CPU executable build and schedule/classical unit tests pass. On a two-level
14-block native Cartoon gauge pulse, CFL0.4,0.2,0.1,0.05 runs through t0.5 have
byte-identical complete restart payloads with and without level batching
(parameter text excluded). Both paths show radial-slice temporal self-convergence
ratios15.1765 and15.2375. JSON evidence is in classical-cartoon/static-*.json and
batch-comparison.json. GPU qualification, dynamic hierarchy changes and actual
asynchronous coarse/fine evolution remain unverified.

Profiling allocation58198556 is confirmed queued for resources; launcher1635650
is live. No duplicate allocation has been requested.

## Dense boundary and covered-parent groundwork

Implemented the scalar component formulas for classical-RK4 coarse dense output
and fine stage reconstruction from Ji et al.2503.09629v2 equations11-19. Tests
compare reconstructed child stage vectors against direct nonlinear,
nonautonomous RK stages in both half intervals. Local boundary-error ratios
17.6055,16.8023,16.3995 approach16. A separate check rejects interpreting the
stage2 vector as merely the physical dense state at its nominal time. These
helpers are not yet connected to runtime coarse/fine ghost filling.

Added a sparse hierarchy containing active leaves and their covered ancestors,
with deterministic indices, source-leaf references and parent/child links.
2D/3D tests require complete child sets and reject overlapping, incomplete and
duplicate leaves. This is topology for future predictor state allocation, not
an implemented parent-state evolution path.

## Profiling source-identity gate

Allocation58198556 was CANCELLED before starting after salloc timed out; verified
by sacct (zero elapsed). Retry58198653 ran the early baseline successfully
(12cycles,51.0634s including initialization/output). The profile binary failed
before evolution with AMR history source-id mismatch. This is not an evolution
instability. Original evidence is retained under profiles and profiles.58198653.json.

Extended the existing explicit amr_history_compatible_source_id option to AMR
record continuation, logging both source IDs and mode. All other header,
checkpoint digest and topology checks remain. Local tests verify default
mismatch rejection, explicit correct-source continuation through a further
cycle, and wrong-source rejection. Evidence: source-compatibility-results.json;
raw local evidence: /tmp/vc-source-compat-test4.

Remote profiling-only source is47a11d37 (aa41ea73 plus this compatibility change;
no classical-RK4 or subcycling changes mixed in). Build PID1674145 is live.
Guarded continuation PID1695372 waits on that build and then runs new cases in
profiles_source_compat, leaving the earlier cases intact. Its script is
finish_compatible_profiles.sh; inspect compatible-launch.log, allocation.log,
profile-launch-status and actual Slurm handles before any retry. It uses the
manifest-aware comparison script copied separately in the remote root.

The full subcycling goal remains incomplete: populated parent predictor fields,
asynchronous Z4c steps, temporal ghost filling, global telegraph consistency,
dynamic AMR synchronization and faster matched-end-time reproduction are still
required. Helper tests are not substitutes for those gates.

## Parent-state initialization on native storage

Added VertexParentStates with storage only for covered ancestors. It populates
active VC points by bottom-up injection from already reconciled child vertices;
parent ghost storage remains NaN. A two-generation polynomial test verifies
exact coordinate-consistent injection, untouched leaf arrays and invalid ghosts.
The first implementation explicitly rejects non-Cartoon/non-even-block layouts.

Z4c::RebuildSubcycleParents builds this hierarchy from the real mesh leaf list.
A unit-test-build-only ATHENA_TEST_SUBCYCLE_PARENTS environment hook exercises it
after startup boundary initialization. On the 14-block static Cartoon fixture,
it allocates two parents (250000 bytes). Four CFL runs through t0.5 produce
byte-identical complete restart payloads to runs without parent initialization.
Evidence: parent-initialization-results.json; raw outputs are under
/tmp/vc-cartoon-parent-runtime. The hook stores an initialization snapshot only;
it does not advance parents, fill their ghosts, or enable subcycling.

Remote compatible profiling build finished successfully. Guarded launcher1695372
has requested allocation58198945, currently pending resources. Preserve this
handle and recheck its actual state rather than launching a duplicate.

## Same-level auxiliary ghost filling

Parent storage now copies ghost values only from same-level active donors,
including covered parents. Exact shared coordinates use deterministic lower-node
ownership; missing physical/coarse-fine donors remain NaN. Layout, topology and
source-index compatibility are checked; logical coordinate products use64-bit
arithmetic. This is not a complete boundary provider: temporal coarse/fine,
axis parity and physical outer boundaries remain to implement.

The polynomial test now covers two adjacent covered root blocks and a second
refinement generation, exercising leaf and parent donors, missing-donor counts,
layout rejection and unchanged leaves. The real startup test reports472 copied
and200 unavailable ghost points for two parents. All four CFL runs retain
byte-identical complete restart payloads. Evidence is in
parent-same-level-ghost-results.json.

Allocation58198945 is RUNNING on nid008213. Early baseline and instrumented
12-cycle restarts completed successfully (~51.27s and51.06s total respectively).
The late baseline is currently running. Early profile evidence is copied to
profiling-early/: RHS inclusive0.49672s, scheduled outputs7.93999s, initialization
topology rebuild11.39476s. The short-window totals include initialization and
final outputs; do not treat them as asymptotic evolution throughput or a subcycling
speedup. Full-state comparison is still pending the four-case workflow.

## Topology construction optimization and first GPU identity evidence

Replaced the per-group full contributor scan with a linear grouped pass after
the existing sort. Finest-authority membership and within-group contributor order
are preserved exactly. Deterministic randomized tests compare all offsets,
authority levels and contributor sequences against the original quadratic
algorithm. Four static Cartoon runs retain byte-identical restart payloads.
This optimization is additional to the required subcycling implementation; it
is not a replacement for asynchronous evolution or its validation.

The early GPU instrumented/base comparison now passes: all history columns
identical, AMR histories identical and complete restart payload SHA256 identical
(4a9102988f25b10c5a91cc1467de38ff63d7a7da8a247c3108334b42d1fef866), through
t62.035027597047765. Evidence: profiling-early/comparison.json.

Late baseline completed12 cycles in248.7253s total, with28.3963s reported after
initialization. First recorded live-AMR event after the input checkpoint occurs
at cycle133506 (17 steps later), outside this12-step sample. A subsequent
performance comparison must extend past that event to measure topology rebuilding
during evolution. Allocation58198945 is still running the late instrumented case;
no source or executable used by that allocation was modified.

## Axis boundary initialization and completed profiling gate

Auxiliary parents now reuse the existing native-VC axis reflection routine,
with explicitly supplied component parities. The real test hook takes parities
from Z4cStateAxisParitySignFromPackedIndex and runs only for an actual axis
boundary. It mirrors positive-rho vertices without touching the evolved axis.
Transverse unavailable ghost corners remain NaN. Unit tests exercise both parity
signs, and four startup-hook runs retain byte-identical restart payloads.
Physical outer boundaries and temporal coarse/fine boundaries remain outstanding.

Allocation58198945 completed0:0. Both early and late profile/base comparisons
have identical history columns, AMR histories and complete restart payloads.
Late profile: initialization135.5483s including topology81.5769s;12 evolution
cycles5.3309s including RHS1.0903s; total246.8573s. Baseline total248.7253s.
Evidence is profiling-comparison.json. These are synchronous runs, not a
subcycling speedup measurement.

Archived the profiling executable as binaries/athena.profile-47a11d37 with its
SHA256 and CMake cache on Perlmutter. Remote source now follows development
branch at a127a1fe; build PID1778061 is live. New topology24 cases use24 steps
at the late checkpoint, crossing the recorded first AMR event at step17.
Guarded launcher1808047 (finish_topology24.sh) waits for the build, compiles/runs
GPU helper tests, and requests a separate shared_interactive single-A100
comparison. Inspect topology24-launch-status and allocation_topology24.log before
retrying. Existing campaign jobs and production outputs remain unchanged.

## Sparse coarse RK histories connected to synchronous test path

Previous goal turn was planning-only (no implementation progress). Revalidated
remote build1778061 as live, then observed successful terminal build. Guarded
launcher1808047 submitted topology24 allocation58199493; last inspected PENDING
(Priority), so no duplicate submission or build. Production58197653 remains
running and unchanged.

Added RK4PredictorStates: owns beginning state and four RHS arrays for an explicit
selected block list, with active vertices only (no invalid ghost RHS or reserved
capacity storage). Requests require a complete ordered parent step and child
interval bounds. Device stage evaluation uses the existing stage-consistent dense
formula. This is distinct from a physical endpoint after algebraic projection.
Spatial boundary sampling still must gather a valid active-donor stencil; these
arrays are not a complete asynchronous boundary provider.

ATHENA_TEST_RK_PREDICTOR in a kernel-test build captures actual RHS stages before
classical updates, without changing the fields or consuming the predictor.
Unit tests cover sparse order, independent ownership after source overwrite,
invalid/incomplete stages, reuse after a new interval and empty batches.
The first unit attempt exposed a test-fixture host mirror alias on the Serial
backend; using an independent host mirror fixed the fixture. Five helper tests
pass after explicitly building the previously unbuilt authority test target.

Full CPU executable rebuilt with changes; four native static-AMR Cartoon tests
with capture enabled retain byte-identical complete restart payloads against
parent-axis baseline. Temporal ratios15.17646,15.23751. Evidence:
rk-predictor-results.json, raw /tmp/vc-cartoon-rk-predictor-runtime; build log
/tmp/vc-predictor-build.log. Build source SHA identifies base96d1e912 plus this
commit's changes (uncommitted at build time). These remain synchronous tests.
Asynchronous parent RHS/ghost evolution, global gauge coupling, dynamic AMR and
faster full matched-end-time production reproduction remain outstanding.

## Temporal/spatial coarse boundary sampling

Previous goal turn made implementation progress (3103ef04). Revalidated guarded
launcher1808047 and allocation58199493; allocation remains pending (resource/
priority reasons). No duplicate allocation was submitted.

Added VertexTemporalBoundary, connecting retained RK stages to native VC spatial
midpoint stencils. A geometry plan resolves every stencil vertex to a same-level
ACTIVE donor, across faces and corners, preferring lower logical donors at ties.
It supports existing orders4/6/8, explicitly binds predictor source-block order,
and rejects missing/physical-boundary support rather than substituting a lower
order. Failed topology rebuilds invalidate the old plan. Stage scratch is reused
when the shape is unchanged. Results are batched (target,component) values;
consuming ghost scatter, axis/outer boundary policies and parent RHS advancement
are still required before runtime subcycling works.

CPU unit test vertex_temporal_boundary passes for spatial polynomials of degree
3/5/7 respectively, permuted block/source order, coincident vertices and
face/corner-spanning stencils. Manufactured u(x,z,t)=shape(x,z)*exp(t) checks all
four child RK stages over both half steps. Successive local stage errors have
ratios16.0819,16.0405,16.0201 for each order. This is a manufactured stage-boundary
test, not global Z4c temporal convergence or production validation. Missing
stencils and changed donor ordering reject. rk4_predictor_states also passes.
Build/test logs /tmp/vc-temporal-build.log, /tmp/vc-temporal-config.log; tests use
/tmp/vc-subcycling-cpu. GPU compilation of these new headers is still pending
(the queued topology24 job uses older immutable source a127a1fe).

Allocation58199493 was revoked after the salloc queue timeout; sacct confirms
CANCELLED elapsed00:00:00 and launcher1808047 is gone. Archived its allocation,
launcher and status logs with job-ID suffixes. Retried the same immutable source
and comparison (no source rebuild/change): new launcher PID1875948, allocation
58199685 confirmed queued. SSH submission session77853 is still attached to the
background shell; do not interpret that observational session as a duplicate
job or retry while the confirmed allocation/launcher is live.

## Level-local numerical RK update used by Z4c

Previous turn progressed via temporal/spatial boundary sampler fcc60689. Verified
allocation58199685 and launcher1875948 still live; allocation pending Resources
at the final check. No new allocation submitted.

BlockBatches now accepts an explicit level range and preserves other levels.
Extracted the classical RK numerical update into a shared kernel taking explicit
dt and block batches, with no mesh time/dt lookup. Z4c's synchronous classical
path now calls this exact kernel (all levels, retaining existing post-update
axis/projection/diagnostics). This does NOT enable asynchronous global tasks.
The production wrapper still obtains dt from the mesh, and global CopyU/BCs need
level-local consumers before recursive Z4c evolution is safe.

New level_rk_update test calls Schedule with a grouped coarse level and ratios
1/2/4, evolving a nonautonomous ODE using StepContext stage times. It checks
coarse/fine update counts, forced-zero components, unchanged inactive levels and
all ghosts, plus empty selected-level batches. Error ratios17.4669,17.1694,
16.725 approach fourth order. This test has no spatial PDE coupling; it does
not qualify asynchronous coarse/fine Z4c evolution.

Full executable rebuilt; four static-AMR native Cartoon runs using level batches
and RK-history capture reproduce full restart payloads byte-for-byte against
3103ef04 capture baseline. Temporal ratios15.17646,15.23751. Evidence:
level-update-results.json; raw /tmp/vc-cartoon-level-update-runtime. Four relevant
CTest cases pass. Build logs /tmp/vc-level-rk-build.log and
/tmp/vc-level-rk-rebuild.log. Build identifies base1bd235a6 plus uncommitted changes
now included in this commit. GPU qualification of these changes remains pending.

## Physical ghost support for covered parents

Previous turn made implementation progress (3b1a1f42). Confirmed allocation
58199685 revoked after queue timeout: sacct CANCELLED elapsed00:00:00 and
launcher1875948 gone. Archived its logs with job-ID suffix, reduced requested
allocation from90min to30min (same24 numerical steps and unchanged inputs), and
retried. New launcher1911927 and job58200152 confirmed pending Resources.
No production campaign modifications.

Moved existing leaf Extrapolate<2/3/4> formulas verbatim into shared
z4c/physical_extrapolation.hpp; leaf callers now use the namespaced function.
Covered parent physical fills use those same formulas, explicit root-domain
extent and enabled faces, x1 then x2 corner ordering. Only configured
outflow/diode/vacuum faces are enabled by the startup test hook. Axis remains
its separate preceding provider. No physical RHS conditions or parent evolution
are enabled by this change; unknown/interlevel ghost support is not substituted.

Cubic parent fixture tests exact values including outer corners, disabled-face
missing counts and domain validation. Initial assertion expecting leftover
interlevel NaNs was wrong: this fixture has same-level donors for every interior
ghost. Corrected that expectation; missing disabled physical faces are still
explicitly checked. Unit test passes.

Full CPU executable built. Four static-AMR Cartoon runs retain byte-identical
complete restart payloads vs level-update baseline; evidence
parent-physical-results.json. Their parents touch only the axis, so additionally
ran a fully refined domain to exercise actual outer-parent fills:8parents,
800physical targets, with and without parent initialization gives identical
payload9ca1d4eac27694fbef13fab09cf013559e2f02ce42d781d43378c4583938bbde.
Evidence parent-outer-results.json, raw /tmp/vc-parent-outer-runtime; four-CFL raw
/tmp/vc-cartoon-physical-parent-runtime. This is initialization/leaf invariance,
not an asynchronously evolved parent or production-gauge qualification.

## Shared continuum bulk RHS evaluated on an auxiliary parent

Previous turn made progress via physical parent fills b26ff2a0. Allocation
58200152 now RUNNING on nid008445. Four GPU helper tests ran before the late
baseline; baseline24steps completed335.87094s total with exit0, crossing the
first AMR event at cycle133506. Development profile now running; full comparison
pending. This remains synchronous optimization evidence, not subcycling.

Extracted the three continuum RHS kernels verbatim into EvaluateZ4cBulkRHS with
explicit geometry, bound field views, block batches, stage time and gauge scales.
It has no mesh/driver pointer or global time lookup. The production leaf wrapper
calls the same function; global gauge reduction, KO, boundary RHS replacement,
axis audits and diagnostics remain in their existing surrounding paths. A shared
BindStateViews function replaces the duplicate state/RHS tensor view setup.

New analytic test evaluates the actual native VC Cartoon bulk operator on an
injected covered parent with axis and physical ghosts. Flat metric and lapse
1+.01*rho^2+.02*z^2 give trace RHS-.08 and the expected anisotropic traceless
Hessian. Maximum error2.62151e-14 across active vertices, with ghosts unchanged.
Initial test expectation swapped the packed axial/suppressed tensor directions;
corrected to the authoritative (rho,z,suppressed) coordinate mapping. Bulk RHS
is not yet a complete parent operator: KO, boundary RHS and axis checks still
need explicit parent consumers before asynchronous evolution.

Full CPU executable rebuilt. Four static Cartoon runs, with parent initialization
and RK capture enabled, retain byte-identical complete restart payloads and
ratios15.17646,15.23751. Evidence bulk-rhs-results.json; raw
/tmp/vc-cartoon-bulk-rhs-runtime. Analytic unit parent_bulk_rhs passes. Build logs
/tmp/vc-parent-rhs-build.log and /tmp/vc-parent-rhs-unit.log. GPU compilation of
this extraction remains outstanding; queued/running topology24 uses a127a1fe.

## Completed topology optimization GPU comparison

Job58200152 COMPLETED0:0 (7m20s), one A100 on nid008445. Baseline24steps took
335.87094s total; optimized sourcea127a1fe took91.49285s (3.671x). All history
columns exactly equal, AMR history identical, complete restart payload SHA256
451f84d0dce7d7f2d9f08f5f3f612cf0de8a9c63a898b2659a5a253637205304 identical.
Matched final time62.2056184751341; one actual AMR event crossed. Profile evolution
24cycles11.71898s; topology rebuild during AMR0.37489s; initialization54.53511s
including topology0.33516s. Evidence topology24-comparison.json and
topology24-timing.json. This is a short synchronous restart speedup from topology
optimization, not full provisional endpoint reproduction or subcycling success.

Archived immutable GPU executable binaries/athena.topology-a127a1fe with hash
72d03a012dd24b46a643c8b2b692ad261caaa47056593733d8851f1f2408715c and CMake cache.
After confirming allocation/launcher terminal, fast-forwarded isolated remote
source to faaf6973. New GPU build PID1953857, parent-rhs-build.log and
parent-rhs-build.pid. No new GPU allocation yet; build completion and new helper
GPU tests must be checked before another evolution comparison. Production
campaign remains unchanged.

GPU build1953857 failed and is confirmed terminal: NVCC disallows an extended
host/device lambda inside private RK4PredictorStates::CopyActive. Moved that
unchanged copy kernel to namespace-scope detail::CopyRKActiveValues. Predictor
and temporal-boundary CPU tests pass after the portability fix. Archive the
failed build log before retrying; no GPU evolution was launched with faaf6973.

Retried GPU compilation only after failed worker disappeared. Remote source now
730f4d99; build PID1962404, parent-rhs-build.pid/log. Failed log preserved as
parent-rhs-build.faaf6973-failed.log. No allocation currently attached to this
build and no new GPU test submitted yet.

## Shared dissipation and queued-after-build GPU validation

Previous turn progressed through shared parent bulk RHS and completed the exact
GPU topology comparison. Revalidated retry build1962404, still live at82% after
6m29s, with no new fatal error observed. Source remains730f4d99 during compilation.

Extracted unchanged KO kernels into AddZ4cDissipation with explicit fields,
geometry, boundary flags and block batches. Existing leaf pre-KO axis check and
stage diagnostics remain in the wrapper; axis projection after adding KO remains
inside the native Cartoon operator. Parent callers still need an explicit
pre-KO axis check and physical boundary RHS before being complete.

Parent bulk-RHS test now also checks KO on the smooth manufactured state and an
even grid-frequency chi mode. Scalar mode matches expected damping with maximum
error7.64363e-18. Full CPU executable rebuild and four static Cartoon runs retain
byte-identical restart payloads vs bulk-RHS baseline. Evidence
dissipation-results.json, raw /tmp/vc-cartoon-dissipation-runtime, build log
/tmp/vc-dissipation-build.log, analytic unit log/tmp/vc-parent-ko-unit.log.

Added optional shell-free launcher prefix to test_classical_cartoon.py so each
GPU executable invocation can use its own srun step. Default CPU path passed.
Copied that script outside remote source as test_classical_cartoon_launcher.py;
no source/executable mutation during the live build. Guarded launcher2001088
(finish_parent_gpu_730.sh; local evidence copy saved here) waits for build1962404,
checks source/hash, builds seven helper targets and requests a15min single-A100
shared_interactive allocation. Runs helper GPU tests then the four-CFL static
Cartoon test with parent initialization/RK capture. Logs parent-gpu-validation.log,
parent-gpu-unit-build.log, parent-gpu-allocation.log; status/pid named analogously.
No allocation exists yet at this check. GPU validation covers source730f4d99,
not the newer KO extraction in this commit. Production campaign untouched.

## Parent axis gate and physical boundary RHS

Previous turn progressed with shared KO. GPU build1962404 succeeded, and guarded
launcher2001088 completed validation job58200675 on nid008552 (COMPLETED0:0,
26s allocation). Seven helper tests passed on A100. Native static Cartoon
four-CFL test ratios15.17644,15.23920; CPU/GPU radial-slice max differences range
2.64e-14 to1.47e-13 with identical coordinates. This is slice comparison, not
full-restart byte equality across backends. Evidence gpu-classical-730-results,
-cpu-comparison, -tables and -evidence files. Validated executable archived as
binaries/athena.parent-rhs-730f4d99 with hash and CMake cache. No GPU validation
allocation/build remains active at this point. It covers730f4d99, not later KO
or boundary extraction commits.

Added EnforceLocalVertexAxis using explicit selected blocks and the unchanged
point correction/tolerance rule; leaf lean wrapper now calls it, leaving its
full audit path unchanged. Expected-failure child processes reject excessive
and nonfinite corrections (returncode-6); evidence parent-axis-rejection.json.

Moved existing Sommerfeld and full-constraint Bjorhus point algorithms verbatim
into boundary_rhs.hpp. Leaf sweeps still call the same functions. Added a
selected-block VC Cartoon boundary sweep for auxiliary parent consumers. Tests
preserve flat RHS at faces/corners and correct a nonzero Theta incoming RHS only
at owned physical boundary points. The initial pulse expectation incorrectly
included axis/outer-boundary intersections: the established CPBC ownership rule
explicitly excludes the whole Cartoon axis, including those intersections. The
parent implementation preserves that policy; adjusted test expectation to match.
Existing full_constraint_bjorhus unit tests also pass.

Full CPU executable rebuilt. Four Sommerfeld Cartoon runs retain byte-identical
restart payloads (boundary-rhs-results.json). Added an explicit boundary selector
to the timestep test; four CPBC runs pass with ratios15.17646,15.23751
(cpbc-rhs-results.json). Raw /tmp/vc-cartoon-boundary-rhs-runtime and
/tmp/vc-cartoon-cpbc-rhs-runtime. The parent operator pieces are available, but
recursive coupled evolution, temporal ghost scatter/ownership, common-time
telegraph gauge, synchronized live AMR and full endpoint reproduction still
require integration and qualification.

## Unified populated hierarchy storage and geometry

Resumed the implementation goal after the review-only response (that response
made no implementation progress). Rechecked the completed cached-storage build
and runtime results before continuing; no stopped build was relaunched.

VertexParentStates now has an explicit all-node mode holding active leaves and
covered ancestors in one node-indexed field allocation. Cached restriction lists
support synchronization of one covered-parent level without per-call allocation.
Explicit active-leaf scatter excludes ghosts and reserved capacity. Same-level,
axis and physical ghost fills accept a target-level range. These are storage
operations; caller time consistency is still required and not yet enforced by a
coupled runtime scheduler.

Added HierarchyGeometry for populated node geometry and physical boundary flags,
with internal interfaces marked block, explicit root-block counts, nonzero root
levels and domain validation. This provides metadata to the extracted actual
Z4c kernels; it is not yet wired to asynchronous evolution.

Expanded vertex_hierarchy_state regression covers polynomial injection, cached
restriction without field reallocation, selective ghost fills, unchanged external
leaf data until scatter, twenty unused reserved blocks, boundary flags and
non-square root-grid geometry, and invalid-domain/mode rejection. Five related
CPU tests pass: vertex_parent_states, vertex_temporal_boundary, level_rk_update,
parent_bulk_rhs, vertex_hierarchy_state. Full athena cached-storage build passed.
All-node startup round-trip static Cartoon four-CFL results retain byte-identical
restart payloads against the boundary-RHS baseline; temporal ratios15.1764614,
15.2375065. Evidence all-node-storage-results.json; raw
/tmp/vc-cartoon-all-node-runtime. No GPU validation of these additions yet.

Full coupled stepping, temporal ghost ownership/scatter, common-time telegraph
coefficient, live AMR/diagnostics and faster single-A100 reproduction through the
recorded endpoint remain incomplete. No production run or campaign was changed.

## Cached same-level active/ghost vertex exchange

Previous turn made verified progress (a9ecd6eb). Added HierarchyVertexExchange,
which caches transfers by logical level for populated hierarchy fields. It
reconciles duplicate active vertices and fills same-level ghosts from canonical
active donors in a single device phase. Canonical donors are never destinations,
so no staging array or read/write race is introduced. Apply selects levels and
performs no topology reconstruction or field allocation. Coarse/fine and physical
missing donors remain for their separate providers; same-stage/time consistency
within each selected level remains a caller obligation. Failed Build invalidates
the old plan; invalid Apply shapes/ranges reject.

The expanded hierarchy-state unit initializes inconsistent duplicate values,
independently scans containing blocks to establish expected authority, verifies
all target values, untouched other levels, idempotence and invalidation. Passes.
The test-only all-node startup path applies this exchange before ghost handling
and leaf scatter. Full CPU executable built and four static Cartoon runs preserve
all restart payloads exactly against all-node-storage baseline; temporal ratios
15.1764614,15.2375065. Evidence shared-exchange-results.json, raw
/tmp/vc-cartoon-shared-exchange-runtime. Final edit after full build only adds
failed-Build invalidation; the updated unit was rebuilt and passed.

This removes a required ownership gap for actual per-level evolution, but does
not enable asynchronous Z4c stepping. Temporal ghost scatter and physical/axis
stencil coverage, common-time gauge, live AMR/restarts and faster matched-endpoint
single-A100 reproduction remain unfinished. No production files/jobs changed.

## Hierarchy temporal predictor ghost scatter

Previous turn progressed with852887a6. Added HierarchyTemporalGhosts to build
actual fine ghost destinations from populated hierarchy topology. It excludes
active fine vertices, ghosts with an active same-level donor, and out-of-domain
physical ghosts. Remaining targets are bound to the existing qualified native
spatial/stage-temporal interpolation. Coarse source IDs match hierarchy storage.
Apply evaluates coarse RK predictors and scatters all components in one device
phase with cached destination lists and reusable scratch. Failed rebuilds and
shape/component mismatch reject without destination writes.

Expanded vertex_temporal_boundary unit uses an interior refined patch in a4x4
root mesh, two fields and degree-five spatial data with exponential time behavior.
All four stages of both halfsteps agree with analytic fine RK stages within the
coarse predictor truncation error; stage1 at interval start is spatially exact to
2e-14. Checks all destination values, target count, unchanged active/coarse and
same-level-owned ghosts, component rejection and stale-plan rejection. Existing
order4/6/8 predictor convergence tests remain passing. Three related CPU tests
pass: rk4_predictor_states, vertex_temporal_boundary, vertex_hierarchy_state.
Build log/tmp/vc-temporal-ghost-build.log. No GPU or evolution job launched.

Boundary-adjacent interpolation whose stencil needs axis/outer support still
rejects explicitly. This requires a boundary-aware extension, not reduced-order
fallback. No complete asynchronous coupled evolution, production gauge or
matched-endpoint faster A100 reproduction is claimed.

## Axis parity for temporal coarse stencils

Previous turn progressed with46f0e440. Native temporal boundary plans now accept
explicit per-component axis parities. Negative-rho stencil points reflect to
active positive-rho donors; their contributions receive the corresponding parity
sign during the same batched interpolation. Target physical ghosts remain the
separate parity-fill responsibility. Caller must establish that logical rho=0
is the physical axis. Empty parity input preserves previous rejection behavior;
invalid signs/component counts reject. HierarchyTemporalGhosts forwards this
policy without changing active/same-level/physical target ownership.

Expanded CPU temporal-boundary tests: analytic even and odd fields, spatial
orders4/6/8, on-axis and near-axis targets with transverse block-boundary crossing,
all four stages of both halfsteps. Stationary polynomial errors below2e-14.
Hierarchy scatter test now repeats with a refined patch touching the axis and
nonstationary even/odd fields, verifying actual scattered ghosts and untouched
active/coarse/same-level/physical ghosts. Three related tests pass (predictor,
temporal_boundary,hierarchy_state); build/tmp/vc-temporal-axis-build.log.

Outer-face stencil extension remains missing; no lower-order fallback added.
Full coupled asynchronous evolution, production global gauge, live AMR and
faster single-A100 matched-endpoint reproduction remain incomplete. No production
source, jobs or outputs changed, and no GPU test was launched this turn.
