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

## Explicit outer-face temporal stencil extension

Previous turn progressed withc2f5f344. Added TemporalOuterExtension: explicit
coarse-domain upper vertex coordinates, extrapolation order2/3/4 and four face
authorizations. Out-of-domain coarse stencil samples expand into active donors
using degree(order-1) polynomial extrapolation, mathematically the existing Z4c
linear/quadratic/cubic ghost rule. Tensor-product corner expansion composes with
axis parity. This does not claim bit-identical floating arithmetic to the native
Extrapolate evaluation. Disabled faces or missing interior donors reject; native
spatial interpolation order remains unchanged. Axis and extrapolation ownership
of the same inner-rho face reject. Hierarchy wrapper derives domain coordinates
from root block counts and takes explicit extrapolation policy.

Tests cover all four outer corners/faces with order8 spatial interpolation and
extrapolation2/3/4 on matching-degree analytic polynomial data at every RK stage
and both halfsteps. Errors below2e-12. Hierarchy scatter now also exercises a
refined patch touching both axis and outer axial boundary, nonstationary even/odd
fields, and confirms only intended fine ghosts change. Predictor, temporal
boundary and hierarchy-state tests all pass; log/tmp/vc-temporal-outer-build.log.

These tests qualify the local interpolation/scatter layer, not the assembled
coupled Z4c integrator. Recursive stage integration, common-time telegraph gauge,
live AMR/diagnostics and faster A100 endpoint reproduction remain incomplete.
No production runs changed and no remote jobs launched.

## Assembled recursive hierarchy RK4 engine

Previous turn progressed with90d980b3. Added HierarchyRK4, assembling real
level-selected RK updates, initial-state copies, coarse predictor histories,
same-level exchange, stage-consistent temporal ghost scatter, recursive two-child
stepping, synchronized covered-parent restriction and shared-vertex reconciliation.
Physics callbacks supply physical/stage preparation, actual RHS and post-update
projection. Scratch arrays use populated node count. Each level's predictor is
retained separately while finer levels advance. Integer parent/child ticks select
fine-start predictor fractions. Completion fences at the common-time interval.

This initial engine runs factor-two steps at every level, with the scheduler's
20-level ratio limit. Bounded synchronous coarse grouping is not implemented;
that needs current-stage spatial boundary filling within the group. It is not a
production driver switch and has no gauge iteration, rollback or AMR wrapper yet.

New hierarchy_rk4 regression evolves u_t=u+0.1(u_x+u_y) with nonuniform linear
spatial data on a two-level, interior-patch hierarchy. Uses actual derivative
ghost reads, polynomial physical extrapolation, temporal predictor boundaries
and covered restriction. Analytic solution exp(t)*(1+0.1x+0.2y+0.03t). At t0.2,
2/4/8/16 coarse steps give maximum errors5.20556e-7,3.42642e-8,2.19807e-9,
1.39187e-10 (ratios15.1924,15.5883,15.7922). Checks exact coarse/fine RHS call
counts4:8 per interval. Four related CPU tests pass; raw evidence
hierarchy-rk4-transport.txt, build/tmp/vc-hierarchy-rk-build.log. Initial target
build needed CMake regeneration and generated-header include path correction;
subsequent build and test completed successfully.

Coupled Z4c consumer, bounded grouping, common-time global gauge and live AMR
remain to be implemented/qualified. Faster full-endpoint A100 reproduction is
still unverified. Production files/jobs untouched; no GPU submission this turn.

## Coupled Z4c engine consumer: convergence failure isolated to refinement

Previous turn progressed with5ea62e6d. Added experimental HierarchyPhysics<NGHOST>
consumer connecting actual native VC Cartoon bulk RHS, KO, physical RHS, axis
checks, prescribed-zero shift, state admissibility and final-stage conformal
projection to the recursive engine. Explicit stage-time callbacks supply maxK,
kappa1 and shift eta; no asynchronous maximum reduction occurs. The consumer
currently supports vacuum zero-shift experiments, not campaign deployment.

New small-Gaussian-lapse test uses telegraph lapse with prescribed maxK=1,
zero constraint damping, Sommerfeld, KO.02/64, and t0.04. A single-level control
passes temporal convergence (successive RMS1.48969e-11,8.57537e-13,5.15110e-14;
ratios17.3717,16.6476). The actual two-level subcycled test FAILS: successive
RMS4.41021e-10,2.24963e-10,1.13796e-10; ratios1.96042,1.9769. It completes finite
steps but exhibits first-order temporal sensitivity. This is not accepted as
qualification. The CTest hierarchy_z4c deliberately has an active convergence
gate and fails; hierarchy_z4c_uniform passes. Evidence
hierarchy-z4c-convergence-failure.txt. Build/tmp/vc-hierarchy-z4c-build.log passed;
core dumps disabled for the failing child test.

The result isolates an assembled coarse/fine coupling issue absent from the
single-level operator path and from the prior linear transport test. Next work
must diagnose stage boundary consistency, synchronization/restriction and
projection at the interface before proceeding to production gauge or GPU claims.
No production source/jobs changed. Full endpoint reproduction is still unproven.

## Order-loss isolation: restriction feedback

Previous turn progressed by exposing the coupled convergence failure. Added
compile-time test-only skip-restriction instrumentation to HierarchyRK4
(ATHENA_SUBCYCLE_DIAGNOSTICS, enabled only for hierarchy_z4c_test). Added test
consumer switches for KO and conformal projection plus a diagnostic-report mode;
default convergence gate remains active and unchanged.

Controlled results at the same timestep sequence:
- No covered-parent restriction: RMS1.41913e-11,8.47100e-13,5.20208e-14;
  ratios16.7528,16.2839.
- No KO: RMS4.41695e-10,2.25378e-10,1.14023e-10; ratios1.9598,1.9766.
- No conformal projection: same first-order values as full coupled baseline.

Evidence restriction-controls/*.log. The prescribed-gauge native Z4c operator
and one-way temporal boundaries can exhibit fourth-order behavior; first-order
sensitivity enters through repeated fine-to-covered-parent feedback. This is an
isolation result, not permission to remove restriction in production. Fine and
coarse spatial operators differ; resetting coarse predictors from the fine state
at synchronization without consistent two-way stage coupling is the next issue
to address. Need retain common-time restriction and test an appropriate coupled
predictor correction, not hide the failure by disabling feedback. Added named
no_restriction_control CTest alongside the still-failing default refinement gate.

Build/tmp/vc-hierarchy-z4c-build.log passed, controlled runs completed. No GPU jobs
or production changes. Full gauge/live-AMR and faster endpoint reproduction remain
unfinished and unverified.

## Restriction footprint and spatial controls

Previous turn progressed withd9ceaff9. Added test-only restriction-margin
instrumentation (same ATHENA_SUBCYCLE_DIAGNOSTICS guard; production margin zero).
Interior-only restriction (one-vertex margin) still gives ratios1.9186,1.9404;
three-vertex margin reduces error but trends toward order loss (14.70,3.55).
Thus this is not solely overwriting coincident boundary vertices.

Added independent nx16/nx32 spatial resolution controls and polynomial lapse
initial data. Neither restores fourth-order timestep behavior: nx16 ratios2.12,
2.03; nx32 ratios3.56,1.97; polynomial lapse1.94,1.97. Do not dismiss the failure
as an already-understood spatial floor. Test now reports maximum-difference
component and leaf-local position: default leading difference is component9
(Axy), coarse leaf1, j0/i1, on the physical boundary adjacent to refinement.

An additional interior-patch control moves refinement away from physical/axis
boundaries: error drops substantially but ratios2.264,2.020 remain. Therefore
physical boundaries amplify the discrepancy but are not required for order loss.
Evidence restriction-controls/vc-z4c-{interior-restriction,deep-restriction,nx16,
nx32,polynomial-lapse,interior-patch}.log. All diagnostic processes completed.
The default convergence test remains failing, not relaxed or marked expected-pass.

Re-read Ji et al.2503.09629v2 sectionII, equations11-19 and algorithm steps1-7
(https://arxiv.org/html/2503.09629v2). It specifies stage and accepted-endpoint
boundary updates before restriction. Our dense stage formulas agree, but this
alone does not prove correctness of the assembled vertex ownership/feedback.
Continue investigating consistent parent/fine stage coupling and accepted
interface states; these controls do not establish a fix. No production changes
or remote jobs; full goal remains incomplete.

## Rejected interface-vertex prescription hypothesis

Previous turn progressed withc6a32eeb. Implemented an isolated experiment adding
fine active vertices on refinement edges to the temporal boundary plan, prescribing
them at each stage and at accepted half/full endpoints with dense physical values.
The full restriction test still gives RMS4.09135e-10,2.11646e-10,1.07578e-10 and
ratios1.93311,1.96738. It does not resolve order loss. Preserved the exact experiment
as restriction-controls/prescribed-interfaces-rejected.patch and its raw log, then
removed the unvalidated implementation changes from all four affected sources.

Rebuilt the restored source. Required hierarchy_z4c still fails at the established
ratios1.96042,1.9769; uniform and no-restriction controls pass. No acceptance gate
relaxed, no boundary ownership change retained, no production work touched.

This rules out merely adding coarse-prescribed shared interface stage/end values
as a sufficient fix. Need consistent two-way coupling or a demonstrated analysis
of mixed spatial/temporal restriction error before claiming qualification. Full
single-A100 faster endpoint reproduction remains unverified and the goal active.

## Stage-coupled synchronous hierarchy reference and coarse groups

Previous turn progressed by rejecting/archiving the interface prescription
hypothesis. Implemented direct current-stage spatial evaluation of the existing
native interpolation plan (active hierarchy donor IDs, same parity and explicit
physical extrapolation, no synthetic RK history). HierarchyTemporalGhosts now
scatters this spatial evaluation for levels sharing an RK stage.

HierarchyRK4 now accepts a bounded maximum_ratio. The resulting coarser group
restricts its internal covered levels before each common RK stage, reconciles
shared vertices, and fills internal interfaces from current-stage spatial data.
Interfaces to genuinely asynchronous children still use the coarse RK predictor.
At interval synchronization, restriction covers every level in the group. Ratio1
is the fully coupled synchronous hierarchy reference; ratio2 on the two-level
test preserves the existing asynchronous behavior.

The real Z4c synchronous hierarchy reference WITH restriction now passes:
successive RMS7.73436e-11,4.40538e-12,2.65808e-13; ratios17.5566,16.5736.
Evidence hierarchy-z4c-synchronous-reference.txt. This establishes a stage-coupled
reference without removing feedback. Direct spatial ghost evaluation is checked
against the independently tested initial-time temporal evaluation for interior,
axis and axis/outer patches. Five of six relevant CPU tests pass; default
asynchronous hierarchy_z4c remains failing at1.96042,1.9769. Build/tmp/vc-group-build.log,
test/tmp/vc-group-ctest.log. No claim that the async error is fixed.

Bounded groups are now numerically implemented but mixed multilevel/group and
GPU qualification remain. Next work is consistent asynchronous two-way coupling
against this reference, followed by production-gauge/live-AMR/endpoint comparison.
No production files/jobs were changed.

## Experimental two-way stage-history corrector passes two-level Z4c gate

Previous turn progressed with2353aa56. Implemented an ATHENA_SUBCYCLE_DIAGNOSTICS
only interval corrector. Each pass restores the same saved full hierarchy state;
retains per-level/per-start-tick RK histories; reconstructs covered-parent stage
vectors and dense predictor RHS from the previous pass's first fine half-step;
then reruns the actual recursive RK engine with endpoint restriction still on.
Exception handling restores the interval start state. Fine-to-parent transfers
use native point injection into covered parents only. No physical coarse leaves
or ghosts are direct correction destinations. Coarse uncovered RHS sees the
updated covered-stage state through the existing shared/ghost exchange.

The reconstruction uses local fine RK Taylor derivatives to form parent stage
vectors and cubic dense history. It is explicitly not physical extrapolation
of the fine solution beyond its stored interval. This experimental path still
needs independent multilevel, stability and backend qualification before any
production option is enabled. It stores whole active fine histories per step;
memory and pass-cost optimization remain.

Actual coupled Z4c test with feedback: two passes insufficient (ratios5.453,4.009),
three passes give17.0094,16.3365; five passes17.0276,16.3827. Three- and five-pass
CTest convergence gates pass, as does the synchronous reference. Corrected versus
synchronous full active-leaf field RMS differences at4/8/16/32 steps are
7.70416e-11,4.38868e-12,2.64858e-13,1.67384e-14, consistent with fourth-order
approach to the same reference. Field ordering/configuration matched; raw arrays
/tmp/vc-{corrected,synchronous}-values-{4,8,16,32}.txt. Evidence corrector-evidence/
contains logs, CTest and comparison JSON. Build/tmp/vc-corrector-build.log.

The uncorrected async regression remains failing and unchanged. The corrected
path is not yet the production driver, and the tests are two-level prescribed-
coefficient experiments. Next qualify deeper/mixed-group hierarchies and GPU,
then promote a bounded/adaptive corrector and integrate common-time gauge/live
AMR. No production runs changed. Full faster-A100 endpoint goal remains active.

## Three-level/coarse-group corrector qualification on CPU

Previous turn progressed with7aa6bce1. Extended coupled Z4c test with a third
refinement level, a ratio2 synchronized coarse group, explicitly time-varying
prescribed maxK(t)=1+0.5t, and full_constraint_bjorhus boundary RHS. Three-pass
fully subcycled ratios17.015,15.4057; five-pass17.0645,15.4974; three-pass coarse
group17.4451,16.5551. Four new CTest gates (three_level,coarse_group,time_dependent,
cpbc) pass. Evidence multilevel-evidence/, build/tmp/vc-multilevel-build.log.
These remain short fixed-hierarchy vacuum tests; common-time gauge is prescribed,
not a self-consistent global maximum.

Perlmutter SSH confirmed working on login09. Isolated source clean at730f4d99;
no matching build/validation process found. Preparing a GPU helper build of the
new committed corrector test version in the existing isolated subcycling area.
No production jobs or sources touched. GPU execution and full endpoint goal
remain unverified.

GPU build dispatch: committed source261b4d25 transferred by incremental Git bundle.
Remote isolated source checked clean, then switched to codex/vc-corrector-gpu-261.
Started build_corrector_261.sh as PID94079; verified live with ps17s after launch.
Log corrector-261-build.log; terminal exit recorded in corrector-261-build.status;
executable hash on success corrector-261-executable.sha256. Targets are coupled
Z4c, recursive transport and temporal-boundary helper tests. No GPU allocation
submitted yet. Re-poll this PID/status before any retry or submission. Remote
source remains261b4d25 throughout this build. Production unaffected.

## CUDA local-type compile correction

Build94079 verified terminal: process absent, status2. NVCC rejects the function-
local TestPhysics type as a template parameter of the enclosing RK device lambda.
Moved the same test consumer to namespace scope, without numerical changes.
Local rebuild succeeded and five corrected Z4c CPU gates passed. Archived full
CUDA failure log in multilevel-evidence/corrector-261-cuda-failure.log. Preparing
one retry from the new committed source; no GPU allocation exists yet.

Retry dispatch: sourcec0464f5c fetched/checked out in the clean isolated remote
source as codex/vc-corrector-gpu-c046. Guarded build_validate_corrector_c046.sh
started as PID153387. Builds coupled Z4c, recursive transport and temporal sampler
GPU tests, then only on success requests a15min singleA10080GB shared_interactive
allocation (accountm3328_g). Run script captures source/hash/Slurm provenance,
executes five corrected Z4c cases with full active-field dumps, then two helpers.
Any build/test failure stops the sequence. Logs corrector-c046-build.log,
corrector-c046-allocation.log; terminal status corrector-c046.status. Do not retry
while this verified process remains live. No production simulation is included.

## GCC callback-name portability correction

Process153387 verified terminal (absent; corrector-c046.status2). No allocation
log/job was created. GCC host compilation rejects callback members named advance
and synchronize because their decltype declarations refer to enclosing lambdas
of those same names. Renamed members to advance_fn/synchronize_fn; numerical
operations unchanged. Local rebuild and corrected Z4c/transport tests pass.
Preparing a fresh named retry after committing this fix. Prior logs retained.

GPU retry103d8052: guarded build/validation PID164330, source branch
codex/vc-corrector-gpu-103d8052. Build log corrector-103d8052-build.log, status
corrector-103d8052.status, allocation log corrector-103d8052-allocation.log.
Scripts archived in multilevel-evidence. CPU full active-field dumps for all five
matching cases completed successfully in /tmp/vc-cpu-gpu103 with source/executable
hashes; these will support actual field comparison after GPU completion, rather
than only comparing convergence ratios. No production simulation changed.

GPU retry build completed successfully for all three targets. Guarded launcher
requested Slurm job58202346, shared_interactive, one A10080GB,15min. Verified queued
with squeue; wrapper164330 remains the live handle. No restart on observation
timeout. Compare CPU/GPU using scripts/subcycling/compare_hierarchy_backends.py
once gpu_validation_103d8052 is complete; CPU dumps are /tmp/vc-cpu-gpu103 and the
script requires equal source SHAs, lengths and finite values. It reports exact
field differences without claiming production-checkpoint validation.

## Single-A100 corrected hierarchy validation completed

Previous turn made progress with successful compilation and a verified queued
job. Job58202346 ran on nid008216 and completed0:0 in30s. All five corrected Z4c
cases and both helper steps completed0:0; launcher status0. GPU field comparison
uses exact matching source103d8052 and all20 case/timestep dump pairs. Maximum
absolute CPU/GPU difference5.137003171608702e-14; maximum RMS9.141817484268351e-16.
Corrector3 GPU ratios17.0095,16.334; three-level17.015,15.3972; coarse-group17.445,
16.5538; time-dependent17.017,15.4925; CPBC13.6214,10.816 (last RMS5.83163e-14).
All configured gates pass. CPBC ratios are less clean than16 and should not be
represented as exact fourth-order asymptotics at the smallest differences.

Evidence gpu-corrector-103-evidence/ includes logs, Slurm metadata, source/hash and
comparison JSON. Durable raw CPU/GPU fields:
/Users/hz0693/research/collapse/subcycling-results/gpu-validation-103d8052/{cpu,gpu}.
Remote raw gpu_validation_103d8052. Archived validated test binary
binaries/athena.hierarchy-corrector-103d8052 with CMake cache/hash; SHA256
522d2d2954ba5f32f2159911da93454bf743e9a338ec45034f9a053e56410db2.
No isolated build/allocation remains running after successful launcher completion.

This validates small fixed-hierarchy prescribed-gauge vacuum tests on one A100,
not a Brill restart or speedup. Next promote a bounded convergence-controlled
corrector out of diagnostic-only code, integrate the actual common-time global
telegraph history and synchronized driver/AMR/restart path, then perform the
required matched-endpoint production comparison. Full goal remains incomplete;
no production run was modified.

## Bounded convergence-controlled hierarchy corrector

Promoted stage correction and RK-history reconstruction out of diagnostic-only
compilation. RunCorrected requires convergence of both active endpoint fields
and all retained RK histories, with RHS differences weighted by the local dt.
Defaults: minimum3/maximum8 passes, absolute1e-12 and relative1e-10. Six passes
were insufficient at the largest test step (history normalized change2.40558
despite endpoint0.000944729); seven converged. No tolerances were relaxed.
Failure restores all hierarchy values including ghosts; a forced failure test
checks byte equality then successful recovery. External physics side effects
are still the caller's responsibility. Initialization now clears histories and
reports; covered-stage injection validates donors and target/packed layouts.

CPU rebuild succeeded. Six selected CTests pass, including adaptive, rollback,
coarse group, prescribed time-dependent gauge, CPBC, and corrected transport
compiled without diagnostic controls. Additional adaptive three-level combined
time-dependent/CPBC test passes, ratios16.1836 and15.2821; registered as CTest.
Logs archived in adaptive-corrector-evidence. These changes have not yet been
CUDA qualified. The uncorrected asynchronous Z4c diagnostic remains known to
lose order; this is not a claim that the entire test suite passes.

Full production goal remains incomplete. Multiple corrector passes materially
reduce the naive work savings and must be included in the eventual benchmark.
Next: self-consistent common-time global telegraph coefficient, actual driver
restart/AMR/diagnostic integration and single-A100 matched-endpoint validation.
No production files or jobs changed in this turn.

## Physical common-time history and leaf maximum primitive

Previous goal turn made progress in commit83b9eb67. Added EvaluateValue to RK
histories, explicitly distinct from fine RK stage vectors. Uses physical cubic
dense output with interval validation. New RK4PhysicalMaximum selects explicit
physical leaf source IDs and reduces |a*u_i+b*u_j|, supporting Khat+2Theta without
including covered-parent blocks. Validates component/donor lists, rejects
nonfinite selected physical values, and returns zero for an empty leaf batch.
It currently reconstructs all packed components before reduction; fusion is an
optimization opportunity after correctness integration.

CPU predictor test passes with independent prescribed cubic-time solution at
off-stage times, covered parent exclusion, Khat/Theta-style combination, a
maximizing-leaf switch within an interval, empty batches and nonfinite rejection.
Initial test caught Kokkos empty max-reduction identity; explicit zero handles
this correctly. Test log physical-history-test.log. No GPU execution this turn.

This is the per-history primitive, not completed production-gauge integration.
Next assemble common-time maxima across containing histories for every physical
leaf level (including synchronous groups), then couple that coefficient history
into interval iteration and convergence checks. MPI aggregation, production
driver/AMR/restarts, and the requested single-A100 matched-endpoint speedup
remain unverified/incomplete. Production campaign untouched.

## Coupled hierarchy common-time maximum prototype

Previous turn made progress in ee3dc31a. Added HierarchyPhysicalMaximum to combine
physical-leaf maxima over each level's containing RK history at a requested
common time. Select later interval at shared endpoints; reject absent coverage.
Roundoff-sized endpoint differences are tolerated, no physical extrapolation.
This remains single-rank and assumes canonical shared active values.

Fixed histories missing from levels inside the synchronous coarse group: retain
all populated levels, not just the group's highest level. RunCorrected now has
a begin-pass history callback and exposes accepted histories, allowing a gauge
callback to evaluate preceding-pass Khat+2Theta at stage times. Uncorrected Run
invalidates prior accepted-report state. Current test seeds the first pass from
the previous endpoint maximum (initial K=0) and corrects subsequent passes.

Actual Z4c three-level global-gauge test passes with temporal ratios17.0454 and
15.6091; grouped test17.4533 and16.5888. The callback is exercised and nonzero
(max approximately0.000238112), not a prescribed constant. Adaptive/rollback/
group existing tests pass. CPU logs archived. This small weak perturbation does
not establish production-gauge accuracy or performance at strong K, switches,
or long intervals. Endpoint/history convergence currently bounds iteration;
explicit coefficient-history residuals and stronger synchronous-reference tests
remain necessary. No CUDA qualification of these new changes yet.

Production integration remains incomplete: common-time aggregation is local,
main driver/AMR/checkpoint continuation is not using this prototype, source
stability/retry and matched-endpoint single-A100 accuracy/wall-time remain
unverified. No campaign files or jobs changed.

## Explicit gauge-feedback residual and fused maximum

Previous turn progressed with coupled histories. CorrectorReport now includes
feedback_change. RunCorrected accepts an optional normalized feedback residual
callback and requires it <=1 together with endpoint/history convergence.
Invalid/nonfinite callback residuals throw through hierarchy rollback.
HierarchyPhysicalMaximum::Difference checks common-time maxima at all nominal
RK stage times, comparing the actual fields of both interval iterates.

The strong-gauge test increases the initial lapse perturbation from0.001 to0.1;
max|K| reaches approximately0.0237967. Before fusion it passed ratios17.0052 and
16.475, using7..8 passes at the largest step and6..7 at the smallest. This is
still smooth short-time test data, not a production-collapse qualification.

Fused physical interpolation and absolute-maximum reduction reads only the two
requested components from retained histories; avoids allocating/reconstructing
all spacetime components at every query. Independent physical-history tests,
coupled weak/strong gauge and coarse-group tests all pass after fusion. Logs
vc-feedback-strong.log and vc-feedback-fused-test.log. CPU suite155.81s is not a
controlled speedup measurement. Repeated-time query caching remains important;
explicit gauge checks are expensive and must be included in the A100 benchmark.

Still incomplete: source stability and retry control; caller cleanup of derived
physics/gauge state on exceptions; strong-gauge comparison to existing
synchronous production path; main-driver checkpoint/AMR integration and actual
single-A100 matched-endpoint reproduction. No production files/jobs changed.

## Checkpoint entry-path qualification hook

Added test-build-only ATHENA_TEST_SUBCYCLE_INTERVAL_DIR hook after restart ghost
initialization. Requires vacuum single-rank VC Cartoon, production max-domain
telegraph gauge, zero constraint damping/shift eta, rho-axis boundary, a new
output directory, and explicit time/subcycle_probe_dt and subcycle_probe_ratio.
Currently one fixed-hierarchy interval, dt capped at the saved finest dt.
This is an accuracy probe, not full driver subcycling or a speedup mode. Dispatch
uses opt.fd_stencil (not ghost allocation width). It copies populated hierarchy
state, never copies evolved fields back to live u0 or changes mesh time, writes
leaf-ordered fields.bin/topology/probe metadata, then exits before evolution
and final production output. Caller must run in an isolated working directory
because normal restart initialization still runs before this hook.

The common-time coefficient callback caches repeated query times per pass.
Corrector still checks gauge/endpoint/history residuals. Source stability/retry
for intervals larger than saved dt is explicitly not implemented here.

Complete CPU athena build succeeded. A four-block flat VC Cartoon telegraph
checkpoint smoke test succeeded, dt1e-4,3passes, zero residuals. Repeated output
fields have identical SHA d07f7c56a70d0342b81ff47dc896f979277275b9a06eda35874f7334af754b83.
Input checkpoint SHA unchanged0f5ee2a323e1bbe0626b5f33db4ead8c7b84ba59bdddf9dd2e8f0d32cb0eba4c.
Final rebuild/smoke also succeeded after metadata/option guards. Evidence in
checkpoint-probe-evidence; raw restart/fields under /tmp/vc-checkpoint-probe-smoke.
Initial smoke rejected a custom input block; parameters now use allowed time
block. No Perlmutter production files or runs changed.

Next: CUDA compile and isolated Brill checkpoint probe, then actual multi-interval
driver integration with live AMR, safe step limits/retry and diagnostic cadence.
Matched-endpoint single-A100 reproduction/speedup remains incomplete.

## Refined real-checkpoint probe qualification; SSH unavailable

Previous turn made progress with the checkpoint hook. Perlmutter SSH via the
requested control socket reached login but failed authentication(publickey etc),
exit255. No remote source/job state could be inspected or modified. Requested
connection renewal; meaningful local work continued, goal not blocked.

Attempted local Brill fixture from archived A=-0.049625 coefficients but current
CPU build lacks the IrisK generator; failed before evolution. Evidence remains
/tmp/vc-brill-probe-local/create.log. No claim of Brill validation.

Added reproducible scripts/subcycling/test_checkpoint_probe.py. It creates an
isolated seven-leaf/two-level Gaussian lapse-pulse checkpoint with telegraph
max-domain gauge, runs ratio1 and ratio2 probes at three dt, verifies finite
output shape and unchanged source checkpoint hash, and records executable hash.
Input parameters must exist before Athena command-line overrides; corrected
the initial script fixture accordingly.

Initial dt1e-4 tests were roundoff-limited. At dt.002,.001,.0005, ratio2-minus-
ratio1 RMS differences2.98133e-13,9.30743e-15,2.93151e-16; max differences
1.18181e-11,3.66860e-13,1.14680e-14. Approximately32x reduction per halving,
consistent with fifth-order local error over ONE interval. Ratio1 needs3 passes,
ratio2 needs5,5,4. These are both hierarchy-probe paths, not an independent
comparison against production synchronous driver. Full raw output under
/tmp/vc-pulse-checkpoint-qualification-v3; committed pulse-checkpoint-evidence.

Next remains CUDA/late-Brill checkpoint validation when access returns, plus
production synchronous comparison, safe interval selection/retry, live-AMR
driver integration and end-time single-A100 performance reproduction.

## Independent production-driver comparison finds refinement mismatch

Previous turn progressed with real-checkpoint probe tests. Extended the probe
script to run the existing classical-RK4 driver to identical checkpoint_time+dt.
Read the native vacuum-only Z4c restart tail at double precision, verifying its
preceding per-block byte count; reorder by source leaf IDs; compare all50575
active values, including coincident vertices. Verify final history time to1e-14.
The general output tab path was unsuitable (requires1D slices); binary output
is float32. No output format/source changes made.

A substantive mismatch exists despite ratio1/ratio2 agreement. Refined CPBC
ratio1-versus-driver max errors8.65017e-7,4.32251e-7,2.16071e-7 at dt.002,.001,
.0005; RMS5.84544e-9,2.93353e-9,1.46927e-9. This is approximately first-order
local scaling, suggesting different semi-discrete evolution/coupling, not
ordinary RK4 integration error. Thus production reproduction NOT qualified.

Maximum: source leaf1, logical(level2,x1,y0), GammaX component14,j1,i16 active
offsets. Root level1 (verified checkpoint header), so this is a FINE block at
rho2 coarse-fine interface, near physical z=-2 boundary. Earlier commentary
misread level2 as root/coarse and was explicitly corrected. Sommerfeld retains
a mismatch at same point in Khat (4.55740e-7 at dt.002). Uniform CPBC control
agrees to roundoff (max3.89229e-16 at dt.002). This isolates refinement/corner
coupling rather than a general physical-boundary or gauge defect.

Driver-reference-evidence contains CPBC/Sommerfeld/uniform outputs. Raw data
/tmp/vc-pulse-driver-comparison-v4, /tmp/vc-pulse-driver-sommerfeld,
/tmp/vc-pulse-driver-uniform. Script reports differences without passing a
scientific qualification gate just because processes exit0. Next inspect
prepared stage1 ghosts and native coarse-fine ownership near that corner,
fix the discrepancy and rerun independent comparisons before production use.
Perlmutter access renewal pending; goal remains incomplete, no remote changes.

## Stage-transition diagnosis and uncommitted hanging-vertex experiment

Previous turn progressed by isolating refinement mismatch. Added optional
checkpoint-probe state/RHS binary snapshot (ATHENA_TEST_PROBE_FIRST_RHS, optional
ATHENA_TEST_PROBE_RHS_STAGE1..4). CPU probe first prepared stage agrees with
production pre-RHS diagnostic across all leaf active AND ghost values to1e-14.
First complete RHS also agrees on all active leaf values to1e-14.
Stage2 differs at424 active component-values; all are ODD hanging interface
vertices, not even coincident points. Native stage transfer reconstructs these
from coarse data; prior prototype excluded every active fine vertex.

Uncommitted experiment in hierarchy_temporal_ghosts.hpp includes odd active
vertices adjacent to a physical coarse leaf in interpolation targets.
HierarchyRK4 synchronization restores spatial targets at parent/child common
time. This reduces ratio1-versus-driver max error at dt.002 from8.65e-7 to
6.87639e-9; now approximately dt^2 local scaling. HOWEVER both adaptive and
coarse-group temporal-order CTests FAIL (coarse-group ratios4.423,2.046).
Do not call this fix qualified or deploy it. The experimental ownership changes
remain uncommitted intentionally for continued analysis. Production final-stage
algebraic projection is only at last RK stage; simple every-stage projection
is not an explanation. Next reconcile RK-history/corrector treatment of hanging
vertices and synchronization/projection order, then rerun independent driver
and temporal-order gates.

Evidence stage-transition-evidence; raw CSV/snapshots /tmp/vc-first-rhs-compare,
/tmp/vc-rhs-value-compare, /tmp/vc-stage2-probe. Current CPU executable includes
uncommitted experiment and MUST NOT be used for claimed production reproduction.
No live build/job remains: build63216 and test68885 finished; both selected
order gates failed. Full goal incomplete; production untouched.

## Hanging-vertex RHS consistency and remaining asynchronous defect

Previous turn progressed with stage diagnosis. Added RK4DenseBoundary::StageRHS,
whose classical RK updates reproduce all prescribed stage vectors and the
physical cubic endpoint at shifted child starts. Independent RK algebra identity
regression passes. This mathematical primitive is committed separately.

UNCOMMITTED coupling experiment extends temporal interpolation with RHS mode,
adds shape-checked RestrictField, applies coarse-reconstructed hanging RHS before
RK history capture/update, and synchronizes same-level RHS. These changes must
not be described as qualified: adaptive/coarse-group gates still fail. Fully
synchronous three-level control now passes ratios18.8818,16.8263. Asynchronous
ratios13.7463,2.74843; disabling final algebraic projection does not remove loss.
Tightening corrector10x to atol1e-13,rtol1e-11,max16 leaves ratios13.7464,2.74843.
Stronger lapse pulse0.1 gives14.0943,2.9035. Thus error is not a loose-corrector
stopping artifact. A100/production reproduction remains unqualified.

Even tighter100x control failed history convergence at12 passes in interval1;
that log is preserved, not called evolution instability. Test option
--tight-corrector currently means10x tighter/max16. All tested local processes
are terminal; no GPU job launched. Worktree has uncommitted hanging/coupling
experiments in hierarchy_rk4, hierarchy_temporal_ghosts, vertex_parent_states,
vertex_temporal_boundary, rk4_predictor_states and hierarchy_z4c_test.
Next investigate asynchronous prescribed-vertex stage/history consistency and
common-time restriction, retaining the independent driver comparison as a gate.
Production campaign untouched; Perlmutter authentication renewal still pending.

## Reconciled hanging ownership restores temporal order

Previous turn progressed through RHS consistency controls. Two-level and no-KO
controls still failed; polynomial lapse passed17.1849,16.5695. This isolated
initial discrete compatibility: independently initialized Gaussian hanging
values do not satisfy coarse interpolation. Added explicit ReconcileInitialState
before defining the rollback state. Restrict covered levels, reconcile shared
vertices, then impose hanging constraints downward. Z4c test defaults to this
consistent initialization; --unreconciled-initial preserves the failing control.
Checkpoint probe reconciles only its copied hierarchy. Main driver unchanged.

Native odd hanging values and their interpolated RHS are now represented in
stage/history transfers; even coincident values retain fine authority. Generic
RestrictField validates dimensions. Synchronization refreshes all descendant
levels at the common time. Gaussian three-level ratios16.4774,15.9122; grouped
17.1144,16.4719; global-gauge16.4293,15.8907. Rebuilt adaptive/rollback/coarse-group
and non-diagnostic corrected transport tests all PASS. Requested nonexistent
athena_hierarchy_temporal_ghosts_test target caused the combined build command
to end2 after athena and relevant existing targets succeeded; not a source error.

Remaining independent classical driver discrepancy is NOT resolved. At dt.002
ratio1 max6.87639e-9, RMS6.14343e-11; local errors scale roughly dt^2. Ratio1/2
agreement again scales32x under timestep halving. Stage2 comparison now has320
remaining mismatched ghost component-values (top differences near fine
coarse-interface/physical-boundary corner); active hanging mismatch is corrected.
Next inspect physical extrapolation/prolongation ordering and physical-corner
coverage: current temporal plan excludes outside-domain targets and physics
Prepare fills physical ghosts after interpolation; native order differs.

Evidence reconciled-ownership-evidence; raw /tmp/vc-reconciled-driver-comparison
and /tmp/vc-stage2-reconciled. No remote jobs or production changes.
Full single-A100 Brill reproduction/speedup and live-AMR integration incomplete.

## Native intermediate-stage physical ghost refresh resolves reference mismatch

Previous turn progressed by restoring hanging/RK order and isolating remaining
ghost differences. Tested extending coarse interpolation into outer ghost rows
and moving physical preparation before transfers. Both failed independent
comparison/order tests; split physical/axis preparation also failed order.
All experimental prototype changes from this turn were REVERTED to59c177e0.
Rejected patches/logs archived; no unqualified ghost-plan changes remain.

Actual cause found in native z4c_tasks.cpp: accepted final stage refreshes
built-in physical ghosts AFTER prolongation, but intermediate stages did not.
Prolongation changes hanging active values and adjacent fine ghosts, leaving
physical corner values based on the earlier state. A test-only post-prolongation
refresh eliminated the discrepancy. Promoted to native VC Cartoon intermediate
stages only (stage>0 && stage<nexp_stages); final stage already refreshes.
Uses FillBuiltInPhysicalBoundaryGhosts, preserving user callback count and
CC/3D behavior. This is an intentional fix to the isolated branch's synchronous
operator, not a claim of bitwise reproduction of old production evolution.

Full executable rebuild succeeds. Checkpoint ratio1 versus independent native
classical driver max errors3.92590e-16,2.22045e-16,8.70614e-17 at dt.002,.001,.0005
for BOTH CPBC and Sommerfeld. Script now enforces max error<1e-12 for this
synchronous reference. Ratio2 error relative reference shows~32x decrease per
halving (local fifth order). Input checkpoint hash preservation remains checked.
Adaptive/rollback/coarse-group gates all pass. Native synchronous smooth-pulse
static-AMR temporal refinement script passes ratios15.17646,15.23751, on its
radial slice; no broad production-gauge or dynamic-AMR claim.

Evidence post-prolongation-bcs-evidence; raw /tmp/vc-post-prolong-final-cpbc,
/tmp/vc-post-prolong-final-sommerfeld, /tmp/vc-native-boundary-temporal.
No live local process remains. Production Perlmutter source/runs unchanged.
Next CUDA compilation/checkpoint probe after SSH renewal, then actual driver
subcycling/AMR/stability/retry integration and matched-endpoint A100 validation.
The boundary correction must be separated from subcycling in future performance
and scientific comparisons with the historical provisional-collapse run.

## Per-level synchronization interval contract

Added Schedule::ChooseInterval using the schedule's own per-level substep counts.
Every evolved level (including covered predictors) must provide finite positive
spatial and source ceilings in ascending order. CFL belongs in the supplied
spatial ceiling only, matching z4c_newdt/mesh contracts; source safety is already
included by SourceTimestepCeiling. Requested end/synchronization cap is separate.
Tests verify grouped levels, coarse and fine source restrictions, deterministic
limiter reporting, missing/duplicate contracts and overflow-safe unlimited bounds.
CPU schedule target builds and passes. This is selection infrastructure, not yet
native per-level reductions or an integrated evolution/retry driver.
SSH recheck succeeded (login39); isolated remote source is still103d8052. Existing
production job58200306 is running and is not modified. Proceed with isolated CUDA
qualification of accumulated changes before the actual Brill comparison.

## Bounded corrector retries and checkpoint integration

Previous turn made progress with per-level selector c204ee04 and launched the
isolated CUDA build. Build PID316679 remains live on Perlmutter; do not replace
remote source while it compiles. Production campaign unaffected.

Added RunWithRetry: only CorrectorFailure triggers interval halving after the
existing full-state rollback. Bounded by maximum halvings (default8), minimum dt,
and representable fine steps. Other exceptions propagate. Reports accepted dt,
attempt count and total corrector passes; no pretending a shortened interval
reached the requested endpoint. External gauge caches reset through begin_pass.
Checkpoint probe now uses this path and records requested versus accepted dt.
This does NOT yet retry source-stability violations or Kokkos invalid-state aborts.

Actual transport evolution test forces first-attempt feedback nonconvergence,
then checks half-step retry bit-for-bit against direct half-step integration;
checks exhausted retries preserve state and code failures propagate once.
Corrected temporal ratios15.2456,15.6176,15.8076. Fixed CPU rollback test snapshots:
create_mirror_view_and_copy may alias HostSpace state, so use create_mirror plus
explicit deep_copy for independent before/after arrays. Z4c rollback regression
with independent copies passes (temporal ratios16.4846,16.2056).

CPU full executable builds. Actual checkpoint harness passes all six ratio/dt
cases with exactly one attempt; ratio1 agreement with independent native driver
remains at roundoff and ratio2 local error decreases~32x. Harness now explicitly
rejects mismatched requested/accepted endpoints rather than comparing different
times after a shortened retry. Evidence retry-evidence; raw
/tmp/vc-retry-checkpoint-validation and /tmp/vc-retry-helper.log.

Remaining: native per-level limits, repeated/live-AMR driver integration, source
stability checks, CUDA qualification, and full single-A100 matched-endpoint
Brill accuracy/performance proof. No completion claim.

## Native field-based hierarchy timestep ceilings

Previous turn progressed with fffac431 bounded corrector retry and actual probe
integration. This turn extracted native pointwise spatial characteristic limits
into SpatialTimestepPoint, shared by native z4c_newdt and HierarchyPhysics.
The calculation is unchanged; all six checkpoint output field SHA256 hashes
match the preceding retry validation exactly after extraction/integration.

HierarchyPhysics::TimestepLimits now reduces actual populated hierarchy fields
at synchronization, including covered predictor nodes. Source ceiling uses the
same telegraph coefficient and classical-RK4 negative-real stability radius as
the native contract. Scope deliberately enforced: prescribed zero shift, no
constraint/shift/slow-start damping, max-domain telegraph prescription. CFL is
applied only to spatial limits. These are initial synchronized limits, not yet
intra-interval stability monitoring or a live driver. Current reduction scans
nodes per level; optimize device batching after correctness qualification.

Checkpoint probe now selects its interval with these real per-level ceilings,
and writes timestep_limits.csv. Small qualification case: spatial ceilings
0.0080354032574317669 (coarse) and0.0040177016287158834 (fine), source924.403905495243.
All requested small intervals remain unchanged. Full CPU build and real
checkpoint comparisons pass; all six field hashes identical to prior build.
Analytic flat hierarchy tests verify dx/CFL scaling and classical source ceiling,
CFL independence of source ceiling, and inclusion of fast covered predictors.
The complete adaptive Z4c test still passes ratios16.4846,16.2056.
Evidence timestep-evidence; raw /tmp/vc-limits-checkpoint-validation.

CUDA build c204ee04 remains separate and in progress; do not replace its source
until terminal. Full live-AMR repeated evolution, per-stage bound monitoring,
CUDA execution and matched-endpoint single-A100 Brill speedup remain incomplete.

## Repeated synchronized checkpoint evolution

Previous turn progressed with3d36cc7a actual per-level limits. Added optional
<time>/subcycle_probe_duration (default0 preserves single-interval probe).
Positive duration repeatedly selects stable synchronization intervals capped by
subcycle_probe_dt and remaining duration, runs corrected/retried evolution, and
refreshes max-domain K from current synchronized physical leaf fields. The gauge
maximum comes from actual projected/restricted endpoint fields, not stale dense
history or covered nodes. Cached physical leaf IDs allocated once.

Removed saved-finest-dt cap: actual per-level characteristic/source ceilings now
control the interval, permitting coarse intervals larger than saved finest dt.
Records each accepted start/end/dt, attempt/pass counts and endpoint gauge maximum
in intervals.csv, plus per-interval timestep_limits.csv. Output describes actual
elapsed time and aggregate attempts, preserving shortened-retry semantics.
Fixed hierarchy only; no live mesh/time mutation or production output writes.
One-million interval guard bounds this qualification path. Intra-interval source
monitoring, dynamic AMR and production driver wiring remain future work.

CPU full build passes. test_checkpoint_probe.py --intervals3 compares three
synchronized intervals against independent existing native classical restart
steps at each matching endpoint. Ratio1 max errors2.03396e-15,3.40873e-16,
2.22045e-16 for interval caps.002,.001,.0005. Ratio2 RMS differences9.39843e-13,
2.91551e-14,9.09281e-16. These cases shorten total duration with dt, so their~32x
reduction is local-error evidence, NOT fixed-final-time temporal order proof.
Original single-interval compatibility test also passes. Checkpoint hash unchanged.
Evidence multi-interval-evidence; raw /tmp/vc-multi-checkpoint-validation.

CUDA c204ee04 build remains live (PID316679), last observed~65percent. No remote
source replacement while it compiles. Single-A100 Brill endpoint reproduction
and lower end-to-end wall time have not yet been established.

## Fixed-final-time checkpoint convergence qualification

Previous turn progressed with4d3c6038 repeated intervals; its variable duration
error reduction did not prove global temporal order. Added --fixed-duration to
actual checkpoint regression, comparing dt,dt/2,dt/4 at identical endpoint and
checking self-convergence of every active field (including duplicate copies).
Test --fixed-duration.016 --dt.004 passes: synchronous ratio16.2127240283,
factor-two subcycling ratio16.0440794985. Respective successive RMS differences
3.7893542935e-11/2.3372718162e-12 and1.0670421932e-11/6.6506912615e-13.
Independent native restart comparison at synchronization times remains enforced.
This is smooth fixed-hierarchy gauge-pulse evidence, not full live-AMR Brill proof.
Raw /tmp/vc-fixedtime-checkpoint; archived fixed-time-evidence/results.json.

Remote CUDA c204ee04 build verified live PID316679 at~73percent. Prepared and
copied run-gpu-c204ee04.sh, with source/build-status guards and new result directory,
for adaptive, coarse-group, common-time global-gauge, CPBC and rollback cases plus
three helpers on one GPU. No allocation launched yet. Do not mutate remote source
until existing build is terminal. Latest local commits afterc204ee04 still need
CUDA compilation after that qualification. Production untouched; end-to-end
single-A100 reproduction/speedup still incomplete.

## Intra-interval stage stability and rollback

Previous turn made progress with fixed-time temporal qualification92857748.
Inspected native Execute loop: accepted-step time/central updates, stopping,
AMR, output and next timestep occur outside stage tasks. This is the intended
integration location; no live driver change yet.

Added optional hierarchy stage timestep enforcement (enabled in actual probe).
At each RHS, reduce spatial limits only on the stage's currently advanced level
group; asynchronous other levels do not participate. Source coefficient is the
common-time stage gauge. Exceeding a valid spatial/source bound throws typed
IntervalStabilityFailure. RunWithRetry catches this and CorrectorFailure through
a dedicated RetryableIntervalFailure base, preserving full-state rollback and
bounded halving. Invalid geometry/nonfinite states and other code failures still
terminate. Started passes are counted even if RHS aborts before pass completion.
The per-stage reductions currently prioritize correctness; batch/cache optimization
is necessary before interpreting production performance.

Real Z4c regression prescribes a source coefficient that tightens at t.005.
Requested interval.01 rejects, .005 rejects, .0025 accepts; accepted result matches
direct .0025 evolution bit-for-bit. Added hierarchy_z4c_stage_limits CTest entry.
Existing retry and adaptive tests pass. Actual fixed-duration checkpoint test
also passes with checks enabled (dt.004,.002,.001 to duration.016); no change to
fourth-order convergence. Evidence stage-limit-evidence; raw
/tmp/vc-stage-fixedtime-checkpoint, /tmp/vc-stage-limits-test.log.

CUDA c204ee04 build still being monitored, source untouched. Live-AMR integration
and actual single-A100 Brill reproduction/lower wallclock remain incomplete.

## Avoid whole-hierarchy scans in every level's timestep reduction

Previous turn progressed with21a8e720 intra-stage stability rollback. CUDA build
c204ee04 remains live PID316679, last visible completed percentage96; no duplicate
build/allocation launched. Its prepared A100 script remains ready, not submitted.

Replaced each level's all-node scan with cached device block IDs grouped by level.
Now a stage reduction launches exactly its level's point count rather than all
hierarchy points with a per-point level rejection. Synchronized all-level checks
visit every node once in total rather than once per level. Cache constructed once
per fixed-topology physics instance, explicitly recreated after regrid. Cached
classical RK4 negative-real stability radius avoids repeated host root solving.
No claim of measured wallclock gain yet.

CPU full build, real Z4c source-tightening rollback test and fixed-duration actual
checkpoint comparisons pass. All SIX field hashes exactly match preceding
stage-limit build. Fixed-time ratios remain16.2127240283(sync),16.0440794985(ratio2).
Evidence timestep-batching-evidence; raw /tmp/vc-limit-batching-checkpoint.
Still need live driver/AMR integration, CUDA execution and full matched-endpoint
single-A100 provisional-collapse accuracy and wallclock comparison.

## Reusable synchronized evolution owner; CUDA build complete

Previous turn progressed with241e6db7 level-batched stability reductions. Extracted
SynchronizedHierarchyEvolution from checkpoint analysis and made the probe consume
it. Owns an independent hierarchy copy, predictor/geometry state, physical-leaf
IDs, stage physics, common-time gauge cache and interval controls. Advance returns
actual accepted interval and per-level limits; object tracks its synchronized
time. Explicit CopyAcceptedLeavesTo provides the next driver integration boundary;
caller must rebuild native ghosts/ADM before diagnostics. Object is noncopyable
(callbacks bind this) and must be reconstructed after AMR. Probe still never exports
to production fields or advances live mesh time.

CPU full build and fixed-time real-checkpoint comparisons pass. All SIX field
hashes exactly match preceding timestep-batched implementation. Evidence
evolution-owner-evidence; raw /tmp/vc-owner-checkpoint. This is consumed reusable
infrastructure, not yet live-AMR driver wiring or proof of performance.

Perlmutter build c204ee04 completed status0. Launched prepared single-A10080GB
shared_interactive qualification allocation58207663 on nid008304 (15minute cap).
Last authoritative squeue state RUNNING; local observing session99491. Remote
allocation log allocation-c204ee04.log; result directory gpu_validation_c204ee04.
Do not replace remote source until this job terminates. It qualifies c204ee04,
not subsequent retry/timestep/owner changes. Production jobs left untouched.
Goal remains incomplete until live-AMR and actual matched-endpoint single-A100
Brill scientific reproduction plus lower measured wall time succeed.

GPU qualification58207663 terminal FAILED134 after47seconds. All five actual
Z4c evolution cases passed: adaptive, coarse-group, global-gauge, global-gauge
CPBC, rollback; hierarchy RK helper also passed. vertex_temporal_boundary_test
then threw generic "temporal boundary regression" (step.6 aborted); schedule test
was not reached. Archived logs gpu-c204-evidence. Overall CUDA qualification is
NOT passed. Added assertion line diagnostics to local temporal boundary helper;
next rebuild/relaunch that small target to locate failure before any broad claim.
Remote source remains clean c204ee04, no build/job live from this qualification.
Observer99491 terminal134. Production unchanged.

## Diagnose CUDA temporal-boundary test failure

Previous turn progressed with reusable owner365daf05 and GPU evidence f18a1a1a.
Targeted assertion-line GPU job58207830 failed at line137: the test expected
odd active hanging vertices to remain at the sentinel even though native transfer
now reconstructs them. With independent CPU snapshots, CPU fails at exactly the
same assertion. create_mirror_view had aliased live CPU arrays and masked this.

Fixed test initial/scratch mirrors to allocate independently, and made expected
scatter targets independently include active odd vertices on coarse-fine patch
edges, excluding physical faces and same-level ghost copies. Numerical tolerances
UNCHANGED. CPU helper now passes axis/outer/interior scatter and all temporal
orders. Earlier diagnostic attempt also caught ghost coordinates along the patch
edge, corrected by requiring active destination for hanging ownership. No solver
or interpolation code change required for this failure.

Remote c204ee04 has only this test-file patch. Targeted rebuild plus one-A100 helper
and scheduler rerun observing session7913; build last verified in progress. Files
boundary-fixed-build.log, boundary-fixed-gpu.log (created after compile),
boundary-fixed.patch and boundary-fixed.sha256 in isolated remote root.
No source replacement until test terminal. Newer local evolution changes still
need their own CUDA build and Brill tests. Overall goal incomplete.

Targeted GPU rerun58207979 on nid008285 completed successfully (observing7913 exit0).
Temporal boundary and previously skipped schedule helper both PASS; interpolation
ratios16.0819,16.0405,16.0201 at all three orders, matching CPU. GPU log archived.
Together with58207663 evolution cases this resolves c204ee04 qualification failure
without changing interpolation or tolerances. Does not qualify later evolution code.
Restored only the archived test patch on isolated remote, fetched bundle and
switched cleanly to codex/vc-evolution-befad6a4. Started build-befad6a4.sh, log
build-befad6a4.log/status in isolated root; observer96618. Do not replace source
until this newer CUDA build and subsequent tests terminate. Goal incomplete.

## Native driver opt-in integration

Previous turn progressed by resolving GPU test oracle and launching newer CUDA
build. Added experimental live driver entry via subcycle_max_ratio>0, explicit
subcycle_cycle_unit=synchronization, and subcycle_interval_cap. Defaults disabled;
validates single-rank vacuum VC Cartoon classical RK4. Persistent hierarchy owner
advances intervals and exports leaves; native accepted-state boundary rebuild,
ADM, timestep, tracking/horizon and final-stage diagnostics then run before
existing stopping/AMR/output loop. Import native finalized leaves back to owner
without reallocating, reconcile covered/hanging values, preserving state continuity.
Owner resets after actual topology change. Physical-time output/endpoints cap dt.

Native ncycle and cycle-based output/AMR explicitly mean synchronization cycles.
Track accepted leaf-block steps separately (including fine substeps, excluding
covered predictors/corrector repetitions) in counter and subcycling_intervals.csv.
No claim this counter measures all computational work. AMR is wired but actual
refine/derefine events remain to be qualified.

CPU full build passes. New test_live_subcycling.py generates isolated actual
checkpoint, compares ratios1/2 live versus frozen evolution over4 intervals and
compares split/restarted live evolution to uninterrupted run. Max errors1.62549e-15
and1.06057e-15; restart comparison EXACT for both. Checks final history time,
continuous intervals, leaf-block substep counts, checkpoint input hash unchanged.
Evidence live-driver-evidence; raw /tmp/vc-live-guard-validation. Test fixture uses
static hierarchy; no dynamic AMR or production-gauge strong-field proof.

Remote CUDA befad6a4 build verified live PID505749 (~36percent), does not include
this driver entry. Do not replace its source while building. Full actual single-
A100 provisional-collapse endpoint reproduction and lower wallclock outstanding.

## Actual live AMR-event and restart tests

Previous turn progressed with71694609 native-driver entry and fixed-hierarchy
continuation checks. Added reproducible test_live_amr.py using existing test pgen's
deterministic tagging (actual AMR transfers, NOT replay). Smooth lapse0.1 pulse,
physical CPBC, real native outputs/restarts and global telegraph gauge.
Both ratio1 and ratio2 pass refinement/coarsening sequence4->10->4->4 blocks.
Splitting immediately after refinement and continuing through coarsening gives
EXACT active-field equality with uninterrupted run at t.008.

--mixed exercises simultaneous coarsening/refinement,4->7->7->7 blocks. Native
log confirms6blocks created and3deleted, not just unchanged block count. Splitting
immediately after the mixed event gives EXACT equality for both ratios. This
checks topology invalidation when block count is unchanged. Checks finite fields,
checkpoint input SHA preservation, expected event counts, and final history time.
Fixture is bounded/deterministically tagged: not a physical error-based AMR
convergence claim or proof of near-critical Brill stability/performance.

Evidence live-amr-evidence; raw /tmp/vc-live-amr-final and
/tmp/vc-live-amr-mixed-final. Commands:
python scripts/subcycling/test_live_amr.py /path/to/athena /new/output
python scripts/subcycling/test_live_amr.py /path/to/athena /new/mixed-output --mixed

Remote CUDA befad6a4 build verified still live PID505749 (~60percent); no source
change or duplicate submission. That snapshot predates native-driver integration.
Actual A100 Brill endpoint reproduction and lower wallclock still outstanding.

## Recover actual endpoint and prepare isolated Brill comparison harness

Previous turn progressed with actual AMR event/restart tests79052119. Re-read real
Perlmutter checkpoint and final campaign history. Checkpoint t62.20556032298022,
cycle133489,nmb8924,dt2.422989058508578e-6; SHA independently verified unchanged
4b4dc1f576fe2b88d00c72de17bc413bf5bfa0ccbfce4ef36e2980a6cd979b4b.
Final history t62.205994062314062,cycle133668,minLapse.017361930148278765,
maxAbsKret259724.66624038288,C-norm2 131.3371573622191. These reproduce the
recorded provisional run; large constraints are not evidence of trustworthy
physical collapse. Actual input: spatial_order4, transfer6, extrapolation2,
telegraph_tau=kappa=.01,zero shift/damping,chiTE.001,CFL.25,capacity24000.

Added prepare_brill_endpoint.py and run_brill_endpoint.py. Preparation copies
checkpoint-required90925809-byte AMR history prefix into each fresh case and
redirects the embedded absolute production history path; never append to production.
Pins checkpoint/executable/input hashes. Two cases: current classical synchronous
and subcycled, same endpoint and requested physical output cadence (eight samples;
native sync may overshoot sample times by a finest step). Records explicit source
compatibility for copied history. Executor runs serially on one allocated GPU,
exclusive output files, per-case executable verification, fail-fast on errors or
unexpected termination, requires subcycling CSV for subcycled case. Does not mark
scientific reproduction successful merely on exit0. No simulations launched by
these scripts yet; syntax checks pass; actual remote checkpoint parser schema
verified. Must prepare against a pinned executable containing live integration.
This benchmark covers the late checkpoint-to-history-end segment, not a claim of
replaying or accelerating the entire original eight-hour evolution.

Remote befad6a4 CUDA build still live PID505749 (~87percent), predates live-driver
code; no replacement/duplicate run. Strong-field science/performance comparison
remains outstanding, as does newer CUDA native-driver qualification.

### Production spatial-order fixture qualification
Added --production-orders to checkpoint/live/AMR harnesses. Confirmed generated inputs use spatial_order=4 and extrap_order=2. CPU live/frozen differences <1.2e-15; split restarts exact at ratios 1/2, including real refine/coarsen and simultaneous mixed events. Results in production-order-evidence. Remote befad6a4 build PID505749 remains live, now compiling hierarchy test after main targets; no restart or source replacement. Actual Brill reproduction and speedup remain unverified.

### CUDA qualification befad6a4 passed; native driver build started
Build befad6a4 completed status0. One-A100 shared_interactive job58208872 on nid008237 completed exit0 in51seconds: adaptive, coarse-group, global-gauge, CPBC, rollback, stage-limits, hierarchy RK, temporal boundary and schedule all passed. Archived logs/hashes/Slurm metadata in gpu-befad-evidence. Preserved tested athena/helper binaries remotely in pinned-befad6a4. Updated clean isolated remote source to38c1fab4 after job completion; native live-driver CUDA build started as PID710534, build-38c1fab4.log/status. Initial bundle fetch named branch failed because bundle exports HEAD; corrected fetch HEAD succeeded before starting build. No production files or jobs changed. Native-driver GPU tests and actual Brill reproduction/speedup still pending.

### Live GPU harness and endpoint report preparation
Prepared run_live_gpu_38c1fab4.sh: pinned source/build checks, isolated output, separate srun step per executable invocation, fixed/live restart plus real/mixed AMR at production spatial orders, before/after executable hashes. Uploaded only; awaiting build PID710534 (verified live at2m38s,26percent). Added compare_brill_endpoint.py: rejects failed or unmatched-endpoint histories, reports native-sample diagnostic curves and endpoint differences, separates measured segment wall speedup from scientific reproduction. Parser exercised on61 actual archived scaling history samples; full endpoint analysis awaits real experiments. No fake benchmark outputs generated.

### Production gauge and probe endpoint regression
Added production-gauge option tau=kappa=.01 to checkpoint/live harnesses. Fixed near-zero probe leftover interval from floating-point endpoint accumulation; spatial/source ceilings still enforced. Live/frozen/restart pass; fixed-time temporal ratios16.07988/16.01309 atdt=.002,.001,.0005. Preserved smaller-step roundoff-floor failure and larger-step native agreement failure4.97e-12 atdt=.004 (still needs investigation). Evidence in production-gauge-evidence. Remote38c1fab4 build remains active; no remote source replacement while compiling.

### Resolve production-gauge synchronous discrepancy
Actual RK-stage physical-leaf maximum now used when the group includes all levels; physical-time interpolation cannot represent distinctRK2/RK3 stage vectors at the same nominaltime. Stage source/spatial checks consume the same maximum. Larger-step native disagreement reduced4.97e-12 to6.78e-16 with unchanged test threshold. Temporal ratios16.1261/16.0903; live/frozen, restart, real/mixed AMR and stage-limit retry pass CPU. Evidence in stage-gauge-evidence resolves earlier larger-step failure. CUDA build38c1fab4 stilllive PID710534 at53percent; it predates this fix and probe endpoint fix. Need GPU qualification of these changes before production benchmarking.

### Stage incremental CUDA qualification without replacing live build
Verified original build38c1fab4 PID710534 remains live. Uploaded884f5170 source/script delta and per-file SHA256 manifest, guarded build-live-884f5170.sh and run-live-gpu-884f5170.sh; none executed yet. After original build terminates, apply exact delta and incrementally rebuild athena/helper, avoiding unrelated recompilation. Embedded SHA stays38c1fab4; provenance must explicitly include884f5170 patch and per-file hashes. Updated GPU harness covers production-gauge live/restart, production spatial-order real/mixed AMR and the larger-step temporal regression. Do not run older clean-source harness against patchedsource. Production files/jobs unchanged.

### Separate integrator control and historical endpoint comparison
Brill preparation now creates legacy synchronous rk4, classical synchronous, and classical subcycled cases on the same executable. Manifest parses actual archived final history labels/values; report emits differences against archived production as well as classical baseline, plus speedups against both integrators. This is still a late segment, not whole-run qualification; same-current-source legacy mode does not recreate historical source exactly. Syntax checked; actual experiments pending. Keep these analysis scripts outside the hashed884f5170 patched source when deploying to avoid altering GPU source provenance. CUDA build710534 stilllive.

### Verify Brill preparation against actual checkpoint
Deployed endpoint tools e206f34f separately from pinned source. Preparation-only run succeeded with legacy/classical/subcycled inputs, fresh checkpoint hash matching archivedtarget, exact90925809-byte AMR prefixes, and isolated amr_history_file paths. Evidence manifest in brill-preparation-evidence. Used pinnedbefad executable only for preparation, must not run this manifest (no live driver); regenerate against qualified binary. Initial hand-transcribed hash assertion failed due typo; programmatic comparison with recovered-target.json passes. Actual target t62.20599406231406, lapse.017361930148278765, Kretschmann259724.66624038288. No evolution launched.

### Native CUDA base build finished; corrected incremental build active
Verified38c1fab4 build terminated with status0 (PID710534 gone). Ran guarded build-live-884f5170.sh: applied exact staged patch to isolated source and all7 changed-file hashes passed. Incremental build PID880731 verified live compiling driver.cpp; log build-live-884f5170.log, status eventually build-live-884f5170/status. Embedded sourceSHA remains38c1fab4 plus archived patch matching884f5170. Next run run-live-gpu-884f5170.sh only after successful completion. No GPU simulation launched this turn; prior preparation-check manifest must not be used for actual benchmark.

### Live CUDA qualification passed; actual Brill benchmark launched
Incremental build PID880731 finished status0. Job58210242 oneA100 nid008553 passed all live/restart, real/mixed AMR and production-gauge fixed-time temporal tests, exit0. Archived logs in live-gpu-884-evidence; binary hash verified unchanged. Pinned executable at pinned-live-884f5170/athena with patch/file provenance. Regenerated actual three-case manifest at brill-endpoint-884f5170 using endpoint-tools-e206f34f (not preparation-only manifest). Launched45minute shared_interactive oneA10080GB job58210606 nid008325, session38755. Verified live at48seconds reading actual8924-block checkpoint in legacy_sync. Subsequent classical_sync/subcycled run serially. Job script run-brill-endpoint-884f5170.sh; per-case stdout/stderr/result.json, runner.log and job-exit-status. No production outputs modified. Short-tail comparison and longer reproduction/performance validation still pending.

### Brill attempt invalidated by output cadence; fix qualified locally
Stopped owned job58210606 (confirmed CANCELLED after4m45s) when legacy_sync produced no histories/checkpoints. Parser gives any dcycle entry precedence, so preparation dcycle0 plus dt disabled outputs. Legacy completed179steps in163.807s including startup, but no scientific acceptance possible. Classical interrupted143 intentionally; not evolution failure. Added explicit output cadence=auto/time/cycle retaining legacy auto precedence; time overrides inherited dcycle. Preparation now selects time. New restart regression passes: auto disabled, explicit time writes history plus4checkpoints at intended times. CPU build succeeded. Uploaded output-only patch; GPU build/rerun pending. Preserve originalattempt as invalid, rerun fresh directory with corrected output cadence and same settings.
