# Broader gauge-weight comparison: N16 then N24 through 3M

The user authorized a matching N16/N24 test of a smoother, better-resolved gauge
blend, and explicitly limited this comparison to3M. Both cases start from the
same original generator inputs as their donor runs. No restart or direct-variable
continuation is used. This is a separate equation experiment. The original N24 and N32 campaigns
are terminal numerical failures and will not be resumed.

Only the telegrapher coupling weight w and its analytic derivative change.
The original w=S8((0.8-r)/0.24), S8(t)=t^8/(t^8+(1-t)^8), is replaced by
S9(t)=126t^5-420t^6+540t^7-315t^8+70t^9, with exact zero/one extensions.
Its derivative is630t^4(1-t)^4; w_r is this times -1/0.24. Symmetry is used
when evaluating the upper half to avoid endpoint cancellation. This degree-nine
polynomial has four continuous derivatives at the endpoints. The original
rational blend has more endpoint differentiability but is sharply concentrated;
the intended improvement here is a broader resolved transition, not higher
formal differentiability.

The interval0.56–0.8 and midpoint0.68 remain fixed. The separate harmonic collar
on0.8–0.9 is unchanged, so w times collar remains identically zero. Omega, the
reference geometry, initial slices, bulk coefficients, initial shift, mesh,
angular dissipation, penalties and timestep are unchanged. This is a gauge
coupling change, not a remapping of the spacetime slices at initialization.

Use n16 dt=.0005 then n24 dt=.0003333333333333333, each to coordinate time1=3M,
with the same log-chi/log-lapse RK and KO scheme as the original0–3M comparison.
The physical lapse floor remains1e-10; no new chi floor is introduced into DT1.
Preserve horizon lmax12/ntheta24/tolerance.001 and diagnostic interval.05.
Complete candidate horizons are retained as provisional exterior cutoffs while
preserving the original acceptance flags. This matches the earlier diagnostic
replay and prevents missing exterior histories from finder-tolerance failures.

The separate target is hyperbolic_shift_3d_mpi_smooth_gauge. Its checkpoint magic
HYP3D_MPI_GW1 binds the fixed polynomial profile and rejects DT1/AC1 files. The
header stays152 bytes with the original DT1 configuration blocks. The existing
targets default to the original weight. A matching diagnostic-only checkpoint
reader is hyperbolic_shift_3d_checkpoint_constraints_smooth.

Current plan: campaigns/n16-n24-smooth-gauge-v1-attempt02/plan.json.
It runs N16 then N24 in one
allocation, one node/12 MPI ranks/GPU tiles, CompactBinaryMerger/debug. Native
walltime reserve180s and worker budget3200s are retained. The controller restricts
this phase to fresh N16 then N24 inputs, exact3M targets and the matched dt and
diagnostic settings. Preserve the5GB task footprint cap and all source/binary
provenance; archive and verify obsolete data before purge. Do not overlap debug
allocations or resume either original failed fine-grid experiment.

Compare time histories with the original candidate-diagnostic replay and radial
constraint integrals at3M using the same r=0.0753 cutoff and bands. Record horizon
quality, null closure, lapse/floor diagnostics and any numerical failure. Do not
infer convergence from two resolutions or from a change of gauge alone. Stop
after the matched3M comparison; no20M smooth-weight continuation is authorized.

## Prepared Aurora binding

Scientific source commit b27f76f37; manifest source-smooth-gauge-v1-manifest.json
SHA2567308e7e5c3f54d960c0b4bae9b6cf92127a9aa2d66d7f46a5c88c0baaad64fdb.
Executable build-sycl-mpi-smooth-gauge-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_smooth_gauge
SHA25653eda0563a1b48846ac7372422906c35569f6ed685b75bb1dc980ccea83567f1.
The1180 non-Kokkos source files were compared to their exact Git blobs; every
Kokkos file copied from the prior Aurora source is also bound by the manifest.
The initial configure failed because the first source archive omitted the root
config.hpp.in template; it was copied from the same commit before rebuilding.
Both configure records and manifests are preserved. No numerical run used the
incomplete source snapshot.

Both input files have the same hashes as the originals, recorded in
original-inputs.json and submission/binding.json. Local checks passed: analytic
weight derivative vs fourth-order finite differences (maximum error1.14e-10),
monotonicity, normalization, endpoint/collar conditions, a two-step puncture
run with null closure~1e-15, successful GW1 restart, and rejection of a DT1
checkpoint. The controller/preflight suite passed112 tests, and the added
bounded-policy test passed separately. These are preparation checks, not3M
stability results.

State file: remote smooth-gauge-comparison-state.json. Inspect its current plan,
saved submission and root lock before scheduler actions. The original N32 run
failed at5.92425M; its evidence was verified locally and its terminal lock released.
The first smooth job8993278 failed at startup on the storage reservation guard,
before any evolution. The retry uses a fresh attempt02 plan with identical
scientific source, executable, inputs and parameters; only output paths change.

On terminal completion, collect using the new controller. Copy the entire new
campaign directory locally to
output/hyperbolic-core-shell/smooth-gauge-v1/aurora, excluding no scientific
artifacts. Run verify_results.py on that local directory to validate worker
hashes, header/configuration, admission, time, steps, dimensions and finite
payload. Copy/hash-verify before any remote purge. Release only the matching
terminal root lock. If either case failed or checkpointed instead, preserve its
evidence and diagnose; do not claim a completed comparison or blindly rerun it.

After both pass, run analyze_checkpoints.py then compare.py from this directory.
The analysis script picks one radius outside all four mean horizon candidates,
then evaluates old and new checkpoints with their matching diagnostic binaries.
It records hashes before and after and requires all states at exactly3M. Plot
both history and radial comparisons, inspect figures, report measured changes,
commit/push evidence and mark the comparison complete. Stop this experiment at3M.
Delete the hourly monitor once both bounded cases and their comparison have
been fully handled. Neither original failed fine-grid state may be resumed.


## Submission

Job8993278 was submitted after the original N32 allocation terminated, its
failure evidence was hash-verified locally, its large failed outputs were purged,
and its matching terminal lock was released. All-user debug query was empty.
That first submission subsequently failed before evolution; see the startup
retry record below. Inspect the current state and saved submission before any
scheduler action. The hourly monitor aurora-n24-n32-convergence-to20m now tracks this
bounded comparison, with the display name Aurora smoother gauge N16 N24 to3M.
Original N32 failed at5.92425M near the puncture; neither failed fine-grid
original state will be resumed. Both new cases still stop at3M.

## Interpretation of the shared weight

The existing w is also consumed by HyperbolicDamping: delta =
w*(inner_z4*reference_alpha - kappa1*alpha). Therefore changing w changes both
the telegrapher coupling and the spatial blend of Z4 constraint damping. The
constant damping parameters remain fixed, but the resulting spatial damping
coefficient does not. Any reduction in constraint errors cannot be attributed
solely to better numerical resolution of gauge gradients without a further
experiment separating these two uses. No change to the submitted experiment
is made by this clarification.

There is no equation requirement for the original narrow central transition.
For a broader-support follow-up, retain w=1 in an inner neighborhood and w=0
by r=0.8, preserving its disjoint support from the harmonic collar on0.8–0.9.
For example, the same polynomial on0.4–0.8 would have10–90% width0.159225,
approximately8.49/12.74 middle-shell radial spacings at N16/N24. This is a
possible future experiment, not an adopted parameter change or a demonstrated
safe width. Gauge changes can propagate inward even with the local inner
coefficients unchanged. The submitted experiment keeps0.56–0.8 for a controlled
comparison of profile shape.

## Startup storage failure and unchanged retry

Job8993278 exited1 before producing any worker status or native evolution output.
The compute-node guard measured4,169,152,022 bytes, which exceeded the cap after
reserving650,000,000 bytes for outputs and245,000,000 for atomic checkpoints
(headroom -64,152,022 bytes). Submission had measured3,617,957,850 bytes.
Subsequent login-node scans changed from3,618,106,606 to2,914,779,136 bytes with
no task file deletion between them. The accounting uses max(logical size,
allocated blocks); the N32 checkpoint and initial-data block counts were observed
to change from roughly twice logical size to once logical size. The underlying
filesystem reason for that variation is unconfirmed. This is a startup storage
failure, not evidence of numerical instability in the smoother profile.

The obsolete original N32 3M DT1 checkpoint and initial text were SHA256/length
verified against existing local copies, then removed remotely. The receipt records
both originals and retained local paths. Afterwards the measured task footprint
was2,211,454,976 bytes. The fresh attempt02 binding left1,893,527,032 bytes of
headroom after the same895MB of output/atomic reservations. The cap and reserves
were not relaxed. The original failed submission and its terminal lock release
are preserved in startup-failure/; retry provenance is in retry-submission/.

Attempt02 was submitted as job8993603 after an empty all-user debug query.
Its bound plan SHA256 is1a08591ccb110c5a1f802efe5cf441af1ba720d02517199b1082dfddf311c488.
The hourly monitor tracks this job and retains the3M stopping requirement.

## Verified results: both grids completed 3M

Job8993603 completed both fresh runs and exited0. The local verifier checked
source/executable/plan bindings, worker artifact hashes and lengths, both admitted
GW1 headers/configurations, exact3M times and finite payloads. The diagnostic
readers left all four old/new checkpoints unchanged; execution records and
binary hashes are in radial/execution.json. Both runs stopped at3M as requested.
N16 took418.0 seconds and N24 took687.1 seconds for the native commands including
launch; scheduler walltime including setup was20m57s.

At3M the same spherical cutoff r>0.0753 is used for all four states. These are
normalized compactified-coordinate volume RMS norms, not proper-volume norms.

| Grid | Constraint | Original | Smooth | Reduction |
| --- | --- | ---: | ---: | ---: |
| N16 | Hamiltonian | 0.022100108 | 0.017641948 | 20.2% |
| N16 | Radial momentum | 0.047862431 | 0.037134104 | 22.4% |
| N16 | Conformal invariant momentum | 0.035359183 | 0.022126784 | 37.4% |
| N16 | Theta | 0.0005786923 | 0.00034147275 | 41.0% |
| N24 | Hamiltonian | 0.021932896 | 0.011750837 | 46.4% |
| N24 | Radial momentum | 0.048151125 | 0.023819271 | 50.5% |
| N24 | Conformal invariant momentum | 0.042231721 | 0.018102214 | 57.1% |
| N24 | Theta | 0.00057136842 | 0.00022150662 | 61.2% |

The smooth N24/N16 norm ratios are0.666 for H,0.641 for radial momentum and0.818
for invariant momentum. Original ratios were0.992,1.006 and1.194 respectively.
Thus the endpoint refinement trend improves, but two grids do not establish
an asymptotic convergence order. The time histories are not uniformly improved:
the smooth profile produces larger early H/momentum transients before its
later reduction. History curves use each run's own provisional candidate cutoff;
the endpoint table and radial plots use the common fixed cutoff. Connecting
sampled points does not fill missing diagnostic values.

In the r=0.6–0.8 shell, H/radial-momentum band RMS falls by50.2%/52.2% at N16
and67.5%/70.4% at N24. In the previously dominant r=0.675–0.75 peak, the N24
reductions are86.8%/87.0%. The N24 share of exterior H squared integral in that
peak drops from78.8% to4.8%. The core-shell r=0.16–0.18 band changes by less than
1% in H/radial momentum at either resolution, consistent with the main benefit
being in the outer gauge transition. This is evidence for the shared weight as
an important contributor, not proof that gauge gradients alone caused the error:
the same weight changes Z4 damping as explained above.

A remaining shoulder around r=0.4–0.675 is nearly unchanged between N16 and N24
in the smooth run; its H/radial-momentum RMS is larger than in the original run.
This limits claims of restored convergence, even though the old sharp outer peak
is strongly reduced. No longer evolution or broader-support profile was tested.

The largest new surface/ghost null-closure residual is1.5544e-15. Neither run
activated the lapse floor; no chi floor was enabled. Final chi/lapse minima were
1.2143e-4/9.3720e-3 at N16 and2.5695e-5/4.6572e-3 at N24. Complete horizon
candidates were retained as requested, but neither final candidate met the.001
finder tolerance: expansion RMS was.0013764/.0014280. Their radii were
.07518875/.07502866. These quality flags remain in the reports; the norm
comparison is conditional on using those candidate cutoffs.

Plots: constraint-history-comparison.png and radial-comparison.png (PDF copies
also saved). Exact endpoint and band results: comparison.json. Raw histories and
native reports are included here; full checkpoints remain locally under
output/hyperbolic-core-shell/smooth-gauge-v1/aurora. Only after local hash/length
verification were remote checkpoint copies purged. The task footprint after
purge was2.407GB; allocation lock release and purge receipts are in completion/.
