# Broader gauge-weight comparison: N16 then N24 through 3M

The user authorized a matching N16/N24 test of a smoother, better-resolved gauge
blend, and explicitly limited this comparison to3M. Both cases start from the
same original generator inputs as their donor runs. No restart or direct-variable
continuation is used. This is a separate equation experiment, not a replacement
for the frozen N32 campaign.

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

Plan: campaigns/n16-n24-smooth-gauge-v1/plan.json. It runs N16 then N24 in one
allocation, one node/12 MPI ranks/GPU tiles, CompactBinaryMerger/debug. Native
walltime reserve180s and worker budget3200s are retained. The controller restricts
this phase to fresh N16 then N24 inputs, exact3M targets and the matched dt and
diagnostic settings. Preserve the5GB task footprint cap, current N32 inputs,
and all source/binary provenance; archive and verify obsolete data before purge.
Do not overlap debug allocations or silently modify the running N32 experiment.

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

State file: remote smooth-gauge-comparison-state.json. Its current status is
ready_waiting_for_debug_slot. The original N32 job8992832 occupies the allocation;
let it finish normally. The existing hourly monitor gives this comparison
priority before any original N32 continuation. Do not submit twice: inspect the
new plan's submission.json/submission-pending.json and the root lock first.

After original N32 becomes terminal, collect/verify its evidence under its frozen
protocol and release only its matching terminal lock. Check ALL user debug jobs.
When none remain, submit the bound new plan with
source-smooth-gauge-v1/tst/hyperbolic_shift_3d/aurora/control.py, then update the
smooth-gauge state with its job id. The controller rechecks hashes and storage.
Never resume an original failed N32 checkpoint. Do not discard original N32
results or mix its checkpoint schema with this experiment.

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
The original N32 campaign may then resume only if its latest original-schema
checkpoint passed the original admission protocol; preserve its existing monitor
until all previously authorized work is handled.
