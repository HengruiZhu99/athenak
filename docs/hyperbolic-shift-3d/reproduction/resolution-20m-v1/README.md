# N24/N32 convergence campaign through 20M

The user authorized N24 then N32, one Aurora debug node at a time, with hourly
monitoring and repeated healthy-walltime continuation. The target is 20M for each
resolution. This extends the completed N16 experiment; it does not retroactively
claim the older generic 100M stability admission. The explicit
`direct_variable_resolution` campaign policy permits only this bounded N24/N32
protocol and marks results ineligible for the old automatic stability admission.

## Matched numerical procedure

Each resolution starts from newly sampled isotropic wormhole initial data, uses
the original log-chi/log-lapse DT1 scheme to coordinate t=1 (3M), then explicitly
converts compensation and evolves direct chi/direct lapse AC1 to t=20/3 (20M).
This follows the successful N16 history, whose direct-variable continuation also
began at 3M. No N16 solution is interpolated to the finer grids. N16 had an early
restart and final-step rounding that need not be bitwise reproduced at other
resolutions. A later all-direct-from-initial-data study would be a separate test.

| Grid label | Cartesian core spacing | dt | Evolved volume cells | AC1 checkpoint bytes |
|---|---:|---:|---:|---:|
| N16 reference | .01125 | .0005 | 159200 | 79725224 |
| N24 | .0075 | .0003333333333333333 | 469152 | 234413736 |
| N32 | .005625 | .00025 | 1039456 | 518617768 |

The common dt/dx is .0444444444444444. n is the per-block label; the core contains
2n cells per axis. Core half-width .18, spherical shell inner radius .16, overlap
3 cells, angular epsilon2, original penalty scale1, RK4, and all PDE coefficients
remain fixed. Keep horizon lmax12/ntheta24/tolerance .001 and diagnostic interval
.05 coordinate time. AC1 retains physical lapse floor1e-10, positive chi floor
1e-8 at r<.03, and recorded finite nonpositive chi rescue everywhere. Finite
negative candidates are counted before clipping and do not stop a trial.

The original SYCL/MPI executables are reused without rebuilding or modifying:

- DT1: `build-sycl-mpi-timestep-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_radial_batch_timestep_experiment`, SHA256 `6fa27b3a479ee25adb44d4025cbadfbe6a1a7bb5578d774e34995dd386fa48e4`; manifest `source-timestep-v1-manifest.json`, SHA256 `42bd9b41105e8f3bfb9901eb67e15d8f05d5f62d307c6bcb383750e1e3385ca2`.
- AC1: `build-sycl-mpi-direct-chi-lapse-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_direct_chi_lapse`, SHA256 `b8814162c43d2319f9d6db2370b54f270f592a5f7101adf36bf94a83b101d5b8`; manifest `source-direct-chi-lapse-v1-manifest.json`, SHA256 `56706997821cb434e142cbac3a217ac7360d4b7fdfc2efe9858e97df9f191b29`.

The initial-data generator reproduces the original N16 input byte for byte with
the preserved donor library. N24/N32 data use the same 4097-point shift quadrature;
metadata retain its refinement estimate. Only the initialization assumes spherical
symmetry. The new generic DT1-to-AC1 importer exactly reproduces the earlier N16
conversion. It preserves physical fields and resets only the volume chi/lapse
and surface chi compensation whose units change from log to linear. It rejects
unadmitted, wrong-time, wrong-resolution, corrupt or mismatched checkpoints.

## Operations and hourly continuation

Remote root: `/home/hzhu/hyperbolic-core-shell-20261028`.
Controller directory: `launcher-resolution-20m-v1` (fresh frozen launcher).
Use its `control.py` and `resolution_campaign.py`; the latter never submits jobs.
The driver source is `tst/hyperbolic_shift_3d/aurora/resolution_campaign.py`.
Use CompactBinaryMerger/debug, one node and 12 MPI ranks/GPU tiles. Leave unrelated
jobs alone. Check ALL user debug jobs before each submission as well as the root
lock. Do not duplicate queued/running or ambiguously submitted allocations.

1. Inspect the latest saved submission, worker, report and `qstat -fx`. For a
   terminal job, run `control.py collect --root ROOT --plan PLAN`.
2. Copy all large output artifacts locally and verify their worker SHA256/size.
   Run `resolution_campaign.py verify --root ROOT --plan PLAN`. This checks the
   terminal scheduler, healthy worker, source/binary hashes, schema/configuration,
   dimensions, dt, time, steps, length, finite payload, chi positivity and sidecar.
3. After a healthy walltime stop, release the matching terminal root lock using
   `control.py collect ... --release-terminal-lock`. Make the next fresh plan with
   `resolution_campaign.py plan --root ROOT --n N --stage STAGE --segment K
   --restart CHECKPOINT`. Bind it with the stage's exact source manifest and
   submit only after all-user debug checks. Never resume numerical failures.
4. When DT1 reaches exactly 3M, run `resolution_campaign.py migrate --root ROOT
   --plan PLAN --output inputs/nN-resolution20m-ac1-at3m.checkpoint`. This requires
   verified completed DT1 data and writes a migration receipt. Begin AC1 stage
   segment01 using that checkpoint. Keep horizon quality flags; a seed is only
   a hint bound to its exact checkpoint, never evidence of a valid horizon.
5. Continue N24 AC1 to20M, then start the already prepared N32 DT1 initial-data
   plan and repeat. If a resolution suffers numerical failure, collect/verify
   its failure evidence, mark it failed without resuming, and proceed to the
   next requested resolution from its fresh initial data. Storage, scheduler,
   access or integrity failures require resolution before any new submission.
6. After both resolutions are complete or numerically failed, collect results,
   compare successful histories with N16 and preserved 1D, commit/push evidence,
   report, release terminal locks, and delete the hourly monitor.

Worker budget3200 seconds, command timeout3100 seconds, native exit reserve180
seconds leave room for source verification, allocation startup and large checkpoint
writes. A walltime stop is expected and is not a numerical failure. Plans reserve
1.6 checkpoint sizes plus10MB for outputs/failure evidence and one checkpoint
plus10MB for atomic writes. Keep the home task footprint below5,000,000,000 bytes.
Retain only the current required restart and useful final checkpoints remotely;
copy and hash-verify older large files before purging them after their successors
have verified outputs. Source/build snapshots for the two active binaries remain
untouched. Older builds were streamed locally and all archive members individually
hash-verified against the remote originals before removal; receipts are retained.

The hourly monitor stays quiet while state is unchanged and reports continuation,
completion, numerical failure or required user action. Counts are segment-local:
sum floor/negative/zero counts across AC1 segments, keeping surface/interpolation
counters distinct. DT1 does not provide AC1 sign logging; do not call its missing
counts zero. Use a consistent time grid when plotting, preserve horizon gaps and
quality flags, and distinguish each run's own-horizon exterior norms from a truly
common-domain convergence norm. Grid ratios are unequal (16/24/32), so do not use
a factor-two convergence formula. Initial quadrature, fixed horizon angular
resolution and changed horizon integration cutoffs can limit measured convergence.


## Initial submission

N24 preparation job8990772 completed its 3M target normally. Its first plan is
`campaigns/n24-resolution20m-v1-dt1_to3m-segment01/plan.json`. The prepared N32
initial plan is `campaigns/n32-resolution20m-v1-dt1_to3m-segment01/plan.json`;
it has not been submitted, preserving one-node-at-a-time execution. The hourly
monitor `aurora-n24-n32-convergence-to20m` is ACTIVE and follows
`resolution20m-campaign-state.json`. The launcher is frozen at commit
88f72f13ffff57cd79cda266b70ac5d2168d459b; its exact file hashes are in
launcher-provenance.json. Both remote initial-data hashes were verified.

After archival cleanup and both fine-grid inputs, the task footprint was
2,610,689,363 bytes. The local build archive is
`output/hyperbolic-core-shell/resolution-20m-v1/archive/obsolete-builds.tar.gz`.
Its21787 member paths/types and all regular-file SHA256s were verified against
remote originals before deletion. A preliminary whole-tar-stream hash comparison
differed, so that comparison was not used to authorize deletion; the successful
member-by-member check is the recorded verification. The two active scientific
builds and all their source manifests remain on Aurora. Controller/preflight
suite109 tests and three focused migration tests passed. The prior N16 importer
parity and initial-data parity are also recorded. No N24/N32 convergence result
is claimed at submission.

## N24 preparation reached 3M

Job8990772 completed 3001 steps to coordinate time1 (3M) without numerical
failure. The native runtime was679.656seconds (11.33minutes), approximately
2.072million evolved zone-cycles/s including native setup/diagnostics. Scheduler
allocation runtime was13minutes26seconds. All four worker-hashed artifacts were
copied locally and SHA256/size verified; the admitted DT1 checkpoint passed
schema/configuration, finite-payload, time and dimension checks.

Null closure remained1.5544e-15 or smaller for surface and ghost diagnostics.
At3M the horizon candidate's expansion RMS was.00145586, above the fixed.001
acceptance tolerance. Keep its radius provisional and the corresponding exterior
norms missing. This is a horizon diagnostic limitation, not an evolution failure
or proof of convergence. Full-grid H_L2 at3M was.0346151.

The verified conversion to AC1 preserves physical fields and resets only the
three compensation components documented above. Its SHA256 is
5e94516836300ca6645594b95910e2172613dc33d7321651cb66bb6672253926.
The continuation plan is
`campaigns/n24-resolution20m-v1-ac1_to20m-segment01/plan.json`.
N32 remains pending until N24 completes or encounters numerical failure.

Continuation job8990986 was submitted after checking that no user debug jobs
were queued/running. The hourly monitor now follows this AC1 stage through the
updated remote campaign-state file. Task storage at binding was3.192GB.

## N24 failure, recovered constraint histories, and N32 submission

N24 AC1 job8990986 failed at5.869M with invalid accepted volume algebraic
projection. Its checkpoint is unadmitted; failure evidence is locally hash-verified
and recorded in n24-failure/. No restart from that state was attempted.

The requested N16/N24 exterior-constraint time series through3M is now measured
using complete provisional horizon candidates. Diagnostic replay job8991129
completed both cases and exactly reproduced each original3M checkpoint, including
all physical fields and RK compensation. See candidate-history/ for the complete
PNG/PDF plot, histories and verification. No missing-value interpolation is used.

N32 preparation job8991329 was submitted with its original initial data and
frozen DT1 binary, dt=.00025, target3M. The hourly monitor follows its saved plan
and remote campaign state, then will convert the verified checkpoint for AC1
continuation toward20M. After verified local archival and removal of retired
inputs/checkpoints, remote footprint at binding was2.796GB; the N32 planned-output
and atomic-checkpoint reserve fit within5GB. One debug node,12 GPU ranks and
CompactBinaryMerger remain unchanged.

## N32 reached 3M; N24 failure location and plot audit

N32 job8991329 completed 4001 steps to exactly3M in1435.999 native seconds.
Its admitted checkpoint and all worker artifacts were copied and hash-verified
locally; the frozen helper verified schema, configuration, payload and provenance.
The final horizon met the fixed acceptance tolerance (expansion RMS0.000777184).
The exterior H and radial Mr norms were0.02075599 and0.04564455, respectively.
Null surface closure was1.5543e-15. See n32-at3m/.

The N24 failed candidate contains one indefinite conformal metric near the
puncture at r=0.01243734, with minimum eigenvalue -0.0780890. All captured volume
fields are finite. This is a volume metric-positivity failure; the message does
not refer to scri null closure. See n24-failure/README.md and its hashed analysis.
N24 has not reached20M. N16 remains the only completed20M resolution.

The apparent change in the3M constraint results comes from comparing full-domain
norms with exterior norms. The original figure's lower panels omitted unaccepted
horizon times; the candidate replay supplies measured exterior values there.
Every shared finite exterior value is unchanged, and both replay checkpoints
are identical to the originals. See candidate-history/plot-reconciliation.md
and exterior-shared-values-verification.json. No convergence claim follows from
the full-domain decrease alone.

N32 AC1 continuation job8992832 is submitted under
`campaigns/n32-resolution20m-v1-ac1_to20m-segment01/plan.json` with unchanged
scientific executable, timestep and floors. The verified conversion SHA256 is
0604741f745df0f30ce687f2b854e3e549a616f8663790abb76856d94acf70e6.
Storage at submission was2.917GB, with0.529GB atomic-checkpoint reserve and
0.840GB planned outputs, leaving0.714GB headroom under5GB. Ten obsolete inputs
and old thin-grid checkpoints were archived locally and individually SHA256/size
verified before removal; receipts are in n32-ac1-submission/. All current N32
restart inputs, initial data and active scientific builds remain remotely.
The hourly monitor and remote campaign state now follow job8992832.

## Exterior history comparison and spatial localization

The all-available-history/ figures combine all retained N16/N24 diagnostics and
an N32 live snapshot through3.6M, explicitly preserving missing exterior values.
N24 has no saved exterior diagnostics after3M; its evolved history ends5.869M.
The spatial-audit/ measurements use identical radial bands on all three3M
checkpoints. The outer band0.6–0.8 supplies95.3% of N32's exterior Hamiltonian
squared integral and94.0% of radial momentum, while inner/overlap errors improve
strongly. The sharp gauge blend centered0.68 is only3.49 cells wide between its
10% and90% levels at N32; it and the nearby0.75 shell interface are concrete
suspects, not proven causes. See the audit for measured ratios, verification,
finer radial localization and controlled follow-up suggestions. The campaign
scientific executables, timestep, floors and equations remain unchanged.
