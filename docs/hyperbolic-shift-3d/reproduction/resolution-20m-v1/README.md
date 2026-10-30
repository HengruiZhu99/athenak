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
