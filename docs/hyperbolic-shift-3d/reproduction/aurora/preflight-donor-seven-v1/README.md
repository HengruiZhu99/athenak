# V6 seven-node radial closure: Aurora preflight passed

Job **8984220** completed on 2026-10-29 with scheduler state **F/exit0**,
on one CompactBinaryMerger/debug node, twelve MPI ranks and twelve distinct
GPU tiles. Independent verification passed all **179 checks** and all **39
worker-bound artifacts**. This is implementation/restart qualification, not
duration stability, convergence or a 100 M result. The allocation lock is released.

| Check | Result |
|---|---:|
| Complete bulk CPU/SYCL scaled difference | 3.36e-14 |
| Sampled pole numerator difference | 5.67e-16 |
| Surface-rate difference | 7.90e-14 |
| Dependent-D difference | 1.93e-14 |
| Ownership partition difference | exactly zero |
| Numerator dependence on provisional finite D | exactly zero |
| Four-step pulse versus two-plus-two restart | whole files bitwise equal |
| Minkowski state deviation at t=.002 | 8.82e-15 |
| Minkowski H/M RMS at t=.002 | 7.53e-15 / 3.79e-15 |
| Puncture horizon mass at t=.002=.006M | .3333350354 |
| Puncture horizon expansion RMS | 3.25e-4 |
| Largest evolution surface/ghost null residual | 1.11e-15 |

The puncture uses lmax12/ntheta24/nphi48 and the unchanged .001 expansion
tolerance. Its four-step horizon result does not establish accurate puncture
evolution or agreement with the late 1D baseline.

The fresh source-v13/build-sycl-mpi-v13 snapshot is frozen at scientific
commit `7a9d83d4a16e0e43d2439c02478235996a2a7dcf`. All 2,632 source blobs,
including 1,486 Kokkos files, were independently verified against Git.
The SYCL/MPI/PVC binary64 build used `make -j64` and completed successfully
in approximately 400 seconds. Seven named executable hashes and the exact
build settings are retained in [the build verification](verification/build-verification-v1.json).
Campaign footprint after building was **2,524,810,661 bytes**, and after the
preflight **2,721,879,980 bytes**, both below 5 GB.

The [bound plan](plan.json) has seven cases and 179 acceptance checks:
complete CPU/SYCL bulk and boundary parity on each GPU tile; fresh n16
Minkowski; four pulse steps compared bitwise with a two-plus-two restart;
and four isotropic-puncture steps with the hyperboloidal FastFlow finder.
V6 requires header tuple `(2,7,4,3)` and epsilon 2. All earlier sources,
binaries and checkpoint formats remain separate.

The [local implementation checks](../../scri-donor-seven-native-v1/README.md)
passed on Serial and one/two MPI ranks. Their short n6 duration controls
were wall-time bounded; they do not establish stability. Following independent
verification, [admission.json](admission.json) qualifies the named V6 operator
for duration tests. The prepared next experiment is fresh n16
Minkowski to coordinate time 1 at dt=.0005, with the previous state,
constraint and nullness limits unchanged. Puncture duration and convergence
remain outstanding.

The verification script is copied byte-for-byte from its executed staging
location. Its sixteen offline negative tests, build audit and
[campaign verification](verification/verification-v1.json) pass. The decoder
independently checks the exact V6 header and all four finite arrays, including
RK compensation. Full checkpoints and initial data are locally archived in
`output/hyperbolic-core-shell/aurora/preflight-donor-seven-v1`; only small
evidence is copied here. Scheduler environment data remain in the local
archive; the portable terminal whitelist records the original collection hash.

To rerun verification, supply explicit `--repo`, `--raw-dir`, `--build-dir`,
`--initial-data` and a fresh `--output` to `verification/verify.py`; the local
build directory is the archived campaign's `build-v13`, and initial data are
in its `inputs/puncture-n16.txt`. No verifier submits or evolves a checkpoint.
