# V6 seven-node radial closure: Aurora preflight submitted

Job **8984220** was submitted on 2026-10-29 to CompactBinaryMerger/debug,
one node, twelve MPI ranks and twelve GPU tiles, maximum thirty minutes.
The last scheduler observation at this snapshot was **Q**. No GPU parity,
restart or duration admission has been issued for V6 yet.

The fresh source-v13/build-sycl-mpi-v13 snapshot is frozen at scientific
commit `7a9d83d4a16e0e43d2439c02478235996a2a7dcf`. All 2,632 source blobs,
including 1,486 Kokkos files, were independently verified against Git.
The SYCL/MPI/PVC binary64 build used `make -j64` and completed successfully
in approximately 400 seconds. Seven named executable hashes and the exact
build settings are retained in [the build verification](verification/build-verification-v1.json).
Campaign footprint after building was **2,524,810,661 bytes**, below 5 GB.

The [bound plan](plan.json) has seven cases and 179 acceptance checks:
complete CPU/SYCL bulk and boundary parity on each GPU tile; fresh n16
Minkowski; four pulse steps compared bitwise with a two-plus-two restart;
and four isotropic-puncture steps with the hyperboloidal FastFlow finder.
V6 requires header tuple `(2,7,4,3)` and epsilon 2. All earlier sources,
binaries and checkpoint formats remain separate.

The [local implementation checks](../../scri-donor-seven-native-v1/README.md)
passed on Serial and one/two MPI ranks. Their short n6 duration controls
were wall-time bounded; they do not establish stability. After this GPU
preflight passes independently, the prepared next experiment is fresh n16
Minkowski to coordinate time 1 at dt=.0005, with the previous state,
constraint and nullness limits unchanged. Puncture duration and convergence
remain outstanding.

The verification script is copied byte-for-byte from its executed staging
location. Its sixteen offline negative tests and build audit pass. The
campaign verifier has **not yet run**, because the job is pending. Full
checkpoint payloads will be collected under local ignored `output/`; only
small evidence belongs in this package. The inventory binds files present
at this submission snapshot and will be extended when results arrive.
