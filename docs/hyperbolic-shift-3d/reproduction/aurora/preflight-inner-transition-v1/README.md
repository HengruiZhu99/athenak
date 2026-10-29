# V7 Aurora qualification submitted; no result yet

Job **8984332** was submitted to CompactBinaryMerger/debug on one node with
12 MPI ranks and12 GPU tiles. The fresh V7 candidate changes inner-transition
halo interpolation while retaining V6 radial scri closure. This package
records preparation and build receipts only. **GPU qualification and duration
stability are not yet established.**

The immutable source-v14 commit is
`9022f51f26e66acafdd3190d0c33f0c1ea53a661`. Its source-manifest SHA is
`da9304ba17ad5a1336aef4468dc88ce375622104bac2c69cc1d3cc534abcb2a4`.
The reviewed plan SHA is
`a2cdb0ff507a7edc632cb2ce1c226cf50cd30562a7d729e7abc537583995c7d8`.
All2637 source/Kokkos files were independently verified against Git blobs.
The fresh precise-FP64 SYCL/MPI build completed with `make -j64`, and all three
named binaries and the CMake configuration were hash-bound before submission.
Campaign storage after building was **3,195,542,713 bytes**.

The plan requires184 checks across full CPU/GPU component parity, fresh short
Minkowski/puncture and direct/two-part pulse restart cases. The independent
verifier and16 passing negative-control tests are retained in `verification`.
Pins were established before the job's results, and no V7 scientific admission
has been issued. The [local tests](../../inner-transition-native-v1/README.md)
include complete RHS/halo parity and actual one/two-rank checkpoint identity.

Build metadata is under `build-receipts`; the actual source, binaries and large
results remain under the bounded Aurora campaign root. The later full verifier
must be run with explicit `--repo`, `--raw-dir`, `--build-dir`, `--initial-data`,
`--pins` and fresh `--output` arguments after terminal collection and release.
