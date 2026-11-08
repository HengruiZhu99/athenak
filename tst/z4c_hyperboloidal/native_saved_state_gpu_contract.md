# Saved-step native device replay gate

This test-only harness decodes the independent preserved N12 grid ordering on
rank zero, reads fields.bin/boundary.bin once, scatters owned native bulk values,
and distributes the small surface input. No evolution driver or legacy RHS is
executed. The decoder is discarded before native `Diagnostics::Refresh` and
`HistorySums` execute using block-owned storage and sparse transport.

The actual full and strict fixed-radius `r > .94` ten raw sums are computed on
the device, using the legacy `Diagnose` expression literally:
`sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]) > .94`. The test does not replace this by a
squared comparison, a tolerant mask, or a host-computed production cut. It
compares the eight RMS values to the supplied pinned time.json with the
unchanged `5e-11*(1+abs(reference))` gate. Full/harmonic-exterior histories use
`5e-10+1e-6*abs(reference)` and areal radius/irreducible mass use `1e-10+1e-7*abs(reference)`
in M units. All differences and gates are recorded even when the strict replay
fails. Failures produce results.json and point dumps before the nonzero exit.
A successful CPU predecessor oracle does not replace the original GPU gate.

The additional host-cut control reuses the same device values/reduction, with a
host-computed mask, solely to attribute threshold discrepancies. It is never
substituted into the admission observable. The diagnostic band
`abs(r-.94) <= 32*epsilon` counts points only; it does not alter either mask.
Both flip orientations, positive/zero-weight subsets, exact-cut points, total
mask-weight difference, and all five raw device-minus-host-cut contributions
are available. Near-cut CSV files use hexadecimal floats; exact coordinate and
radius bytes are retained in the binary point records.

Owned 31-field values and all eight bulk/scri ChiRecord members must remain
byte-identical. All six RK/RHS arrays retain sentinels; StateVersion and the
stored stage compatibility/clock are unchanged. Every evolved point is checked
for valid diagnostics, including points with zero diagnostic weight. This
saved-state test creates no actual new RK stage observations.

## Portable build and launch

`native_saved_state_gpu.py build` reads the native `diagnostics.cpp` entry from
compile_commands.json and the native target's link.txt. It preserves the actual
compiler, defines, includes, optimization, device target and floating-point
flags; replaces only the main test translation unit; adds the explicitly listed
read-only legacy reference macros/includes; and builds with make -j64. It never
assumes `/usr/bin/c++`, configures a new runtime target, or submits a job. It can
be staged with its cpp and this contract in a separate validation directory,
leaving the frozen native source tree unchanged. Compiler/dependency/source,
input checkpoint and object hashes accompany the test executable.

Verified Aurora paths for this validation are:

```
native: /lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side
build:  <native>/build-sycl
legacy: /lus/flare/projects/CompactBinaryMerger/hzhu/scri-ko01-20261106/source
saved:  <native>/reference/step3551
```

After loading the same oneAPI module/environment as the native build:

```sh
python3 native_saved_state_gpu.py build --build <native>/build-sycl \
  --legacy /lus/flare/projects/CompactBinaryMerger/hzhu/scri-ko01-20261106/source \
  --reference <native>/reference/step3551 --output <native>/validation/saved3551-build
```

Use the already established `gpu_tile_compact.sh` wrapper, `-ppn 12` and the
existing CPU affinity list. The executable takes the native N12 input and saved
snapshot directory, writing into the job's working directory. The same binary
supports 24 ranks on two nodes and 96 on eight nodes. Actual scheduling remains
the root agent's responsibility; this adapter performs no launch/submission.
The input should be the unchanged `source/tst/z4c_hyperboloidal/n12_initialize.athinput`.

## Canonical point records

Each `points-rank<R>.bin` begins with 16 bytes (`NATIVE_DIAG_V1` plus two NULs),
uint64 local record count, and uint32 key/value counts (8,26), using native little
endian storage on Aurora. Each record writes int32[8] and double[26] explicitly,
without structure padding:

- Keys: legacy active index, native global leaf ID, k-origin, j-origin, i-origin,
  strict device fixed mask, strict host fixed mask, complete harmonic AH mask.
- Values: x/y/z; the 19 fields in constraint_oracle.cpp's Pack order; native
  quadrature weight; device radius; host radius; independent legacy host weight.

The xyz values are the actual cached native geometry used by device operators.
The runtime builds those mapped coordinates on the host before uploading them;
the device radius/cut is computed on the GPU. These two claims are kept distinct.
`audit --output RUN [--compare OTHER_RUN]` requires complete unique coverage of
229392 points, reconstructs canonical ordering, names mask counts, and reports
per-component byte differences/max absolute errors and separate fixed/AH mask
changes. An exact cross-rank comparison is required when `--compare` is supplied.
