# Scri recorder validation and cost, 2026-11-15

Implementation/schema: [scri-recorder.md](scri-recorder.md).
Compact evidence: [report.json](evidence/scri-recorder-20261115/report.json),
[performance.json](evidence/scri-recorder-20261115/performance.json),
[CTest log](evidence/scri-recorder-20261115/ctest.log), and
[source hashes](evidence/scri-recorder-20261115/source-sha256.json).

All tests passed on the requested integration worktree, based on `edbe24a3d`.
The machine was an Apple M5 Pro, Apple LLVM 21, binary64, Kokkos Serial backend,
Open MPI, with one, two and eight ranks. No GPU performance/portability result is
claimed by this CPU validation. No long binary evolution was launched. The
integration evolutions were two N12 timesteps; other cases were manufactured
accepted states, static initialization, or actual AMR without time evolution.

Commands:

```sh
cmake --build build-binary -j 8
ctest --test-dir build-binary --output-on-failure
python3 tst/z4c_hyperboloidal/scri_recorder.py \
  --build build-binary --output /tmp/hyperboloidal-scri-release
```

The test script requires a fresh output directory. Large raw test archives and
checkpoints were left in `/tmp/hyperboloidal-scri-release`; the repository retains
only compact logs, numerical summaries and source hashes. Existing production
outputs and research documents were not modified.

## Results

- All 13 CTests passed, including the new quadrature test and existing native
  topology, exchange, RK4, residual-weight and scri reconstruction/reference tests.
- 44 rank-local reconstruction checks passed across two/eight-rank manufactured
  states, actual refinement/coarsening, same-time topology invalidation, and
  static initialization of treatments B and C (including their fourth lapse lift).
- Raw support fields, Cartesian coordinates/Jacobians/Hessians and all reference
  jets match independent reads of the actual module arrays bit for bit.
- Saved endpoint derivative rows reproduce `sampled_residual_jets` exactly
  (maximum error zero). Python reconstruction of the normal lifts has maximum
  absolute error `5.55e-16`; the dependent/reference-restored values also pass.
- Nonzero manufactured states pass bitwise hashes of all evolution arrays, RK
  accumulators/RHSs, residual workspace, chi histories and state-version counter
  before/after recording. No old RK RHS is archived as a time derivative.
- Actual outer refinement/coarsening takes 106 → 113 → 106 native leaves, advancing
  mesh sequence 0 → 1 → 2. A same-time invalidation also produces sequence 3 and
  a fresh epoch instead of being mistaken for a repeated final-output call.
- One/two/eight-rank nodal data agree bitwise and modes agree to reduction
  roundoff. Eight-rank fixtures include ranks with no scri nodes. Refined/coarsened
  data and modes also agree across two/eight-rank ownership.
- Recorder enabled versus disabled: all **24,704,784** checkpoint scalars match
  bitwise after two timesteps, including persistent chi records. Restarting the
  initial two-rank checkpoint onto eight ranks reproduces the same final payload:
  `d9e9fa19ff769823ad1db868578dcdf84db26aeb4dcb67bc507422a186cb4a90` (SHA256).
- Restart records continue numbering without rewriting cycle zero. Native `dt`
  and `dcycle`, initial/final calls, three simultaneous output blocks, nodal-only
  output with ignored lmax, modes-only output and nondefault lmax=3 all pass.
- Negative and underresolved modal lmax, disabling both outputs, truncated files,
  and duplicate-rank assembly are rejected.
- Independent Cartesian formulas for `Y00`, `Y1,-1`, `Y10`, `Y11` verify the writer's
  actual field/weight/reduction wiring for every field, both rank counts, and
  every manufactured epoch. Maximum discrepancy is `8.88e-16`; see
  [analytic-modes.json](evidence/scri-recorder-20261115/analytic-modes.json).

### Angular accuracy

The harmonic test excites every m at l=3 and l=8, and projects onto **every** mode
through l=8. The maximum Gram error includes both normalization errors and
cross-mode leakage, not just total area. Patch subdivision reproduces the same
native-node weights to roundoff. Errors decrease at second order as expected for
midpoint quadrature; lmax=8 admission at N12 is not a claim of negligible error.

| Map | N per face axis | Absolute area error | Maximum harmonic Gram error |
|---|---:|---:|---:|
| Gnomonic | 12 | 2.1386e-2 | 3.14815e-3 |
| Gnomonic | 24 | 5.34602e-3 | 7.66430e-4 |
| Gnomonic | 48 | 1.33647e-3 | 1.90365e-4 |
| Equiangular | 12 | 1.31979e-2 | 3.16044e-3 |
| Equiangular | 24 | 3.29805e-3 | 6.86713e-4 |
| Equiangular | 48 | 8.24425e-4 | 1.65933e-4 |

## Storage and performance

N12 has 864 owned endpoint nodes. In the two-rank fixture the stored donor union
has 8,363 bulk entries and 1,184 endpoint entries across rank files, including
intentional cross-rank support duplication. The 189,888 ordered derivative terms
are static metadata. The archive retains all 31 fields and redundant endpoint/
lift jets, not just 26 independent fields or a truncated modal series.

| Measurement | Result |
|---|---:|
| Complete nodal/support + lmax=8 modes, per accepted output, summed over ranks | **5,699,150 bytes** (5.70 MB) |
| Static epoch metadata, summed over two ranks | **63,207,988 bytes** (63.21 MB) |
| Output storage per unit simulation coordinate time at example dt=0.01 | **569,915,000 bytes** (0.570 GB) |
| Cached synchronized output, two ranks, three repetitions, maximum rank wall time | 0.01638, 0.01656, 0.01659 s |
| Cached synchronized output, eight ranks, same fixture | 0.01121, 0.00792, 0.00800 s |
| Initial two-rank epoch preparation + write, root timing | approximately 0.255 s |
| Two-step Driver time, disabled / enabled (output every step) | 20.83889 / 20.88590 s |
| End-to-end run including setup/restart output, disabled / enabled | 30.55503 / 30.99874 s |

The two-step measured Driver difference is 0.04701 s (0.23%). These are short
local measurements, not statistically precise speedup/slowdown estimates. The
synchronized output-only benchmark isolates the recorder from evolution rank
imbalance; the per-record `preparation_seconds` in an evolving run can include
waiting for a slower rank and must not be interpreted entirely as added work.
Writes are buffered filesystem I/O; no sustained-storage/fsync throughput claim
is made. AMR/restart epoch setup costs are additional and are visible in the logs.

At dt=0.01, 0.0166 s/record extrapolates to approximately 1.66 s of recorder work
per code-time unit. Relative to this fixture's two-step evolution cost, that is
about 0.0075%, excluding occasional static-epoch work. This is an extrapolation,
not a long-cadence campaign measurement. Storage remains substantial: 100 code-time
units at that cadence would produce about 57 GB plus static epochs.

Disabled-output recorder overhead is zero calls, allocations, communication and
I/O: without an active `file_type=scri` object there is no recorder work or stage
hook. The measured disabled baseline is given above; no before/after executable
microbenchmark is used to claim a sub-noise wall-time difference. Evolution math
and task structure are unchanged, with the bitwise enabled/disabled test providing
the direct regression check.

The storage rate is deliberately stated per **simulation coordinate-time unit**.
No astrophysical mass normalization or inertial Bondi clock is assumed. If one
mass unit equals `M_code` code-time units, multiply the stated rate by `M_code`
to obtain bytes per mass unit. Temporal and angular convergence, higher-order
asymptotic reconstruction, physical radiation extraction and BMS/frame analysis
remain subsequent work, as specified in the schema document.
