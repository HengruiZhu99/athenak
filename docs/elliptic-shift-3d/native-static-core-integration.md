# Static native Cartesian core with eight blocks

`NativeCompositeEvolution` now maps either one or eight static native Cartesian
MeshBlocks into its composite domain using actual native physical bounds and
relative levels. The new selected-wormhole input has global12³ cells, per-block6³,
ng3/order4 and eight native blocks. The seven engine-owned shell layers retain
radii `.45,.55,.60,.65,.70,.75,1`; all50 composite blocks have logical6³ shape,
for10800 active cells. Cartesian spacing is1/24. This remains a Serial, double,
vacuum experimental owner. Native AMR/regridding, native shell packs, MPI/device,
tracker motion and generic wormhole-to-trumpet safety are not qualified.

The source still uses the selected isotropic M=1/3 wormhole, Minkowski reference,
positive-storage2, direct signed fields, selected lapse/telegrapher controls and
width.24. Leading-log derivative enrichment and the existing global topology
preconditioner/radial-line smoother are enabled. No exponent, floor, parity,
shift pin, source subtraction or boundary tolerance is changed.

## Mapping and restart

The existing `NativeCartesianCoreMap` validates a bijection between the complete
native pack and all Cartesian composite leaves. `NativeRecords` uses native
`mb_size` bounds and `mb_lev-root_level`; current runtime requires relative0 and
one pack containing the entire native core. Unsupported nonuniform/refined
layouts fail. Scatter, gather verification, native physical ADM conversion and
constraint output now address every mapped block, including all native ghost
points. The operator's canonical unknown remains the global Cartesian vector.

Eight-block sidecars use a separate static-core format and bind native count,
logical core subdivision, native/composite slots, relative levels and physical
bounds in addition to the inherited gauge/slice/receipt/chart/RK state. The
one-block sidecar layout, values and arithmetic are preserved. The actual native
restart header stores an absolute `composite_restart_file`; relocating a native
checkpoint requires explicitly setting that path to the copied paired sidecar.
The topology negative controls below supply it explicitly.

## Mechanical qualification

Build: `evidence/native-static8-build-20261027/manifest.json`, two workers,
7.29 seconds, zero source changes. Its archive contains the360 source files
appearing in actual Athena compiler dependencies, each checked against its
before-build hash. The frozen executable SHA256 is
`6f4eee9dce9d94daaf57662500afcb975bd98fcdf6947024e6adb7c1edd1a627`.
Unrelated standalone research prototypes are outside this archive.

`evidence/native-static8-controls-20261027` records:

- actual Driver/RK two-cycle t=.0002 execution31.91s, eight stage updates and
  eleven RHS evaluations;
- initializer-hidden restart15.54s, with bitwise final volume/scri, complete
  values/registers/shift warmstart, topology and actual RK arrays;
- same-engine matched actual Driver RK reference29.53s, all22 coefficients and
  three shift corrections bitwise at cycles0/1/2 over all10800 cells;
- all25 native physical restart fields bitwise equal to independently decoded
  global active values on all1728 native active cells, matched by topology;
- independent double nativeRST-to-ADM conversion for all eight blocks and all
  ghosts, every output field bitwise after float32 quantization. Maximum
  unquantized relative rounding5.913e-8 meets the unchanged3e-7 control;
- old Minkowski15-control suite67.70s, all34 previously frozen CSVs bitwise;
- old one-core selected-wormhole26.34s, all6 previously frozen CSVs bitwise;
- wrong receipt, taper, chart and Minkowski reinterpretations rejected.

The rounded-input BIN32 ADM proxy still fails:4.759e-7 exceeds3e-7. It is retained
separately from the stronger fullprecision nativeRST oracle. No tolerance was
relaxed. Native field-layout/stage/core-map CTests pass3/3 in.07s.

`evidence/native-static8-topology-qualified-20261027` changes copies of the
checksummed sidecar count, map and bounds. Actual native restart rejects all
three at their intended guards in2.16/2.34/2.22s. The earlier
`native-static8-topology-controls-20261027` failed test setup is retained: its
copied native header still named the original absolute sidecar, so one copied
restart evolved a cycle instead of reading the mutation. No topology-rejection
claim is made for that attempt and original checkpoints were unchanged.

Each newly owned process has a180s cap; no original job is signaled. Compression
manifests retain hashes and lossless gzip copies of raw outputs; originals remain
locally untouched. Decompress copies to the recorded original names before
rerunning readers. Compiled executables can instead be rebuilt from source.

## Scientific qualification remains failed

| Final diagnostic | One native block | Eight native blocks |
|---|---:|---:|
| Cartesian spacing |.0833333|.0416667|
| minimum sampled radius |.0721688|.0360844|
| all-cell H RMS |7.326832|.616059|
| maximum H |87.08314|2.153461|
| all-cell M RMS |.07522385|.07109833|
| Theta RMS |.000486180|.0000100082|
| Z RMS |.000225066|.000217977|
| tensor scri pole |1.41508e-5|1.44116e-5|
| original shift RMS stage maximum |6.96038e-12|5.18509e-12|
| original shift point maximum |3.01290e-10|1.85835e-10|

Clock/null/tangential residuals remain at roundoff. The tensor pole fails its
unchanged1e-7 target, and physical constraints remain large. A cell inside the
throat does not resolve the endpoint or establish collapse safety: throat radius
.0629245 is only1.51 Cartesian cell widths in the eight-block pilot.

The independent selected-width.24 N2049/N1025 radial comparison is in
`evidence/native-static8-radial-comparison-qualified-20261027`. It retains every
active cell with its Penrose proper-volume weight, full-region diagnostics,
physical-field errors and actual-node raw directions/values/angular scatter.
No angular average enters evolution. Geometry-coincident nodal radii are grouped
only for reporting; these are not interpolated64-ray samples. Beta_r volume RMS
error decreases from.00136819 to.00102225 and lapse error from1.278e-6 to9.222e-7.
The eight-block r<.1 H RMS is1.7678; cube/shell stencil-neighbor H RMS is.19299.
Maximum nodal beta radial scatter is.00171138, so radial reproduction remains
incomplete. The shell mesh is unchanged; these two controls do not establish a
uniform spatial-convergence order. Native Ketcheson2S RK4 versus the stored
radial classical RK4 also requires an independent timestep control.

The first read-only comparison was intentionally rejected by its immutable-input
hash check because the runtime manifest was still changing. Its failure record
is retained separately; only the qualified directory supports numerical claims.
The scientific gate remains false, with no late lapse-front/collapse, arbitrary
puncture regularity, translating-hole or binary claim.

Reproduce in an idle shared build directory, always with fresh output paths:

```
python3 tst/elliptic_shift_3d/build_native_compatibility.py --output NEW_BUILD
python3 tst/elliptic_shift_3d/run_native_wormhole.py --core-bulk 2 --output NEW_RUN
python3 tst/elliptic_shift_3d/analyze_native_wormhole.py NEW_RUN
python3 tst/elliptic_shift_3d/audit_native_core_restart.py --controls NEW_RUN --output NEW_TOPOLOGY
python3 tst/elliptic_shift_3d/compare_native_radial.py --native NEW_RUN --output NEW_COMPARISON
```
