# Native static Cartesian refinement qualification

`NativeCompositeEvolution` now accepts a complete Serial static-SMR Cartesian
core, using actual AthenaK leaf bounds and relative levels. Dynamic AMR remains
rejected. The canonical global tensor engine still owns shell/scri state;
shells are **not native MeshBlocks**. This implements static native tree/field/
Driver integration, not generic native multipatch/MPI/device support.

## Actual native geometry and retained guards

AthenaK `Mesh` rejects odd ghost counts for SMR (`mesh.cpp:256`); Z4c permits
order4 with ng4 (`z4c.cpp:185`). The new path therefore requires ng4 for static
SMR while preserving the original one/eight-block ng3 uniform path. It does not
merely remove the old multilevel guard.

`NativeRecords` reads `mb_size.h_view(m)` and
`mb_lev.h_view(m)-pmesh->root_level`. Root subdivisions come from the three
actual `nmb_rootx*` counts, which must agree. Each leaf width must equal the
root width times 2^(-relative level), and its bounds must lie on that dyadic
lattice anchored at -half_width. The full native pack must be a bijection onto
all Cartesian leaves, covering the inner cube without holes or overlaps.
No leaf is discarded. Native physical domain bounds must equal the inner box.
Every native ghost corner must remain inside scri.

`NativeCartesianCoreMap::FromNativeLeaves` constructs those Cartesian leaves
before unchanged shell descriptors. Native pack order, not a fabricated uniform
order, fixes the coordinate/initializer catalog. All blocks retain identical N³
active shape and the common ng4 contract. A preflight
`CompositeCoarseningPlan` merges complete finest Cartesian sibling families and
coarsens logical cells to N2; it must retain the existing1024-cell dense coarse
cap. The active-cell bound remains50000. Unsupported topology fails before
constructing the tensor engine. Actual runtime tests below have400 terminal
coarse cells.

Serial/double/one complete pack, vacuum, separate physical ADM gauge, order4,
chi^-4, selected dedicated pgens and RK3/RK4 guards remain. Adaptive regrid,
MPI/device, matter, tracker motion, floors, drift control, native shell packs
and midstage restart are rejected. No PDE, lapse source, reference, shift
boundary condition, scientific tolerance or puncture power is changed. These
native tests retain the default57-column log derivative; the separate optional
supported58 radial-first prototype is not silently enabled here.

## Native state, mirror and restart

All native mirror/RHS/ADM/constraint operations use the validated physical leaf
map. Static ghost coordinates use each real native block's `dx*` and bounds;
the former one-box coordinate shortcut is retained only on its original uniform
path. All native ghosts are filled from the canonical global active solution,
not independently frozen patch data. The global mapped operator and multigrid
interfaces remain responsible for the solve.

Static-SMR sidecars have distinct Minkowski/wormhole magic strings, plus the full
native count, root subdivision, native/composite slots, relative levels and
physical bounds. Existing uniform sidecar layouts are unchanged. The inherited
coefficient/register/scri/shift/RK metadata remain complete. `Restore` intentionally
increments the geometry epoch once to invalidate old shift certification, then
resolves the accepted shift. Thus numerical restart payloads and physical CSVs
are bitwise, but entire sidecar bytes differ by the epoch and checksum. This is
explicitly tested rather than hiding the difference or weakening a PDE target.

The actual mesh/coordinate receipt is written once in `PrepareNative`, reached
by `Driver::Initialize` after `ChangeRunDir`. Constructor and pgen `Initialize`
both precede that directory change; two failed receipt-placement attempts are
retained separately. Their evolutions succeeded, but their runner setup did not
qualify. The earlier directory named `native-static-smr-minkowski-qualified-20261027`
contains such a failed setup and does not support a qualification claim.

## Bounded evidence

All newly owned numerical processes use one Serial CPU and at most180 seconds.
No original1D process or checkout was signaled or modified.

- `evidence/native-static-leaves-20261027`: nine root/shell geometries,282 mapped
  leaves,45 malformed-geometry/budget rejections, largest terminal848<=1024.
  A complete root5 example exceeding the coarse cap remains rejected.
- `evidence/native-static-smr-build-receipt-20261027`: shared Athena target
  rebuilt in7.58 seconds with two compiler workers and no source changes during
  build;428 source/config files and executable are frozen. This binary still
  contains the second receipt-placement attempt.
- `evidence/native-static-smr-audited-relink-20261027`: only the final owned native
  translation unit rebuilt against those immutable headers, then linked to the
  exact hashed Athena object/library set into a separate evidence executable.
  Remaining vendor/generated headers were archived. The final compiler dependency
  probe has **zero live project headers**; all link inputs and vendor headers
  passed before/after hash checks. No shared object or executable was changed by
  this relink. Its SHA256 is
  `35cc34c4c90e781bc4fe13fc824631bec6b37f42b28eb813f9e0c0aa12b9f322`.
  Do not identify it with the shared build binary until a later full rebuild.

The runtime mesh is root2 with one negative octant refined once:15 actual native
leaves at mixed relative levels0/1, seven shell layers
`.45,.55,.60,.65,.70,.75,1`, N6/ng4, and12312 active global cells. The native core
contains3240 active cells;9072 shell cells remain in every full-domain norm.

| Control | Minkowski | Selected isotropic M=1/3 wormhole |
|---|---:|---:|
| Actual Driver two cycles, t=.0002 |18.72s|63.16s|
| Restart one cycle |12.30s|29.51s|
| Matched actual-topology/live-RK reference |13.34s|54.21s|
| Independent physical mirror/ADM analysis |1.97s|1.93s|
| Maximum unquantized ADM output rounding |5.8303e-8|5.9317e-8|

Each native run executes eight actual Driver/RK stage updates and eleven RHS
calls. All22 volume/scri coefficients and all three canonical algebraic shift
corrections match bitwise at cycles0/1/2 against the same-engine reference driven
by the actual native leaf/RK receipts. That is a dispatch/recurrence control,
not an independent PDE derivation. Its compile dependency probe also has zero
live project headers.

For both runs, all25 physical native active fields match fullprecision nativeRST
and independently decoded global CSVs bitwise on all15 leaves. Independent
Penrose-to-ADM conversion covers every native active and ghost point. Every
quantized output field matches bitwise and the unchanged3e-7 rounding target
passes. Complete numerical restart values, registers, chart, topology, actual
RK arrays, shift warm start and accepted time match bitwise; geometry10->11 is
the required rebuild invalidation.

The wormhole uses the existing independent isotropic M=1/3 initial-data generator
on the **exact actual native/composite coordinate catalog**, whose bytes are
checked against the wormhole runtime receipt. Its initial file is hidden before
restart, which still succeeds. Minkowski reference, selected sources and taper.24
are unchanged. Evidence is in `native-static-smr-controls-r3-20261027` and
`native-static-smr-wormhole-20261027`; each has raw diagnostics, manifests,
reference output, dependency probes and `independent-analysis.json`.

`native-static-smr-topology-controls-20261027` mutates checksum-correct copies of
native count, map slot, relative level and physical bound. All four actual native
restarts reject at the intended guards in3.78--3.92 seconds. The original native
checkpoint, paired sidecar and frozen executable remain unchanged. The uniform
ng3 Minkowski regression takes2.97 seconds and retains all six established
volume/scri CSVs bitwise. Adaptive ng4 and odd-ghost SMR are also rejected.

## Scientific failures retained

Minkowski final H RMS6.5562e-16, H max9.4369e-15, state deviation2.0365e-16,
scri tensor pole2.7210e-15; null/tangential residuals remain at roundoff. This is
a useful exact-equilibrium gate across mixed native levels and the host shells.

The wormhole is **not scientifically qualified**:

| Final diagnostic | Value |
|---|---:|
| Full-domain H RMS |.3509118|
| H maximum |26.82605|
| M RMS |.07128091|
| Theta RMS |1.22267e-5|
| Z RMS |.000215833|
| Original shift RMS maximum |7.4969e-12|
| Original shift point maximum |2.9937e-10|
| Tensor scri pole |1.44249e-5|

The tensor pole fails the unchanged1e-7 target. H peaks at ordinal1506,
`(-.23958333,-.01041667,-.01041667)`, rho=.2400358, where a refined Cartesian
leaf meets coarser Cartesian directions near the cube/shell interface. The same
point has H26.82145 initially, so most of this error precedes time advancement.
No exact-ghost/source diagnostic here proves which transfer term dominates.
The core r<.1 H RMS is.81355; Cartesian H RMS.63928; shell H RMS.075925.
All cells and complementary regions remain reported in
`all-region-H-localization.json`. This asymmetrically refined mechanical control
is not a radial or spatial-convergence family. No late-front, puncture-collapse,
translating-hole, binary, scalability or generic changing-regularity claim follows.

## Reproduction

Use fresh output directories. Existing evidence binaries are immutable; a new
build/relink must use its own recorded source/objects rather than silently mixing
current headers with a frozen executable.

```sh
python3 tst/elliptic_shift_3d/run_native_static_smr.py --output NEW_MINKOWSKI
python3 tst/elliptic_shift_3d/run_native_static_smr.py --wormhole --coordinates NEW_MINKOWSKI/full/native_smr.coordinates.csv --output NEW_WORMHOLE
python3 tst/elliptic_shift_3d/analyze_native_static_smr.py NEW_WORMHOLE
python3 tst/elliptic_shift_3d/test_native_static_restart_controls.py --controls NEW_WORMHOLE --output NEW_TOPOLOGY_CONTROLS
```

The remaining native dynamic transaction is audited in
[native-dynamic-owner-lifecycle.md](native-dynamic-owner-lifecycle.md). Its absence
is unfinished integration, not a mathematical obstruction or an invitation to
remove the adaptive guard.

## Final shared build and package

The final shared Athena target was rebuilt against the current engine/default0
boundary path in8.18 seconds. `native-static-smr-final-shared-build-20261027`
freezes the actual187 compiler dependency files and679 transitive project,
vendor and generated files;368 applicable recorded pre-build hashes match their
post-build content. The remaining dependency contents are frozen with exact
hashes without inventing a pre-build verification. Its actual shared executable
SHA256 is `e852583817bc877dcc211555ffe86a896193bf3cba0d57773000679e3c802f6b`.

The actual shared binary passes uniform ng3 in3.01s and the mixed native SMR
Minkowski case in17.90s. All six snapshot CSVs per case are bitwise against the
independently qualified earlier controls; the seventh static coordinate receipt
also matches bitwise. The first post-run assertion counted that seventh receipt
as an unexpected snapshot; its manifest is retained separately. Correcting the
snapshot count required no numerical or executable change and no repeated
physics run. Evidence is `native-static-smr-final-shared-controls-20261027`.
This completes final shared-native default-path qualification; it does not
change the failed wormhole scientific gates above.

All12 owned new native-SMR/leaf evidence directories are losslessly packaged by
`native-static-smr-package-20261027/manifest.json`. Source/header trees are
verified deterministic tar.gz archives. Repeated nativeRST/ADM/CSV outputs use
an exact-byte content-addressed data capsule and index, preserving every failed
attempt. Every original file SHA256 is checked after independent reassembly.
Raw originals and per-file gzip copies remain locally available but are excluded
from Git, along with MachO executables, objects and private source/build trees.
The data capsule contains scientific output bytes, not executables. Code/source
archives and exact build commands support rebuilding executable artifacts.

After checkout, restore scientific bytes and optional source/header trees with:

```sh
python3 tst/elliptic_shift_3d/rehydrate_native_static_evidence.py docs/elliptic-shift-3d/evidence/native-static-smr-package-20261027 --sources
```

Use `--check-only` to verify the package without restoring files. Frozen MachO
binaries are deliberately omitted; rebuild the native target from the recorded
source/configuration before replaying its commands. The final shared control
can be repeated with `qualify_native_static_shared.py --output NEW_SHARED_CHECK`.
