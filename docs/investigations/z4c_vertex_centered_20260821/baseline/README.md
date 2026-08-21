# Cell-centered regression baseline

This baseline was captured before vertex-centered production edits.

- AthenaK commit: `6daa774d7451dbc5f7cac640c6e32a6fd11de7f9`
- AthenaK tree: `cbb702f4da954cf630da261790d5c21ef3142235`
- Kokkos commit: `6739bc623081648af9e752b616d9671527922cbf`
- Kokkos tree: `052310d99e85c4cfd72b6f561659c6b0a2aa19f3`
- Compiler: GNU C++ 13.3.0
- MPI: OpenMPI wrapper using `g++`
- CMake: 3.28.3
- Kokkos backend: OpenMP
- Problem generator: `built_in_pgens`
- Precision: double
- MPI: enabled
- OpenMP: enabled
- IrisK interpolator: enabled
- Executable SHA-256: `7a65f7be074ffa5fb006f2b4d8ddd00ed993d4d9470c2b22083e5e0a4a6044aa`
- CMake cache SHA-256: `4aff7c1166fdb53ae4bc697c3c9682ac4c69792cf28fecaa71db6fd6f27103c9`

The fresh build registered 65 tests. All 63 enabled tests passed in 184.17 s
with `OMP_NUM_THREADS=1 ctest -j16 --output-on-failure`. The two
CUDA-required tests were disabled in this CPU build:

- `athena.z4c_cartoon_production_kernels_cuda_required`
- `athena.pdf_scatter_cuda_required`

This establishes the pre-selector host/MPI CC red-green baseline. CUDA/SYCL
baselines remain open until their dedicated qualification phases.

## Implicit versus explicit cell selector

After adding only the immutable selector and index-geometry layer, the same
one-cycle Kerr half-plane input was run twice with one OpenMP thread:

- the repository input with `grid_centering` absent;
- an otherwise byte-identical input containing `grid_centering = cell`.

Both executions used the rebuilt executable from the same source tree. The
history file and timestep-contract CSV were byte-identical:

- post-selector executable SHA-256: `e6c1500f8c67c97b0538c876b16fc14bf1fca6a8330d9ae3983e8697c673f402`

- history SHA-256: `4896c333ceda81d99cf1e4c15a28996d73c999c6222d4b83e770c9f4f4d0f598`
- timestep-contract SHA-256: `dad954f5938eea76aca74493ec5bd1ac8c66cdc67ac7ad24225988c19e5e3037`

Athena binary files include the input deck in their header, so their whole-file
lengths differ by the added selector line. The repository binary reader was
used to compare time, cycle, variable inventory, MeshBlock indices, logical
locations, geometry, and every numerical array with exact equality. Canonical
numerical-payload SHA-256 values were identical for both runs:

| payload | SHA-256 |
|---|---|
| constraints cycle 0 | `0b34b0c1886ca18622058aa1f42d8953bf11ed194cb302005e769e77f3db970b` |
| constraints cycle 1/final | `c0ba4a0da9982e78bfd5665f9421b48b2b019110171f0f30b24deae729ace2e3` |
| Z4c cycle 0 | `b5f8dd57e2902fe18d4af8ac1c27f35124379292f389197323dd2ae144b0166a` |
| Z4c cycle 1/final | `8058fed583f1eca73ad9b4d54fe2171a634214b93f440561720c3367cfc92fea` |

The post-selector host/MPI suite registered 68 tests. All 66 enabled tests
passed in 183.77 s; the same two CUDA-required tests remained disabled. The
three new tests were `athena.z4c_vc_layout`, `athena.z4c_vc_coordinates`, and
`athena.z4c_vc_collapsed_dimension`.

As a transitional fail-closed check, `grid_centering=vertex` is admitted by the
allocation-free schema validator, allocates and verifies true native `N+1`
Z4c arrays, and then rejects before any cell-centered boundary object can be
attached. It exits 1 and produces no output files. This boundary-seam guard is
temporary and must be removed only when deterministic VC communication is
complete.

## Native-kernel centering dispatch checkpoint

After host-selected centering templates were wired through the RHS, ADM
initialization/constraints, curvature, timestep, AMR sensor, Weyl, and
Sommerfeld paths, all 72 enabled host/MPI tests passed in 180.50 s. The two
CUDA-required tests remained disabled in this OpenMP build. The one-cycle CC
reference remained byte-identical to Phase 0:

- history SHA-256: `4896c333ceda81d99cf1e4c15a28996d73c999c6222d4b83e770c9f4f4d0f598`
- timestep-contract SHA-256: `dad954f5938eea76aca74493ec5bd1ac8c66cdc67ac7ad24225988c19e5e3037`
- checkpoint executable SHA-256: `e5eeff30233d2bfc6adcf1e0f7f5375579f38084407e1990df1fad9c770ef88a`

This checkpoint proves the CC arithmetic and output payload remained frozen;
it does not qualify VC communication, AMR, restart, or evolution.

## Evolved Cartoon-axis checkpoint

The native VC axis now has an explicit production contract: the rho=0 vertex
is never overwritten by parity reconstruction; negative-rho ghosts mirror the
corresponding positive vertex; scalar/vector/tensor identities are enforced on
the active axis RHS before RK consumption and on active state immediately after
each RK update.  Nonfinite or materially large corrections fail visibly and a
nonzero correction is recorded with cycle, stage, component, GID, and z.

The four exact VC axis tests (`scalar`, `vector`, `tensor`, and
`rhs_regularity`) pass for ghost widths 2, 3, and 4 on the device execution
space.  Together with the six Cartesian/Cartoon derivative tests and legacy CC
axis test, all 11 focused tests pass.  The complete OpenMP/MPI suite registers
78 tests; all 76 enabled tests pass in 180.97 s and the two CUDA-required tests
remain disabled in this host build.

The post-axis executable SHA-256 is
`97715f7fa5d6d1a1f73827688ce906bd65d05ce81076117f6a8f0f77711f8c18`.
The frozen one-cycle CC reference remains byte-identical:

- history SHA-256: `4896c333ceda81d99cf1e4c15a28996d73c999c6222d4b83e5e0a4a6044aa`;
- timestep-contract SHA-256: `dad954f5938eea76aca74493ec5bd1ac8c66cdc67ac7ad24225988c19e5e3037`.

This still does not qualify VC communication or AMR; the constructor guard
continues to reject VC before attaching the legacy CC boundary object.

## Canonical vertex identity checkpoint

The communication-independent topology core uses an overflow-checked dyadic
integer key, never a floating coordinate.  It canonicalizes periodic upper
endpoints before promotion to the configured maximum level and assigns key zero
to collapsed directions.  A compact device record distinguishes independent,
same-level shared, coincident coarse-fine, hanging fine-interface, physical,
axis, and ghost nodes.  The focused test covers same-level and cross-level
identity, periodic wrapping, overflow rejection, every face/edge/corner subset
in 1D/2D/3D, and the collapsed Cartoon half-plane.

This checkpoint defines identity and role semantics only.  Building those
records from production neighbor metadata, deterministic contributor lists,
MPI synchronization, and AMR reconstruction remains pending.
