# Brill Cartoon Z4c contract fixes: CPU and Aurora qualification

Date: 2026-08-20  
Branch: `codex/brill-collapse-contract-fixes-20260819`  
Base: `1c95db8a2adc743672b49a525c21c4f762f35223`  
Compiled physical-run source: `e8781f4057c73a0e97f5802413aefde899e24123`
Kokkos: `6739bc623081648af9e752b616d9671527922cbf` (4.7.2)

## Disposition

The completed work hardens and makes observable the known state, timestep,
AMR-transfer, high-frequency, and secondary AMR configuration contracts. A
fresh Aurora PVC one-tile build, zero-PDE AMR transaction, and user-authorized
from-scratch N256 Brill trajectory through `t=10 M` executed successfully.
This is **not** a Brill-collapse qualification, convergence result, Figure-3
reproduction, or demonstrated production AMR-operator cure.

The original exact cycle-1721 restart remains unavailable.  Rather than
substitute unrelated bytes, the physical run began from the archived direct
IrisK coefficient payload at `t=0`.  Its authenticated cycle-1800 restart was
continued to the physical target window.  The exact old parents 28 and 45 were
refined at cycle 2833, `t=9.476710063617325 M`.

## Fresh N256 physical event

Aurora job `8768636` executed the `t=0` segment and job `8768689` completed the
same trajectory at `t=10 M`; the latter requested one hour and used one PVC
tile.  The full compact report and evidence are under
[`aurora_scratch_n256/fresh_n256_physical_run_20260820/`](aurora_scratch_n256/fresh_n256_physical_run_20260820/).

At the target transaction, coordinate ring volume is conserved to `2.22e-15`
relatively, while T5/T0 proper-volume integral ratios are `3.047` for C,
`12.252` for H, `61.774` for M, and `1.00035` for Z.  The Cartoon measure is
already the proper ring measure, so this is not a collapsed-y normalization
artifact.  Transfer/phase ledgers close to roundoff and active fields do not
show a corrupt-copy jump.  The constraint change is present after boundary
reconstruction; algebraic projection contributes only about 1.40% of the
fixed-lattice stage-change sum.

The worst point lies at the equatorial MeshBlock edge, not the Cartoon axis.
Pre-event parent self-shadow is small for chi (`1.67e-4` relative L2) but large
for K (`0.353`) and Atilde (`0.280`), with edge bands much worse than interiors.
This supports a dchi-blind parent-resolution/edge-transfer mechanism but does
not uniquely distinguish parent under-resolution from an AMR-interface mode.

## Implemented, separately committed source contracts

| commit | scope | evidence |
|---|---|---|
| `b490b3b9` | Fail-closed Z4c state admissibility, including finite values, positive chi/lapse, SPD conformal metric, no determinant substitution, and checkpoint failure records. | CPU state-admissibility tests pass. |
| `cb61b64a` | Separate spatial and explicit-source timestep ceilings and diagnostic contract. | CPU timestep contract tests and short linear-wave smoke pass. |
| `71b07572` | Production-symbol Cartoon Z4c transfer qualification across tensor classes, O2/O4/O6, repeated transfer, and Fourier sweep. | CPU and Aurora device tests pass. |
| `34f9f620` | Optional, non-authoritative fourth-difference AMR high-k shadow sensor. | Synthetic CPU and Aurora device tests pass. |
| `e4da52e2` | State-manifold diagnostic fields in AMR-jump T5 aggregation. | CPU zero-PDE event serializes finite fields. |
| `0a740714`, `e55685f2` | Exact radial AABB distance plus AMR cadence/tracker validation. | Focused CPU tests pass. |
| `d64b499c` | Aurora SYCL compatibility: replace device-lambda `llround` in the Cartoon MMS pgen with the equivalent half-away-from-zero lattice helper. | CPU MMS and Aurora full SYCL build pass. |

The production high-order AMR operator is unchanged. No chi floor, clipping,
determinant substitution, positivity-gate weakening, gauge/damping/KO/CFL
retuning, or arbitrary limited-O2 replacement was made.

## CPU-head-node checks

CPU runtime checks used one OpenMP thread and at most two CTest workers.
Builds used at most 64 workers. The targeted contract suite passed all 11
selected tests. A serial zero-PDE event emitted both the shadow CSV and the T5
state-manifold aggregate:

```text
min_lapse                     = 0.04379562207510142
min_conformal_pivot           = 0.9320445624929372
max_abs_detgtilde_minus_one   = 7.771561172376096e-16
max_abs_atilde_trace          = 7.18283939271647e-19
```

This is a small synthetic zero-PDE transfer event, not physical Brill evidence.

An additional 17-test CPU-head-node suite also passed under the same runtime
cap. Its log is `cpu_head/contract_cpu_supported_ctest.log` (SHA-256
`7ada4b2bc2b2a4022f0cd5a35fd8a0c42eb41c6c416c8b3920a7a94d84e2ffb2`).
The deliberately broader 20-test audit is preserved separately in
`cpu_head/contract_cpu_extended_ctest.log` (SHA-256
`620df2fbf3c7c48dc28161fd469b61e744cebdc622d346d3a143508323c3e2cd`):

- 17 selected supported tests passed;
- `athena.z4c_cartoon_mms_static` could not run because the Aurora Python
  module lacks SymPy;
- the serial build correctly has no MPI-only coarse-cache ownership executable;
- the oneAPI serial build's `athena.z4c_cartoon_mms_structure` reports one
  bitwise Cartesian-delegation/tensor-variance mismatch at `rho=0.5`.

The mismatch is compiler-path-specific rather than evidence against the
current branch: the test source and `src/z4c/cartoon_derivatives.hpp` are
byte-identical to the frozen base, and the same single-threaded target passes
with Aurora GCC 13.4 (`cpu_head/contract_cpu_gcc_mms_structure.log`, SHA-256
`39b668a57ee36e67c35ba5718b9c23dfa11326b4ae6d25c1819b0756c8bd080d`).
The oneAPI bitwise discrepancy is retained as an unresolved toolchain/test
reproducibility issue, not suppressed or reclassified as a Z4c source defect.

The expanded CPU build was made from exact revision
`828ee0ab0b30c8c08de3acb55102d398ec78e371`, clean tree
`f7a0d222b7f08c035a1cf4c2a783c2d4ff8f8154`; all changes between that revision
and source-bearing commit `d64b499c` are documentation/evidence only. The
compiler/cache and focused executable hashes are recorded in
`cpu_head/contract_cpu_reproducibility.log` (SHA-256
`ba8c17ed34ed7d2d3eca3a4c2980b7406955d18b4a839d563be5860725cf182b`).

## Aurora one-tile device qualification

Aurora allocation `8768479` used account `CompactBinaryMerger`, debug queue,
one allocated node, and an application process pinned to PVC tile `0.0` with
`OMP_NUM_THREADS=1` and `KOKKOS_NUM_THREADS=1`. Aurora allocates whole nodes;
the process did not use the other tiles. The job completed with exit status 0
after 7m21s.

The actual device was `Intel Data Center GPU Max 1550` through Unified Runtime
Level Zero. The cache confirms `Kokkos_ENABLE_SYCL=ON`,
`Kokkos_ARCH_INTEL_PVC=ON`, `Kokkos_ENABLE_SERIAL=ON`, and MPI/OpenMP off for
this single-rank qualification. The application printed
`AthenaK Kokkos default execution space: SYCL`.

Both focused device tests passed:

```text
Cartoon Z4c AMR transfer qualification passed
Z4c AMR shadow sensor regression passed
```

The zero-PDE AMR transaction then completed T5 before the next RHS, with six
accepted created leaves, zero stderr, and a valid shadow CSV. Its source is the
built-in Kerr-puncture test input supplied by
`tst/inputs/z4c_amr_jump_diagnostic.athinput`; it is deliberately not a Brill
substitute.

## Aurora two-rank cache-ownership qualification

Aurora head-node MPICH does not expose a launcher without the PALS modules, so
the actual multi-rank check ran in the short debug allocation `8768526` under
`CompactBinaryMerger`. It used the same compiled CPU-MPI target and exactly two
ranks with one runtime thread each. The test log contains exactly two success
records:

```text
Z4c coarse-cache ownership regression passed
Z4c coarse-cache ownership regression passed
```

The wrapper's SHA-256 manifest is `SHA256SUMS` in `aurora_mpi_v2/`. This
qualifies the targeted two-rank ownership model only; it is not a multi-rank
physical Brill continuation or an end-to-end MPI AMR campaign.

The first wrapper attempt, allocation `8768523`, reached the same two passing
rank records but returned failure because the wrapper grepped a stale success
phrase. That phase is preserved under `aurora_mpi_v1_failure/`; the second
wrapper verifies the actual two-record result strictly.

The prior Aurora attempt, allocation `8767286`, failed during the SYCL build
before any device test because Aurora oneAPI 2025.3 llvm-spirv rejects the
`llvm.llround.i64.f64` intrinsic from the Cartoon MMS pgen. Its preserved build
log is `aurora_v1_build_failure/build.log`. The `d64b499c` compatibility change
removes that failure path; the succeeding job rebuilt the full application.

## What remains unproven

1. The original cycle-1721 restart remains unavailable; the completed physical
   check is instead a checksum-bound from-scratch trajectory.
2. The zero-PDE and physical-event suites characterize transfer symbols and
   selected transactions;
   it does not establish that a unique transfer source bug causes the nonlinear
   Brill runaway.
3. No claim is made that the high-k sensor should become a production AMR
   selector, that parent under-resolution is resolved, or that any physical
   result converges.

## Phase status and numerical-contract matrix

| phase / contract | status | exact scope and evidence |
|---|---|---|
| Phase 0 baseline freeze | QUALIFIED | `dd149c13`; baseline static logs retained under `baseline/`. The expanded CPU audit adds 17 supported passes and a separately classified oneAPI-only MMS bitwise discrepancy. |
| Fail-open AlgConstr determinant/SPD | FIXED | `b490b3b9`; invalid determinant, indefinite metric, nonfinite values, nonpositive chi/lapse, and unmodified-failure paths are covered by the state-admissibility test. |
| Active-stage state admissibility | QUALIFIED | `b490b3b9`; deterministic checkpoint/failure machinery plus synthetic zero-PDE T5 manifold diagnostics. A physical first failure is pending the authenticated restart. |
| Explicit gauge/source timestep | FIXED | `cb61b64a`; distinct `dt_spatial`/`dt_source`, source-rate diagnostics, and focused contract tests. |
| Gauge characteristic CFL | QUALIFIED | `cb61b64a`; characteristic/source contract code and flat/parameterized unit coverage; no physical Brill activation has been measured. |
| AMR coarse-cache provenance | QUALIFIED | Existing ownership fix retained; zero-PDE transaction evidence and strict two-rank Aurora ownership test (`8768526`) pass. |
| Block-edge restriction | STILL OPEN | Production operator was intentionally not changed; symbol/decomposition qualification measures it but does not identify a unique cure. |
| Refine/restrict high-k amplification | QUALIFIED | `71b07572` Fourier transfer sweep and device transfer target expose, rather than hide, high-k gain. |
| Coarse-fine derivative compatibility | QUALIFIED | Zero-PDE transfer/interface suite and same-rank/two-rank ownership scope; not a nonlinear Brill validation. |
| dchi Nyquist blindness | QUALIFIED | `34f9f620` adds a non-authoritative fourth-difference shadow sensor; legacy AMR selection remains unchanged. |
| Transfer metric/state admissibility | QUALIFIED | `e4da52e2` records chi/lapse/SPD/determinant/trace diagnostics at T5; no nonlinear transfer redesign was justified. |
| Axis regularity | QUALIFIED | Cartoon parity/boundary and regular-functional tests pass in the supported suite; GCC 13.4 also passes the MMS structure target. |
| RefineRadii geometry | FIXED | `0a740714`; exact AABB distance tests cover inside/face/edge/corner/half-plane cases. |
| AMR cadence parsing | FIXED | `e55685f2`; positive integer cadence validation is unit-tested. |
| Empty tracker handling | FIXED | `e55685f2`; configuration fails before empty-tracker use. |
| Phase 7 bounded Brill execution | QUALIFIED FOR DIAGNOSIS | User-authorized from-scratch N256 trajectory reaches the same physical parent pair at cycle 2833, `t=9.476710063617325 M`, and completes at `t=10 M`.  The event is quantified but does not qualify convergence or a production cure. |

## Reproduction commands

The Aurora one-tile and two-rank PBS scripts are committed beside the evidence
and are the authoritative scheduler commands. The CPU-head-node suite used
one runtime thread and two CTest workers:

```bash
OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1 \
ctest --test-dir /home/hzhu/athenak-brill-contract-fixes-20260819-cpu-compat \
  -j2 --output-on-failure \
  -R 'athena\.(amr_cadence|amr_history_format|z4c_cartoon_mms_generated_reference|z4c_cartoon_axis_parity|z4c_cartoon_axis_boundary|z4c_cartoon_axis_centered_derivatives|z4c_cartoon_regular_functionals|z4c_cartoon_amr_transfer_qualification|z4c_cartoon_lattice_index|z4c_cartoon_m0_fastflow|z4c_chi_prolongation|z4c_amr_jump_diagnostic|z4c_amr_radius|z4c_amr_shadow_sensor|z4c_state_admissibility|z4c_timestep_contract|z4c_amr_configuration_static)$'
```

Builds used `cmake --build ... --parallel 64` only, as permitted for
compilation. The exact build cache, source/tree/Kokkos identities, and hashes
of the principal focused executables are in
`cpu_head/contract_cpu_reproducibility.log`; each runtime log names its test
command/result and is checksum-bound in the manifest.

## Evidence files

- `aurora_v2/provenance.log`: device, module, source, and selector proof.
- `aurora_v2/configure.log`, `build.log`: fresh SYCL build proof.
- `aurora_v2/transfer-test.log`, `shadow-test.log`: focused device tests.
- `aurora_v2/zero-pde/`: stdout/stderr, shadow diagnostic, and T5 aggregate.
- `aurora_v2/SHA256SUMS`: remote phase checksum list.
- `aurora_v1_build_failure/`: preserved failed-build provenance.
- `aurora_mpi_v2/`: strict two-rank cache-ownership evidence.
- `aurora_mpi_v1_failure/`: preserved wrapper-only predecessor failure.
- `cpu_head/`: supported passing suite and the broader, explicitly classified
  CPU audit, including the GCC cross-compiler MMS result.
- `EVIDENCE_MANIFEST.json`: strict hashes and scope labels for this report.
- `aurora_scratch_n256/fresh_n256_physical_run_20260820/`: head-build,
  from-scratch/continuation logs, combined history, target-event writer ledger,
  parent-state audit, plots, and strict summary.
