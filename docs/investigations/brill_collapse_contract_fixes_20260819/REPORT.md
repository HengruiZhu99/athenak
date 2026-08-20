# Brill Cartoon Z4c contract fixes: CPU and Aurora qualification

Date: 2026-08-20  
Branch: `codex/brill-collapse-contract-fixes-20260819`  
Base: `1c95db8a2adc743672b49a525c21c4f762f35223`  
Current source: `d64b499c01aa4369bb6c37cfd11bc4c0590e69f2`  
Kokkos: `6739bc623081648af9e752b616d9671527922cbf` (4.7.2)

## Disposition

The completed work hardens and makes observable the known state, timestep,
AMR-transfer, high-frequency, and secondary AMR configuration contracts. A
fresh Aurora PVC one-tile build and zero-PDE AMR transaction also executed
successfully. This is **not** a Brill-collapse qualification, convergence
result, Figure-3 reproduction, or demonstrated production AMR-operator cure.

The exact cycle-1721 N256 Brill restart needed for the mandated bounded
pre-failure continuation has SHA-256
`83e996d2d5069307888a69fff47a7524c2f63f11869fb628630bca54dd5943ea`.
It is not present in this checkout or on Aurora; its historical Perlmutter
location must not be replaced with a different initial-data case. That bounded
physical check is therefore pending authenticated staging of those bytes.

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

The prior Aurora attempt, allocation `8767286`, failed during the SYCL build
before any device test because Aurora oneAPI 2025.3 llvm-spirv rejects the
`llvm.llround.i64.f64` intrinsic from the Cartoon MMS pgen. Its preserved build
log is `aurora_v1_build_failure/build.log`. The `d64b499c` compatibility change
removes that failure path; the succeeding job rebuilt the full application.

## What remains unproven

1. The required authentic N256 Brill restart is unavailable on Aurora, so no
   short physical continuation has run from this source.
2. Aurora head-node MPICH supplies compiler wrappers but no MPI launcher;
   consequently the two-rank ownership runtime test remains allocation-only.
   The single-rank ownership contract and CPU MPI compilation passed, but that
   is not a substitute for an actual multi-rank run.
3. The zero-PDE suite characterizes transfer symbols and selected transactions;
   it does not establish that a unique transfer source bug causes the nonlinear
   Brill runaway.
4. No claim is made that the high-k sensor should become a production AMR
   selector, that parent under-resolution is resolved, or that any physical
   result converges.

## Evidence files

- `aurora_v2/provenance.log`: device, module, source, and selector proof.
- `aurora_v2/configure.log`, `build.log`: fresh SYCL build proof.
- `aurora_v2/transfer-test.log`, `shadow-test.log`: focused device tests.
- `aurora_v2/zero-pde/`: stdout/stderr, shadow diagnostic, and T5 aggregate.
- `aurora_v2/SHA256SUMS`: remote phase checksum list.
- `aurora_v1_build_failure/`: preserved failed-build provenance.
- `EVIDENCE_MANIFEST.json`: strict hashes and scope labels for this report.
