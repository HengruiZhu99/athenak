# Phase 6 rank-packed same-level contract

Status: local correction and validation passed; Aurora PVC gates pending.

## Proven mismatch

Before this correction, `MeshBoundaryValues::GetVarDataSize()` selected
`isame_z4c_ndat` for every same-level message whenever `is_z4c_` was true.
Both same-level CC pack and unpack append the coarse payload only when all of
the following hold:

```text
neighbor level == local level && is_z4c_ && pmesh->multilevel
```

Therefore uniform-grid Ref-GH/Z4c-like rank-packed metadata advertised
`nvars * isame_z4c_ndat` elements while the kernels wrote/read only
`nvars * isame_ndat`. The extra interval was uninitialized payload.

The correction centralizes the same-level count in
`RankPackedSameLevelVarDataSize()` and makes metadata use the exact pack
condition. It does not alter any PDE, stencil, gauge parameter, task ordering,
physical boundary value, or true-multilevel Z4c payload.

Because the old W source completed positive time with this pre-existing
inconsistency, this evidence does not classify it as the W-to-R regression
root cause.

## Truth table

| Z4c-like | Multilevel | Metadata and pack payload |
| --- | --- | --- |
| false | false | `nvars * isame_ndat` |
| false | true | `nvars * isame_ndat` |
| true | false | `nvars * isame_ndat` |
| true | true | `nvars * isame_z4c_ndat` |

## Fresh local validation

All commands ran from the repair worktree on 2026-08-30. Build directories
are intentionally untracked.

| Gate | Configuration | Result |
| --- | --- | --- |
| Compile | Release, MPI ON, Kokkos Serial, source unit ON | PASS |
| Ref-GH source unit | one-rank MPI singleton | PASS |
| q-controlled/all-61 oracle | Release, OpenMP, expanded radial matrix | PASS; all 4320 RHS samples |
| Z4c uniform communication | two MPI ranks, two blocks, one evolved cycle | PASS at `t=0.01875` |
| Z4c AMR communication | two MPI ranks, four initial blocks, one evolved cycle | PASS; refined to 32 blocks |
| Ref-GH uniform communication | two MPI ranks, two blocks, one fully-subtracted RK4 cycle | PASS at `t=0.01` |
| Sanitizers | ASan + UBSan + Kokkos bounds, full q-controlled/all-61 source unit | PASS |

The q-controlled source oracle retained its binary64-conditioned maxima:

```text
expanded analytic coefficients: 1.48837e-13 (2160 samples)
generated geometry:              2.33147e-15 (2376 samples)
moving gauge:                    1.24829e-14 motion maximum (2160 samples)
all-61 RHS:                      4.13003e-14 (4320 samples)
production cache:                1.63758e-14
```

The sanitizer invocation used
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1`,
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`, and
`Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON`.

Local OpenMPI's hwloc OpenCL component initially entered an Intel DRM cleanup
wait before launching AthenaK. The successful MPI tests explicitly used
`HWLOC_COMPONENTS=-opencl`; this changes launcher discovery only and is
unrelated to the Aurora PVC executable or its Level Zero backend.

