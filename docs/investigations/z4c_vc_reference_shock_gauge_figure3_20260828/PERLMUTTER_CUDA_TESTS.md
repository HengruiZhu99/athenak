# Perlmutter CUDA qualification

## Source and build authority

The fresh Perlmutter checkout is:

```text
/pscratch/sd/h/hzhu/z4c-vc-reference-shock-gauge-figure3-20260828/source
commit: 00e66dfa5e0e7f7f2c711166998806a05decbd55
tree:   578bfd4ede74ddc7464f82e4c7b76f4111a4ad76
Kokkos: 6739bc623081648af9e752b616d9671527922cbf
```

The checkout was clean when configured and after the build. The fresh build
directory is `build-cuda-mpi` with:

```text
CMAKE_BUILD_TYPE=Release
CMAKE_CXX_COMPILER=CC
Athena_ENABLE_MPI=ON
Athena_BUILD_UNIT_TESTS=ON
Athena_ENABLE_IRISK_INTERPOLATOR=ON
Kokkos_ENABLE_CUDA=ON
Kokkos_ENABLE_SERIAL=ON
Kokkos_ENABLE_OPENMP=OFF
Kokkos_ARCH_AMPERE80=ON
PROBLEM=built_in_pgens
```

The exact IrisK interpolator archive is
`d4afad6d3a20a8dd8197eb7d70d5a23903a7e2401a5d8b034d32005bf07f3f39`.
The first configure attempt failed before compilation because it supplied the
archive but not the IrisK header root. Supplying both `IRISK_ROOT` and
`IRISK_INTERPOLATOR_LIBRARY` corrected the configuration without changing
source. The full CUDA AthenaK target then built successfully with `-j64`.

```text
Athena executable SHA-256:
c5153a5c25c4b2aba22737061baa00628badcd6eade2c66461ac182708677e55

CMakeCache.txt SHA-256:
273fe5f79f18d39f19748aff573c1ac204ebe9aae14f61533ca99c42a3c13f8a
```

## Runtime qualification gate

Status: `UNEXECUTED_PARTITION_DOWN`

The focused runtime bundle covers:

- shock-avoiding lapse production smoke and source contract;
- finite negative-lapse admissibility policy;
- timestep characteristic-speed contract;
- prescribed-zero shift source and exact runtime invariant;
- native VC Cartoon production kernels with `--require-cuda`;
- O4 dynamic/static multilevel VC Cartoon paths;
- native VC AMR history record/replay/restart/2x-cell test;
- Cartoon AMR transfer qualification.

Final scheduler disposition on 2026-08-28:

- `shared_interactive` rejected submission because its GPU partition was down;
- normal interactive allocation request `57670453` reported `Partition in DOWN
  state` and was intentionally revoked before allocation;
- debug qualification job `57670461` remained pending with reason
  `PartitionDown` and was cancelled at wrap-up. Final accounting reports
  `CANCELLED by 102811` and elapsed time `00:00:00`.

Neither allocation consumed GPU time. The CUDA binary was built but never
executed on an A100, so build success is not runtime qualification. N256
production was not submitted and remains forbidden until the runtime bundle
writes `PERLMUTTER_CUDA_QUALIFICATION_PASS` and its evidence is inspected.

The exact terminal scheduler evidence is frozen in
`PERLMUTTER_SCHEDULER_DISPOSITION.txt`.
