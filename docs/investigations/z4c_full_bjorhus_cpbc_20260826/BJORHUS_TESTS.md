# Full-field Z4c Bjorhus test record

## Provenance

- Base commit: `c4a31dd68e3cd92ff47839a8f93d1be44c09399a`
- Implementation branch: `codex/z4c-full-bjorhus-cpbc-20260826`
- Read-only mathematical/architectural reference:
  `origin/codex/z4c-characteristic-cpbc` at
  `575b85d7175969e2f28b93d5aa14d99507477c36`
- Execution environment: local CPU, Kokkos Serial, GNU 13.3.0

No production Brill evolution was run.

## Manufactured algebra tests

Target:

```text
athena_z4c_full_constraint_bjorhus_unit_test
```

The test covers:

1. a manufactured outgoing Theta/Z pulse represented by arbitrary outgoing
   rate data and zero incoming projection: the correction and induced outgoing
   change are exactly zero within tolerance;
2. a manufactured incoming Theta/Z pulse: all four corrected incoming rows
   vanish within `2e-13` relative tolerance;
3. the same incoming pulse's induced outgoing-rate change, which is required to
   be nonzero so the sparse-projection limitation cannot regress out of the
   evidence;
4. a composite rho-z corner normal and deterministic x1-before-x2 ownership;
5. a Cartoon axis/z-boundary intersection, which is rejected by the CPBC owner;
6. invalid `chi` and an indefinite conformal metric, which fail closed;
7. a deterministic 257-sample correction checksum.

Observed serial output:

```text
full constraint Bjorhus manufactured tests passed
checksum=1074739016429263659
induced_outgoing_rate=0.686111
```

CTest result:

```text
athena.z4c_full_constraint_bjorhus ... Passed
100% tests passed, 0 tests failed out of 1
```

The `0.686111` value is an induced outgoing characteristic-**rate** projection
for the fixed manufactured pulse. It is not a physical reflection coefficient.

## Production compile and bounded smoke test

Serial configuration:

```bash
cmake -S . -B build-cpbc-serial \
  -DPROBLEM=built_in_pgens -DBUILD_TESTING=ON \
  -DAthena_BUILD_UNIT_TESTS=ON -DAthena_ENABLE_MPI=OFF \
  -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=OFF \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build-cpbc-serial --target \
  athena_z4c_full_constraint_bjorhus_unit_test athena -j16
```

Both targets built successfully. The bounded input
`tst/inputs/z4c_vc_minkowski_full_constraint_bjorhus.athinput` exercised the
native vertex-centered Cartoon path on four MeshBlocks with an axis, outer rho
face, both z faces, and rho-z corners. It reached its requested `t=0.01` limit
cleanly in two cycles.

This is a wiring/smoke result only. It does not measure boundary convergence,
reflection, or nonlinear stability.

An override with `z4c/boundary_rhs=invalid_choice` exited with status 1 and the
expected fatal selector error, confirming that unknown modes fail closed.

## Neighboring regression tests

After building their targets, these existing serial tests passed:

```text
athena.z4c_cartoon_axis_boundary
athena.z4c_vc_cartoon_axis_rhs_regularity
athena.z4c_sommerfeld_derivatives
athena.z4c_symmetry_validation
```

Result: `100% tests passed, 0 tests failed out of 4`.

An earlier CTest attempt before building these `EXCLUDE_FROM_ALL` targets
reported them as `Not Run`; that was missing test executables, not a test
failure. They were then built explicitly and rerun as above.

## MPI compile and runtime limitation

The MPI-enabled configuration found OpenMPI 3.1 and both the manufactured test
target and the complete `athena` target compiled and linked successfully.

MPI execution could not be qualified on this host. The local OpenMPI runtime is
already unhealthy: many unrelated, days-old `orted` processes are stuck in
uninterruptible `D` state, and even the pre-existing diagnostic command
`mpiexec -n 2 /bin/hostname` is stuck. Both singleton and `mpiexec` attempts of
the new test consequently hung before producing test output. No MPI pass is
claimed. The test contains a two-rank min/max checksum reduction and CMake
registers `athena.z4c_full_constraint_bjorhus_mpi2`; it still needs execution in
a healthy MPI environment.

## Remaining limitations

- The closure cancels four incoming principal constraint rates but generically
  perturbs paired outgoing rates; see `BJORHUS_DERIVATION.md`.
- The outgoing/incoming pulse checks are manufactured principal-symbol tests,
  not evolved pulse-reflection measurements.
- Same-rank and cross-rank duplicated native-VC boundary values were not
  compared in a healthy MPI run.
- Cell-centered and Cartesian 3D templates compile, but only the primary native
  VC Cartoon path received a bounded runtime smoke test.
- No nonlinear Brill campaign, boundary-distance convergence study, or
  production qualification was performed.
