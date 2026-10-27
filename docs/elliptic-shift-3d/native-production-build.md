# Production AthenaK compilation compatibility

The actual production `athena` target compiled and linked successfully on
this Mac, including native Z4c translation units that include the refactored
regular ConformalRHS and RegularPenrose helpers. This checks compilation and
link compatibility of the common tensor kernel. It does **not** establish
production integration of the composite elliptic shift solver, native shell
state/task execution, or any scientific evolution result.

```sh
cmake --build build-shift-3d --target athena --parallel 2
build-shift-3d/src/athena -h
```

The build exited0 after139.658 seconds with an empty compiler stderr log.
It used two CPU build workers and completed within the explicit180-second
bound. The first90-second checkpoint remained active; no termination signal
was needed. The build ran in its own newly created session/process group10003.
No existing process or job was signaled. Only the `athena` target was requested;
no pilot/CTest target or source CMake/header file was edited by this subtask.

The linked binary SHA256 is
`78d6bbf17f943535d5cdf599829591cef3b8c0eba30121948458985d91637338`.
Its read-only `-h` invocation exited0 and reported Athena v0.1, double precision,
built-in problem generators, MPI disabled and OpenMP disabled. No input file,
restart, mesh generation or timestep was requested. This is a Serial Kokkos
CPU compatibility check; no GPU/MPI execution or scalable performance claim
is attached to it.

The starting repository HEAD was
`9cced4c72698e9e2a37c4726d139bbb51e877e6d`, with the dirty status and exact
source hashes recorded before the build. Concurrent parent-owned work changed
`src/multigrid/composite_poisson.hpp`; the manifest preserves that change.
That header is absent from the production target's compiler dependency files,
so its concurrent edit is not evidence of native composite-solver integration.
All tracked repository dependencies actually referenced by the production
objects remained unchanged during this check. Source/config copies from that
tracked dependency set are archived for review; vendored/system dependencies
are not described as a fully frozen toolchain.

Compiler dependency files for `z4c.cpp.o`, `z4c_hyperboloidal.cpp.o` and
`z4c_calcrhs.cpp.o` explicitly include both `conformal_rhs.hpp` and
`regular_penrose.hpp`. The evidence stores those dependency files, the object
hashes, generated config, compile/link flags, complete output, build command,
process bound, linked-binary hash and usage output under
`evidence/native-production-build-20261027/`. `native-kernel-dependencies.json`
distinguishes actual target dependencies from the unrelated concurrent edit.

The reusable `tst/elliptic_shift_3d/build_native_compatibility.py` reproduces the
bounded build/usage check into a fresh output directory. Its timeout handler
signals only the process group it created, and records any denied or uncertain
cleanup rather than signaling other jobs. No timeout path was exercised in
this successful build.
