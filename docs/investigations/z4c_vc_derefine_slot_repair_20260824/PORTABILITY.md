# Regression and portability record

## Host suite

The freshly rebuilt final-source local host suite passed:

```text
137/137 enabled tests passed
0 failed
2 CUDA-required tests disabled
387.33 s
```

The source was `6dd20656a305f2543bbbd7001550c6ac67019180`.
`LastTest_final_6dd.log` SHA-256 is
`f309bc00a4af92744788be2b7d62187ff50f84d180f15658f23d6d75c272f316`.
The final inventory SHA-256 is
`23cae87dbe92e0e1fd5ccb6d8d150ec843a1caabb8d13a4cb6a63d7ab84abd0a`.

This suite covers the new same-rank 2D/3D O2/O4/O6 and q4/q6/q8 cases,
constant data, mixed refinement/derefinement, existing VC transfer and Cartoon
axis paths, restart/output tests, and cell-centered lifecycle controls.

## Perlmutter MPI and CUDA

Final-source job `57504956` used one shared A100 and completed with status 0
in 67 seconds. It passed:

- the CPU MPI4 dual-split ownership test;
- CUDA same-rank dynamic AMR, eight multi-family 2D/3D cases, and the writer
  lifecycle test;
- CUDA MPI2/MPI4 local-lower, remote-lower, mixed-migration, and dual-split
  ownership tests;
- CUDA O2/O4/O6 three-dimensional multi-family cases;
- both production Cartoon kernel tests.

All 20 recorded test statuses are zero. The bundle manifest SHA-256 is
`e655a842435ecf532e2b31901f3b3d91cc75fbccb4cb5d1a72f1ac9b430e999f`.
The final-source executables are:

```text
CPU MPI/OpenMP  dde704e91579727b6004efd1b7693bc2411f9785435d5c1e11d425d4809eeb8b
CUDA MPI        7f5bd2fc5d05233822c583ec677e91867eaf6f0c8e2d7ca521cfba93c091a55b
```

The source tree was
`551b16fab36ec1d4ce913b39a6478384723aa382`, with Kokkos
`6739bc623081648af9e752b616d9671527922cbf`.

The broader Perlmutter attempt in job `57504412` was deliberately retained
despite returning 1. Its actionable dual-split fixture failure was corrected
by commit `6dd20656`; the repaired CPU and CUDA tests then passed in job
`57504956`. The other full-suite failures were environment/test-harness
limitations: wrong SymPy version, a two-rank test with only one root block, a
shared-node timeout, and a compiler-dependent exact-host fingerprint. The
same exact-host fingerprint test passes in the complete local Serial suite.
Several large 3D CUDA tests also encountered shared-GPU memory contention in
job `57504412`; all required 3D cases passed when rerun separately in job
`57504956`.

### Earlier production-repair qualification

Job `57502832` used one A100 in `shared_interactive` and completed with status
0 in 76 seconds.

- split-family pure derefinement at MPI2 and MPI4: 2/2 passed;
- split-family mixed refinement/derefinement at MPI2 and MPI4: 2/2 passed;
- CUDA same-rank multi-family matrix (tests 26--33): 8/8 passed;
- CUDA cell-centered lifecycle controls (2D Cartesian, 2D Cartoon, 3D
  Cartesian): 3/3 passed;
- production Cartoon kernel and CUDA-required kernel: 2/2 passed.

The CUDA executable SHA-256 is
`d08ea0ca307416a2735b58c77eb61c1e59a7119909c60bed7ed1094746863898`;
the CPU MPI executable SHA-256 is
`3c794967aa8ef70ea99d5ff379698d36377e402007345563d164ec2997a31a90`.
The source was the detached, clean production-repair commit
`d2596707e808aea7ec6167df937d71dc4dbe429e` with Kokkos
`6739bc623081648af9e752b616d9671527922cbf`.

The first qualification allocation, job `57502692`, failed before executing
the intended matrix because its allocation permitted only one task while the
tests requested nested MPI2/MPI4 steps, and because the CUDA CTest cache named
Perlmutter's obsolete `/usr/bin/python3`.  This is retained as a launcher and
environment setup failure, not counted as a source regression.  The corrected
runner reserved four task slots and substituted only the supported NERSC
Python 3.11 interpreter in CTest-generated Python commands.

## Cell-centered preservation

The production changes dispatch only for native vertex-centered Z4c.  The
three CUDA cell-centered lifecycle controls and the complete host suite passed
without tolerance changes.  This supports the required CC-unchanged boundary;
it is not a whole-application bitwise replay of every possible CC problem.

## SYCL

No real current-source SYCL runtime was available in this qualification, so no
SYCL claim is made.  Compilation alone is not treated as qualification.

Exact commands, logs, return codes, cache/executable hashes, device-memory
censuses, and environment metadata are retained under `evidence/regressions`.
