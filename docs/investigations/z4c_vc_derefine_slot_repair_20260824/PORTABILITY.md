# Regression and portability record

## Host suite

The complete enabled local host suite passed:

```text
136/136 enabled tests passed
0 failed
2 CUDA-required tests disabled
398.14 s
```

`LastTest.log` SHA-256 is
`2b4e660e531b73d0f8964fe5526ce903d2dd852ee32459b8dc02dc01291e4976`.
The inventory SHA-256 is
`5d202e29fb49c09b86309613f55c61d9eeb5b65accc230f70936b903b25bd83c`.

This suite covers the new same-rank 2D/3D O2/O4/O6 and q4/q6/q8 cases,
constant data, mixed refinement/derefinement, existing VC transfer and Cartoon
axis paths, restart/output tests, and cell-centered lifecycle controls.

## Perlmutter MPI and CUDA

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
The source was the detached, clean production commit
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

Exact commands, logs, return codes, cache/executable hashes, and environment
metadata are retained under `evidence/regressions`.
