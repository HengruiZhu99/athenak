# Ref-GH relative-damped single-hole compact evidence

This directory is an incremental evidence checkpoint for branch
`codex/ref-gh-relative-damped-single-hole-20260830`.  It establishes local
implementation correctness and one evolved Aurora/PVC RK4 cycle.  It does not
establish stationary-trumpet stability, convergence, wormhole relaxation, or
spinning-hole robustness.

## Local gates

Source commit `5755ed30281f6094a9b9a53fa445da61d04a7e13` was built in fresh
Serial Release and ASan+UBSan/Kokkos-bounds configurations.  The complete
source-unit run passed without weakened tolerances, including:

- the new matched-state `D_A=0` and exact-zero-core oracle;
- direct derivative checks with maximum error `1.15917e-10`;
- direct-source versus ordinary-GH oracle error `3.46945e-18`;
- the existing 216-point coefficient oracle;
- the expanded 2160-point coefficient and moving-gauge oracles;
- the 2376-point generated-geometry oracle;
- the existing 4320-sample all-61 oracle;
- the production-cache oracle at conditioned error `1.63758e-14`;
- exact Minkowski evolution.

The full ASan+UBSan/Kokkos-bounds execution completed with status zero and no
sanitizer report.  Its executable SHA-256 was
`88b84fb93467aa4432f1c7db2c868cf4b3970329615d54d4295d3d979f1d03fb`.
The Serial executable SHA-256 was
`eb54bde511ae61c1ea0b6363cfbbc42ee8e4f8735f457c0c374989421cdfb523`.

Fresh one-cycle Serial controls used one `16^3` MeshBlock.  Both the pure
wave-map and relative-damped exact q=1 trumpets reached
`t=2.090302173081467e-2M`.  The relative case ended with field Linf
`1.665335e-15` and constraint Linf `1.007589e-14`; the wave-map case ended
with field Linf `1.665335e-15` and constraint Linf `1.007601e-14`.  The new
relative histories and RHS-sector decompositions are retained under `local/`.

## Aurora PVC gate

Job `8792239` was a harness-only failure: compute-node GitHub access timed out
before configuration or compilation.  Its complete log is under
`aurora_pvc_8792239_harness_failure/`.  The source-staging correction changed
no solver code.

Job `8792264` passed the required bounded gate at source commit
`a7d82a367eb94f367dd5fe977248172cdd516465`.  It used one Aurora node and 12
MPI ranks mapped to 12 distinct PVC tiles, with 216 MeshBlocks (18 per rank).
The Athena target was compiled and linked with
`-fsycl-device-code-split=per_kernel`; Kokkos selected SYCL/PVC and the
executable SHA-256 was
`e21dbd0c4c24677a41d89c0cfa26dc3aedbe0d8f3431439e5b96005890a7ba39`.

The exact q=1 relative-damped trumpet evolved one RK4 cycle to
`t=3.404497233383562e-3M`.  Final squared norms were GH
`3.943423625377059e-28`, reduction `4.576235990967554e-27`, and curl
`7.300487714932300e-33`.  Final relative diagnostics were
`|D|_inf=1.054711873393900e-15`, `|WD|_inf=1.054711873393900e-15`, and
added-source Linf `9.606691837577596e-15`.  Hhat, theta, and Upsilon RHS
families were exactly zero.  This is only a PVC portability result.

The full remote artifact and 273 MB executable remain at:

```text
/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_relative_damped_single_hole_20260830_pvc_v2
```

No Aurora jobs were active when this checkpoint was collected.
