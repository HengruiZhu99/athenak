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

## Stationary progression through 5M

Aurora job `8792288` evolved the fresh 96^3 case through `t=3M` on 12 PVC
tiles.  It reached the endpoint with GH RMS `3.54051e-13`, reduction RMS
`1.69867e-12`, curl RMS `1.10104e-13`, metric-error RMS `3.29427e-11`,
relative-D Linf `1.30951e-12`, and relative-source Linf `1.03333e-11`.
The GH log fit over `t>=0.2M` had slope `0.369543/M` and `R^2=0.729736`, not
the old `26.6549/M` fast mode.

Job `8792431` performed an actual restart from the hashed `t=3M` checkpoint
and continued through `t=5M`.  It remained finite with GH RMS `5.01723e-13`,
metric-error RMS `4.83030e-11`, relative-D Linf `1.84064e-12`, and
relative-source Linf `1.40622e-11`.  Its segment fit had slope `0.180206/M`,
`R^2=0.979664`, and no old-mode recurrence.  The restart file paths and hashes
are recorded, but the multi-GB restart files are not committed.

Job `8792462` ran fresh 64^3, 96^3, and 128^3 cases independently from
`t=0` through `t=5M`.  All three stayed finite and passed the old-mode
discriminator.  The absolute errors are roundoff-scale but are **not
resolution-improving**:

| Resolution | GH RMS | Metric-error RMS | Relative-source Linf |
| --- | ---: | ---: | ---: |
| 64^3 | `3.37076e-13` | `1.65407e-11` | `9.22505e-12` |
| 96^3 | `5.00964e-13` | `4.82665e-11` | `1.40252e-11` |
| 128^3 | `6.56073e-13` | `1.06940e-10` | `1.86723e-11` |

For example, the signed GH orders are `-0.977` from 64 to 96 and `-0.938`
from 96 to 128.  This evidence supports finite stationary evolution without
the former fast inner instability through 5M; it does not support a positive
stationary convergence claim.  `convergence.json` and `convergence.tsv`
retain all signed ratios and orders.  Histories are downsampled to roughly
0.1M spacing in Git; the full histories and maximum-location tables remain
under the recorded Aurora job directories.

## Stationary 96^3 extension through 20M

Aurora job `8792505` independently evolved the 96^3 stationary exact-reference
case from `t=0` through `t=20M` on 12 PVC tiles.  It exited normally, stayed
finite, recorded no bad state, and did not reproduce the former
`26.6549/M` fast mode.  At `t=20M`, GH RMS was `3.46274e-12`, reduction
RMS `2.69229e-12`, curl RMS `1.46371e-12`, metric-error RMS
`4.85810e-10`, relative-D Linf `1.42217e-11`, and relative-source Linf
`4.95378e-11`.  The GH log fit over `t>=0.2M` had slope `0.149669/M`,
e-folding time `6.68139M`, and `R^2=0.953579`.

This is a single-resolution exact-stationary extension, not a convergence or
long-time trumpet-relaxation result.  The compact history, complete run log,
analysis summary, provenance, rank/tile mapping, remote hashes, and restart
hashes are retained under `stationary_t20_8792505/`.  The five large restart
files and full histories remain in the Aurora campaign directory and are not
committed.

## Wormhole-to-trumpet local implementation gate

The mature `controlled_transition` problem generator was configured with
genuine isotropic Schwarzschild wormhole data exactly matched to the initial
wormhole reference, followed by its existing smooth `shrinking_width`
reference transition.  The new controlling inputs retain STANDARD Phi,
`gamma0=gamma2=1`, FD4, RK4, CFL 0.05, and KO 0.02.

A deliberately reduced Kokkos-Serial case evolved one complete RK4 cycle to
`t=0.0460447286M`.  The exact initial match had zero state and GH mismatch,
lapse-ratio error `2.22e-16`, zero shift mismatch, and no cell at the
puncture.  The evolved state remained finite; Hhat, theta, and Upsilon RHS
families stayed exactly zero.  At the endpoint, GH RMS was `2.25466e-7`,
reduction RMS `8.44052e-7`, curl RMS `2.45349e-7`, relative-D Linf
`1.95055e-6`, and added-source Linf `2.75766e-5`.

This is only a local implementation and output-schema gate.  Its resolution is
too low and its duration too short to demonstrate physical lapse collapse,
trumpet settling, convergence, or the absence of a later instability.  Those
questions require the bounded Aurora progression.
