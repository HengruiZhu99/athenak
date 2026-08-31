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

Aurora job \`8792573\` did not launch Athena.  It failed in 18 seconds while
invoking the rank-to-tile mapping wrapper because the first committed PBS
script contained literal \`+\` tokens where shell line continuations were
intended.  The complete scheduler/provenance evidence is retained under
\`wormhole_short_8792573_harness_failure/\`.  The correction is mechanical
and changes neither the controlling input nor solver/formulation code.

Corrected Aurora job \`8792609\` passed the bounded two-resolution startup gate
on 12 distinct PVC tiles.  Fresh N96 and N144 cases both reached \`t=1M\`,
remained finite, recorded no bad state, and did not reproduce the old fast
mode.  At the endpoint, signed N96-to-N144 orders were 1.98 for GH, 0.24 for
reduction, and 1.56 for curl constraints.  The moving-reference \`Psi_error\`
did not improve (signed order -0.14); this quantity measures the intended
physical departure from the moving reference and is not a standalone
convergence error.

The result supports proceeding to the outer-domain relaxation discriminator,
but does not establish lapse/profile settling, trumpet formation, convergence,
or long-time stability.  Complete compact histories and resolution tables are
under \`wormhole_short_8792609/\`; endpoint restart files remain only at the
recorded Aurora path.

The AMR physical-profile reducer was checked against synthetic binary64 CBin
snapshots containing exact isotropic-wormhole and exact stationary-trumpet
fields.  It reproduced lapse, determinant conformal factor, and radial shift
to the recorded binary64 tolerances while excluding the conservative
three-cell puncture-stencil cube.  This qualifies the reduction tool, not an
evolution; compact oracle evidence is under
\`wormhole_profile_analyzer_oracle/\`.

The first outer-24M N96 run, Aurora job \`8792643\`, is a genuine negative
result.  It used the initial \`shrinking_width\`, tau-4 reference path and
failed closed at RK stage time \`1.64846M\` when relative conditioning became
invalid.  The last finite history row at \`1.60318M\` already had GH,
reduction, and curl RMS \`2.8409e-2\`, \`1.3891e-3\`, and \`1.2395e-1\`;
the GH log-growth fit had slope \`6.2603/M\` and \`R^2=0.9779\`.

The maxima localize around \`r=0.16--0.24M\`, while retained Hhat/theta/Upsilon
RHS families remain zero.  This reproduces the repository's previously
documented shrinking-core reference-path failure window; it does not qualify
the new gauge.  Complete compact failure evidence and remote large-output
locations are under
\`wormhole_outer24_t4_8792643_shrinking_width_failure/\`.

The next bounded discriminator therefore changes only the reference homotopy
to the repository's fixed-core, tau-8 path.  It retains the new
\`relative_damped\` gauge, STANDARD Phi, gamma0=gamma2=1, FD4, RK4, CFL 0.05,
and KO 0.02.  This correction is motivated by the prior independent
reference-path diagnosis; it is not gauge-parameter tuning.

A reduced Kokkos-Serial one-cycle check of that revised input reached
\`t=0.0460447286M\` with finite state and no bad-state flag.  Endpoint GH,
reduction, and curl RMS were \`2.23585e-7\`, \`8.40904e-7\`, and
\`2.46390e-7\`; relative-D and added-source Linf were \`1.95056e-6\` and
\`2.58297e-5\`.  The exact initial relative match had zero D and source.
This only validates the revised input locally.  It does not cross the failed
\`t~1.65M\` interval or support a relaxation claim.

Aurora job \`8792686\` then ran the same fixed-core/tau-8 input on one node,
12 MPI ranks, and 12 distinct PVC tiles.  It crossed the former failure window
and exited normally at \`t=2.2M\`.  At \`t=1.60239M\`, GH RMS was
\`3.45058e-4\`, about 82 times smaller than the last finite shrinking-width
outer result.  At \`t=2.2M\`, GH, reduction, and curl RMS were
\`8.91313e-4\`, \`1.74411e-4\`, and \`2.75011e-3\`; the relative condition
number was \`1.29073\`, and no bad state was reported.

The late-time GH log fit still has positive slope \`1.87885/M\` over
\`1M<=t<=2.2M\`, so the result is not settling or stability evidence.  It is
a passed failure-window discriminator that authorizes the outer-domain
\`t=4M\` test.  Compact evidence is under
\`wormhole_fixed_core_t2_8792686/\`; four large restart files remain at the
recorded Aurora path and are represented by hashes only.

Aurora job \`8792715\` evolved the full outer-24M, 328-block N96 tree on 96
PVC tiles through \`t=4M\`.  It exited normally and stayed finite, but the
scientific discriminator is unfavorable so far.  GH, reduction, and curl RMS
reached \`2.48217e-3\`, \`6.69745e-4\`, and \`7.98434e-3\`; the GH log slope
over \`2M<=t<=4M\` was \`1.51369/M\`.  Relative-D and added-source Linf were
\`0.20909\` and \`2.06831\`, and the relative metric condition number reached
\`3.66074\`.

The largest GH, Pi/Phi, and RHS values localize at
\`r=0.47--0.53M\`, overlapping the gauge window's outer edge \`r1=0.5M\`.
Hhat/theta/Upsilon RHS remain exactly zero.  This points to the algebraic
relative-gauge/window sector, rather than the disabled hyperbolic driver, but
does not yet separate the lapse term, shift term, and window derivative.

The physical profile also has not approached the stationary trumpet at
\`t=4M\`.  The radial shift is nonzero, but the trumpet-over-wormhole error
ratios exceed one for alpha and psi4 in every aggregate region; for \`r<1M\`
they are \`1.11\` and \`4.62\`.  Because the fixed-core reference transition
is only 50% complete at this time, one bounded checkpoint continuation to
\`t=8M\` is required before classifying the transition as failed.  Compact
histories and physical profiles are under
\`wormhole_fixed_core_outer24_t4_8792715/\`; five restart and ten CBin files
remain on Aurora and are represented by hashes.
