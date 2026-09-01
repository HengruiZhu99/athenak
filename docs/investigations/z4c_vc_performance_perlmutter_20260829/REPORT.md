# Latest-source Perlmutter A100 performance check

## Disposition

`RUNTIME_QUALIFIED_SINGLE_A100_TIMING; SCIENCE_REACHED_TLIM; POST_CHILD_SRUN_CLIENT_INTERRUPTED`

The optimized native-VC Cartoon N512 interval reached `t=9.85 M` on one
Perlmutter A100 and delivered `1.186673e7 zone-cycles/s`.  This is `1.992x`
the matched optimized Aurora one-PVC-tile end-to-end throughput and `2.126x`
faster on the non-output portion.  The numerical histories agree to
backend-roundoff scale; this is a performance check, not a new science or
convergence result.

## Provenance

- Perlmutter allocation: Slurm job `57701761`, node `nid001021`
- source branch: `codex/z4c-vc-performance-aurora-20260829`
- source HEAD: `02704c79ffe95312cfeba9acde3d38f8b9677dec`
- optimized source commit: `62993e7bac8fbaed13f592834282ca09142a5c2d`
- source tree: `9605962eed2b74fba6f1d70e6fc526a8f5f56bba`
- Kokkos: `6739bc623081648af9e752b616d9671527922cbf` (`4.7.02`)
- executable SHA-256: `a13a58be5e45b37c4c29f65eab2f6fa0d43f8d235d0c51e7ed3459b18e9e247e`
- CMake cache SHA-256: `079d2d8b3bb18ac716293b8c7498f6564ac257b8f8099b6778a74adf01076721`
- restart SHA-256: `44b8e55957d3b455adf24862d36946e08fc10465df7a30cc5f247ac0e19fa997`
- device: one `NVIDIA A100-SXM4-40GB`, rank binding verified to visible device `0`
- build: Release, MPI, Kokkos CUDA+Serial, `Kokkos_ARCH_AMPERE80=ON`, OpenMP off

The exact cycle-3994 restart and authority used by the optimized Aurora
benchmark were transferred and hash-verified.  The interval retained the same
212 MeshBlocks of `64 x 64` cells, O4/RK4/CFL 0.15, KO 0.50, P6 transfer,
shock-avoiding lapse, prescribed-zero shift, zero constraint damping, and
output cadence.

## Timing

| quantity | Perlmutter A100 | Aurora PVC tile | A100/PVC speedup |
|---|---:|---:|---:|
| reported total time | 6.878481 s | 13.69965 s | 1.992x |
| output time | 1.357807 s | 1.962092 s | -- |
| non-output time | 5.520674 s | 11.737558 s | 2.126x |
| zone-cycles/s | 11,866,730 | 5,958,188 | 1.992x |

The A100 reached 100% sampled utilization and used at most 20,313 MiB during
the sampled N512 execution.

## Cross-backend numerical check

Both histories have 25 rows and 69 columns with identical coordinate times.
Maximum differences over the interval include:

| quantity | maximum absolute difference | maximum relative difference |
|---|---:|---:|
| dt | `3.04e-18` | -- |
| C | `7.23e-15` | `2.55e-11` |
| H | `7.00e-15` | `9.72e-10` |
| M | `4.36e-17` | `1.87e-11` |
| Z | `1.12e-16` | `1.65e-12` |
| max abs K | `8.60e-14` | `7.42e-13` |
| max Kretschmann | `7.06e-12` | `2.34e-11` |

## Completed N1024 four-A100 run

Perlmutter job `57702588` completed the from-scratch N1024 exact replay on one
four-A100 node in `02:37:46`.  Athena reached the matched endpoint
`t=38.652331986867424 M` at cycle `30413`; the scheduler and launcher both
exited zero.  The reported aggregate rate was `1.119726e7 zone-cycles/s`,
essentially the same as the one-A100 N512 aggregate rate.  Thus this small 2D
four-rank workload delivered little aggregate strong-scaling gain; the four
GPUs were needed principally to fit the N1024 memory footprint.

The successful run used `max_nmb_per_rank=128`.  Its final load balance was 53
MeshBlocks per rank.  An earlier zero-science startup attempt with capacity
2048 failed during allocation because large Z4c arrays are sized to that
ceiling.  Reducing the capacity ceiling changed neither the fixed hierarchy nor
the numerical method.

## Limitation

For the short single-A100 N512 benchmark, Athena and its CUDA rank wrapper
completed and wrote the endpoint, but the Slurm `srun` client remained stuck
after child exit and was interrupted. Its performance timing and endpoint are
retained without claiming a clean launcher exit. The production N1024 run did
not reproduce that behavior: its launcher and scheduler exits were both clean.

## N1024 CFL 0.40 / KO 0.10 ablation

A subsequent exact-tree N1024 run changed both CFL (`0.15 -> 0.40`) and KO
dissipation (`0.50 -> 0.10`). Perlmutter job `57706639` accepted the same two
authority events and reached coordinate time `16.72465031794259 M` before the
strict post-RK state check stopped it. The conformal metric was not positive
definite at `(rho,z)=(0.0625,19.96875)`; chi remained positive. Its last
history row is at proper time `8.687343281197583`, with C squared integral
`4.9679e4`. This is a failed partial science trajectory, not a completed
Figure-3 run, and the simultaneous parameter changes prevent separate CFL and
KO attribution.

Job `57706621` was an earlier startup-only failure due to the unrecognized
parameter spelling `time/cfl`; it produced no science data. The corrected
runner uses `time/cfl_number`.
