# N1024 exact-replay result

## Disposition

`REACHED_TLIM; FINE_TRIPLE_O4_COMPATIBLE_THROUGH_FIRST_PEAK; PEAK_STILL_FAILS_STRICT_C_GATE`

The N1024 run reached the matched endpoint
`t=38.652331986867424 M`, cycle `30413`, and central proper time
`14.98066136 M`. It replayed the accepted N256 hierarchy exactly and retained
212 leaf MeshBlocks. No curve transform was used in the Figure-3 comparison.

## Frozen setup and execution

- native vertex-centered Cartoon SO(2), O4, P6, RK4, CFL `0.15`;
- KO dissipation `0.50`;
- shock-avoiding lapse with `kappa=1`, prescribed-zero shift;
- Z4c `kappa1=kappa2=0`;
- root grid `512 x 1024`, `128 x 128` cells per physical MeshBlock;
- same root MeshBlock layout, physical bounds, LogicalLocation tree, and replay
  event times as N128/N256/N512;
- Perlmutter interactive job `57702588`, one node, four MPI ranks, and four
  distinct A100-SXM4-40GB UUIDs;
- clean scheduler and launcher exit `0:0`, elapsed time `02:37:46`.

The two replay events matched at zero ULP time difference, with 200 and 212
leaves and checksums `24316947a3a67cd8` and `cf0d2384b11c1d42`.

The run used source `02704c79ffe95312cfeba9acde3d38f8b9677dec`, tree
`9605962eed2b74fba6f1d70e6fc526a8f5f56bba`, Kokkos
`6739bc623081648af9e752b616d9671527922cbf`, and executable SHA-256
`a13a58be5e45b37c4c29f65eab2f6fa0d43f8d235d0c51e7ed3459b18e9e247e`.
The replay authority SHA-256 is
`7055de601e6181e5ad7e1432b5c20a111b0ba67e0e8d5377c170ea80e7bedcde`.

Two startup attempts produced no science data and are excluded from analysis:
job `57702533` failed the explicit lean-runtime consistency guard, and job
`57702560` exhausted device memory because `max_nmb_per_rank=2048` reserved
far more capacity than the 53 blocks assigned to each rank. The successful run
used a capacity ceiling of 128, leaving more than a factor-two margin without
changing the accepted hierarchy or numerical method.

## Figure-3 and constraint observations

| feature | N512 | N1024 | published range |
|---|---:|---:|---:|
| first-peak proper time | 10.30811 | 10.30964 | 10.30683--10.31384 |
| first-peak `log10(abs(Kretschmann))` | 5.38112 | 5.47867 | 5.47778--5.48688 |
| deep-minimum proper time | 12.62280 | 12.61950 | 12.61674--12.73112 |
| deep-minimum `log10(abs(Kretschmann))` | -6.07875 | -7.88841 | -6.54553---5.20673 |
| rebound proper time | 13.21629 | 13.21271 | 13.18978--13.21977 |
| rebound `log10(abs(Kretschmann))` | -2.81849 | -2.81258 | -2.95731---2.81225 |

The unshifted N1024 RMSE is `0.01754` against the published bamps curve and
`0.02367` against Prague over their common sampled interval.

Maximum squared constraint integrals are:

| family | N512 | N1024 | N1024/N512 |
|---|---:|---:|---:|
| C | 4.09930 | 0.0388727 | 0.00948 |
| H | 3.63079 | 0.0315101 | 0.00868 |
| M | 0.682824 | 0.00453466 | 0.00664 |
| Z | 0.00462028 | 0.00400120 | 0.866 |

The N1024 C integral first exceeds `0.01` at proper time `10.17835` and never
exceeds `0.1`. Thus the peak is dramatically cleaner but still fails the
campaign's pre-existing strict C validity gate.

The N256/N512/N1024 median Richardson orders for central Kretschmann are
`4.10`, `3.89`, and `5.91` in the `0--8`, `8--10`, and `10--11.286` windows;
for central lapse they are `3.99`, `3.79`, and `3.34`. The value above four is
reported descriptively, not as an accuracy-order claim beyond O4.

## Artifacts and claim boundary

Compact run evidence is under
`../z4c_vc_performance_perlmutter_20260829/evidence/n1024_four_a100_full/`.
Updated plots and the four-resolution table are under
`analysis/aurora_n128_n256_n512_perlmutter_n1024/final/`.

Supported: direct first-peak agreement, strong same-tree resolution
improvement, and O4-compatible fine-triple central fields through the peak.

Not supported: a constraint-qualified Figure-3 reproduction, three-level
late-minimum/rebound convergence, or attribution to a unique source-level bug.
