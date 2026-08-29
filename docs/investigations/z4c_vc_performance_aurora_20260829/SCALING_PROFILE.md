# Optimized N512 strong-scaling profile

## Configuration

Aurora job `8790675` ran the frozen 212-MeshBlock N512 replay sequentially at
1, 2, 4, 6, 12, and 24 ranks, one rank per PVC tile.  Every case used source
`02a9b465`, the same restart and AMR-history authority, O4/RK4/CFL 0.15,
KO 0.50, P6 transfer, shock-avoiding lapse, prescribed-zero shift, zero
constraint damping, and unchanged physical boundaries.

## Measured curve

| tiles | blocks/rank | execution (s) | output (s) | non-output (s) | zone-cycles/s | speedup | efficiency |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 212.00 | 16.480 | 4.720 | 11.760 | 4,952,884 | 1.000 | 100.0% |
| 2 | 106.00 | 44.081 | 24.543 | 19.538 | 1,851,726 | 0.374 | 18.7% |
| 4 | 53.00 | 24.059 | 10.157 | 13.902 | 3,392,717 | 0.685 | 17.1% |
| 6 | 35.33 | 35.360 | 20.842 | 14.517 | 2,308,427 | 0.466 | 7.8% |
| 12 | 17.67 | 19.518 | 4.968 | 14.550 | 4,182,024 | 0.844 | 7.0% |
| 24 | 8.83 | 19.065 | 5.097 | 13.968 | 4,281,501 | 0.864 | 3.6% |

The 2- and 6-tile output times are filesystem outliers, so end-to-end rows are
not monotone.  The non-output column is much tighter and still shows no strong
scaling: every multi-tile case is slower than one tile.  The limited workload
(only 212 blocks) does reduce available work per rank, but the source-level
global host-staged shared-vertex reconciliation supplies a concrete repeated
serialization mechanism consistent with the curve.

Machine-readable values are in `scaling.csv` and `scaling.json`; per-run logs
and provenance are under `evidence/scaling_*_tiles/`.

## Lean-postcondition and sparse-exchange ablations

The initial curve still ran the exact synchronization postcondition on every
MPI call. Source `4260e5ba` made lean MPI behavior match the lean one-rank
contract. It reduced non-output time by 4.6% at 2 tiles and 10.1% at 24 tiles,
but did not restore scaling:

| tiles | execution (s) | output (s) | non-output (s) | zone-cycles/s |
|---:|---:|---:|---:|---:|
| 2 | 47.954 | 29.323 | 18.631 | 1,702,151 |
| 24 | 18.587 | 6.038 | 12.549 | 4,391,482 |

Source `62993e7b` then replaced the global host-staged value exchange with the
sparse GPU-aware canonical exchange described in `SHARED_VERTEX_SYNC_AUDIT.md`.
Aurora jobs `8790725`, `8790731`, and `8790735` produced the final-source
single-PVC, one-node, and two-node endpoints:

| tiles | blocks/rank | execution (s) | output (s) | non-output (s) | zone-cycles/s | speedup | efficiency |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 212.00 | 13.700 | 1.962 | 11.738 | 5,958,188 | 1.000 | 100.0% |
| 2 | 106.00 | 31.904 | 15.066 | 16.838 | 2,558,434 | 0.429 | 21.5% |
| 12 | 17.67 | 16.952 | 10.187 | 6.765 | 4,815,004 | 0.808 | 6.73% |
| 24 | 8.83 | 11.703 | 5.060 | 6.644 | 6,974,422 | 1.171 | 4.88% |

Relative to the matched postcondition-only cases, sparse exchange improves
throughput by `1.503x` at 2 tiles and `1.588x` at 24 tiles. The 24-tile
non-output portion improves by `1.889x`. The histories are byte-identical and
the final restart numerical payloads are exact, so this is measured removal of
implementation overhead rather than changed evolution arithmetic.

The final sparse path was not rerun at 4 or 6 tiles; those rows above remain
measurements of the earlier global-exchange path and must not be presented as
part of the final-source curve. The final 12-tile output time is a filesystem
outlier. On the less I/O-sensitive non-output measure, 12 and 24 tiles obtain
`1.735x` and `1.767x` speedup over the final one-tile run, respectively.

## Interpretation boundary

This curve establishes poor strong scaling for this bounded hierarchy.  It
does not by itself assign all lost time to one routine, nor does it qualify
larger science runs. The sparse ablation confirms that shared-vertex staging
was a major cost, but the final 4.88% end-to-end efficiency at 24 tiles also establishes
substantial residual fixed/communication and low-occupancy overhead for this
212-block workload. It does not quantify those residual components separately.
