# Native-VC Cartoon Z4c Aurora performance report

## Verdict

The final source passes the required single-PVC performance gate while
preserving the bounded N512 trajectory bitwise.

| comparison | original | final | result |
|---|---:|---:|---:|
| zone-cycles/s | 1,763,040 | 5,958,188 | **3.3795x** |
| reported execution time | 46.29792 s | 13.69965 s | 0.2959x |
| output wall time | 2.623702 s | 1.962092 s | 0.7478x |
| non-output time | 43.67422 s | 11.73756 s | **3.7208x** |

The hard `>=3x` target passes. The `>=5x` stretch target does not. The final
profile shows that required RHS and P6 transfer arithmetic dominate the
remaining device time, so a further low-risk 1.48x improvement was not
credible within this campaign.

## Fixed numerical workload

All authoritative comparisons evolve the same retained 212-MeshBlock frozen
N512 hierarchy from cycle 3994 near `t=9.59463 M` to `t=9.85 M`. They retain:

- vertex-centered Cartoon SO(2), O4 finite differences, P6 AMR transfer;
- RK4, CFL `0.15`, KO `0.50`;
- shock-avoiding lapse with `kappa=1`, prescribed-zero shift;
- `kappa1=kappa2=0` Z4c constraint damping;
- the same recorded AMR authority and no topology changes in the benchmark;
- axis at `rho=0`, Sommerfeld Z4c RHS closure at the outer radial/axial
  boundaries, and periodic collapsed direction;
- physical domain `rho in [0,128]`, `z in [-128,128]`.

No evolution, gauge, damping, dissipation, transfer, AMR, or boundary
mathematics were changed.

## What was optimized

The measured optimization sequence was:

1. Add a default-off lean runtime that keeps fail-closed checks at consumption
   and accepted-state boundaries while avoiding repeated observational scans.
2. Move one-rank shared-vertex canonical averaging entirely to persistent
   device metadata/buffers.
3. Remove exhausted replay-shadow work and structurally zero gauge/source-rate
   scans.
4. Compact P6 and physical-boundary launch domains.
5. Fold all 25 variables into each same-level VC neighbor team.
6. Keep the lean P6 positivity gate on device.
7. Mirror each homogeneous output source once rather than once per
   variable/MeshBlock.
8. Honor the lean postcondition setting on MPI runs.
9. Replace the lean hot-path global host-staged `MPI_Allgatherv` with a
   deterministic sparse owner/participant GPU-aware exchange.

The exhaustive/default diagnostic path remains available. The rejected
aggregate-output experiment was reverted and is retained only as measured
negative evidence in `OPTIMIZATION_LOG.md`.

## Profile result

Over the matched 21-cycle profile window, wall time fell from `21.35330 s` to
`5.78705 s`; outside-Kokkos time fell from `15.84830 s` to `2.38654 s`; and
Kokkos calls fell from 37,177 to 3,635. The largest final kernels are:

| kernel | calls | device time |
|---|---:|---:|
| main Z4c RHS | 84 | 0.857972 s |
| P6 coarse-fine prolongation | 106 | 0.699207 s |
| vertex-axis regularity | 294 | 0.150255 s |
| one-rank canonical VC average | 212 | 0.121462 s |

The four same-level VC pack/unpack families total `0.096686 s`, about one
tenth of the original profile. See `OPTIMIZED_PROFILE.md` for the full budget.

## Final-source scaling

The final sparse path was measured at the required single-PVC, one-node, and
two-node endpoints, one rank per PVC tile:

| tiles | blocks/rank | execution | output | non-output | zone-cycles/s | end-to-end speedup | non-output speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 212.00 | 13.700 s | 1.962 s | 11.738 s | 5,958,188 | 1.000x | 1.000x |
| 2 | 106.00 | 31.904 s | 15.066 s | 16.838 s | 2,558,434 | 0.429x | 0.697x |
| 12 | 17.67 | 16.952 s | 10.187 s | 6.765 s | 4,815,004 | 0.808x | 1.735x |
| 24 | 8.83 | 11.703 s | 5.060 s | 6.644 s | 6,974,422 | 1.171x | 1.767x |

The 2- and 12-tile output times are filesystem outliers. Even on the
less-I/O-sensitive non-output measure, 24 tiles achieve only 7.36% efficiency.
The original slowdown therefore had two distinct parts:

- dominant single-rank launch, scan, staging, and output overhead, reduced by
  `3.72x` outside output;
- a real global host-staged MPI/VC serialization cost, whose sparse repair
  improves matched throughput by `1.50x` at 2 tiles and `1.59x` at 24 tiles.

Residual poor scaling is consistent with only 8.83 blocks per rank at 24
tiles plus remaining task-launch, boundary-exchange, reduction, and occupancy
costs. This campaign does not apportion those residual costs individually.
The final source was not rerun at 4 or 6 tiles; the earlier complete curve is
kept separately in `SCALING_PROFILE.md`.

## Numerical equivalence

The final one-PVC N512 run (Aurora job `8790731`) is bitwise identical to the
unmodified baseline:

- complete history SHA-256:
  `de03537e989bfdba1425af13efde52dd060cb5a52fe5d470354df81f962fb0fd`;
- final restart numerical-payload SHA-256:
  `e498614cad5e50677a1698bc20680e5b34131e0a098c3421e1d64564702c6ab6`;
- identical final time, cycle, MeshBlock count, and maximum level.

Matched old/new 2-, 12-, and 24-rank histories are also byte-identical, and
their final restart numerical payloads have the same hash above. This checks
the changed MPI path on the production trajectory rather than only one rank.

## Provenance

- optimized source: `62993e7bac8fbaed13f592834282ca09142a5c2d`;
- optimized source tree: `339b8f6a134a50fe7916013fd96f5cf93ea3a58d`;
- optimized Aurora executable SHA-256:
  `b070bf3b856be712134b0e38028304bbb2fde506aa271350f98b3d8ee243c1e2`;
- baseline source: `f8303c6be7eb214fa1e91b646123ee0d434b3698`;
- baseline executable SHA-256:
  `aae7ccb8739fb4951221ad7be69ea0e220548b52d402086f57d7857fa2c97a13`;
- retained restart SHA-256:
  `44b8e55957d3b455adf24862d36946e08fc10465df7a30cc5f247ac0e19fa997`;
- AMR authority SHA-256:
  `7055de601e6181e5ad7e1432b5c20a111b0ba67e0e8d5377c170ea80e7bedcde`.

The executable's embedded `athena-0.1-git-*` string remains the older
configure-time identifier `9c98bd14`; the runner independently fail-closed on
the clean checked-out source commit/tree and executable hash above. This is a
build-metadata limitation, not an ambiguity in which source or executable ran.

## Validation and limitations

- Fresh host and MPI-enabled host builds pass.
- Focused topology and policy tests pass.
- The broad host suite has three reproducible pre-existing stale whole-file
  metadata golden failures; numerical payload/history checks remain exact.
- A local two-rank runtime launch could not start because the workstation
  `mpirun` process entered uninterruptible I/O. Aurora jobs exercised the
  actual multi-rank GPU path successfully at 2, 12, and 24 ranks.
- GPU utilization and MPI-time decomposition were not available as reliable
  final-source counters; conclusions use wall budgets, Kokkos/PTI profiles,
  source audit, and matched ablations.
- No N1024 production run was performed.
- The bounded frozen-hierarchy checks do not establish convergence, long-time
  stability, Figure 3 reproduction, boundary independence, or physical
  critical behavior.

Machine-readable evidence and hashes are indexed in `EVIDENCE_MANIFEST.json`.
