# Optimized single-PVC profiles

## Scope

This profile measures the first lean-runtime candidate at source commit
`98bb0ac3a0037c3bfc7ee0831b4cc22f7ed1b6c2`, before aggregate output staging
was added.  Aurora job `8790602` evolved the retained N512 hierarchy for 21
RK4 cycles, from the authoritative restart to `t=9.65 M`, on one PVC tile.
The Kokkos and PTI instrumentation are the same as in
`SINGLE_RANK_PROFILE.md`; profiler timing is not a throughput baseline.

## Measured change

| quantity | unmodified | first candidate | change |
|---|---:|---:|---:|
| profiled wall time | 21.35330 s | 8.88470 s | 2.403x faster |
| Kokkos kernel time | 5.50499 s | 4.82201 s | 12.4% lower |
| outside-Kokkos time | 15.84830 s | 4.06269 s | 74.4% lower |
| outside-Kokkos fraction | 74.22% | 45.73% | -28.49 points |
| Kokkos calls | 37,177 | 34,256 | 7.9% lower |
| Level-Zero device time | 3.981615 s | 3.724168 s | 6.5% lower |
| Level-Zero API time | 5.581110 s | 4.839667 s | 13.3% lower |
| `zeEventHostSynchronize` calls | 51,453 | 45,307 | 11.9% lower |
| `zeEventHostSynchronize` time | 5.132570 s | 4.463974 s | 13.0% lower |

The first candidate therefore removed the dominant per-cycle host-staged
shared-node reconciliation and audit overhead.  Its complete unprofiled N512
benchmark reached `4.123011e6` zone-cycles/s, a `2.33858x` speedup over the
unmodified one-PVC baseline, with exact history and final-restart numerical
payload equivalence.  It did not meet the required `3x` gate.

## Remaining measured output bottleneck

The 21-cycle profile still performed the final state, constraint, and
curvature binary outputs through the historical per-variable/per-MeshBlock
path.  That path submits one allocation/copy sequence for each of 48 variables
on each of 212 MeshBlocks: exactly 10,176 small staging instances.  The run
reported 3.105902 s of output wall time and retained 21,124 Level-Zero copy
submissions plus 45,307 host synchronizations.

This measurement motivated the second, default-off change at `5c346599`: in
lean mode, each output variable is gathered over all selected MeshBlocks on
the device and all output data are transferred to the host in one aggregate
copy.  The local native-VC and cell-centred output fixtures are byte-identical
between the historical and aggregate paths.  Aurora throughput and N512
equivalence remain qualification requirements; this document does not infer
their outcome.

Raw evidence is under `evidence/optimized_profile_one_tile/`.

## Frozen hard-pass profile

After the compact P6/boundary launches, variable-folded VC communication,
device-side P6 gate, and homogeneous-source output mirror were combined,
Aurora job `8790685` profiled source `02a9b465` over the same 21-cycle window.

| quantity | unmodified | frozen hard-pass | change |
|---|---:|---:|---:|
| profiled wall time | 21.35330 s | 5.78705 s | 3.69x faster |
| Kokkos kernel time | 5.50499 s | 3.40051 s | 38.2% lower |
| outside-Kokkos time | 15.84830 s | 2.38654 s | 84.9% lower |
| outside-Kokkos fraction | 74.22% | 41.24% | -32.98 points |
| Kokkos calls | 37,177 | 3,635 | 90.2% lower |

The dominant remaining kernels were the main Z4c RHS (`0.857972 s`, 84 calls)
and P6 coarse-fine prolongation (`0.699207 s`, 106 calls). The four same-level
VC pack/unpack families totaled `0.096686 s`, down from approximately
`0.958 s` in the original profile. Shared-node averaging plus application
cost `0.135621 s` on one rank.

This profile shows why the campaign stopped pursuing the `5x` stretch target:
the remaining device budget is dominated by required RHS and transfer
arithmetic, while the largest removable launch/staging path had already been
reduced by roughly an order of magnitude. More aggressive arithmetic fusion
would have carried substantially greater numerical-risk for limited measured
headroom. Raw evidence is under `evidence/optimized_profile_one_tile_v8/`.
