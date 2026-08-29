# Unmodified single-PVC profile

## Authority and tools

- Aurora job: `8790568`
- state/restart/hierarchy authority: identical to `SINGLE_RANK_BASELINE.md`
- Kokkos Tools simple-kernel-timer commit:
  `833ad8a58862f64822fa97ef1a66444a90b39622`
- Level-Zero profiler: PTI GPU `unitrace` `0.16.0-rc1`
- bounded interval: retained N512 restart to `t=9.65 M` (21 RK cycles)

The profiler changes timing and is not itself a performance baseline.

## Primary disposition

The unmodified one-tile path is launch/synchronization/host-overhead bound:

| quantity | result |
|---|---:|
| total profiled wall time | 21.35330 s |
| time in Kokkos kernels | 5.50499 s (25.78%) |
| time outside Kokkos kernels | 15.84830 s (74.22%) |
| Kokkos kernel calls | 37,177 |
| Level-Zero device time | 3.981615 s |
| Level-Zero API time | 5.581110 s |
| `zeEventHostSynchronize` | 51,453 calls, 5.132570 s |
| memory-copy submissions | 24,003 |
| kernel-launch submissions | 14,858 |

The very short profile includes the mandatory final output, which contributes
4.178752 s and exactly 10,176 small `d_out_var` initialization/copy pairs.
Consequently, steady-state launch counts are lower than the raw total, but
remain far too high.

## Named device-kernel budget

The largest measured kernels were:

| kernel | calls | device time (s) | profiled wall (%) |
|---|---:|---:|---:|
| P6 VC coarse-fine prolongation | 106 | 0.888392 | 4.160 |
| main Z4c RHS | 84 | 0.857450 | 4.016 |
| outer-radial Z4c boundary fill | 256 | 0.800375 | 3.748 |
| same-level VC pack/unpack, fine and coarse | 424 | 0.954253 | 4.470 |
| state admissibility scan | 656 | 0.125336 | 0.587 |
| Hamiltonian plus momentum constraints | 44 | 0.160812 | 0.753 |
| prescribed-zero-shift diagnostic | 84 | 0.059902 | 0.281 |
| timestep source-rate scan | 22 | 0.031382 | 0.147 |
| timestep spatial plus speed scans | 44 | 0.011621 | 0.054 |

The device time alone understates the cost of the diagnostic/synchronization
paths. For example, native-VC shared-node reconciliation executed 212 times,
and each call performs device packing, a roughly 9.68 MB device-to-host copy,
CPU canonical averaging, host-to-device replacement, device application, and
a second device/host postcondition check. PTI recorded 212 such 9.68 MB
device-to-host copies. The admissibility scans similarly force a fence,
device-to-host scalar copy, and MPI reduction at every checkpoint.

## Optimization order supported by measurement

1. Keep the exact shared-node averaging rule but move the one-rank canonical
   reduction and replacement entirely onto the PVC using persistent metadata
   and buffers.
2. Add a lean fail-closed mode that checks states before consumption and at the
   final accepted-state boundary, while retaining the exhaustive writer-level
   provenance mode.
3. Separate vertex-axis enforcement from its expensive host audit; the lean
   path must execute the identical projection and tolerance gate on device.
4. Stop native replay-shadow criterion work after the final authority event.
5. Skip the source-rate full-grid scan when the configured source rate is
   structurally zero; disable the explicitly optional zero-shift diagnostic in
   the production benchmark.
6. Re-profile before changing constraint cadence, P6 transfer, boundary
   mathematics, or the RHS.

Raw Kokkos and PTI reports are under `evidence/profile_one_tile/`.
