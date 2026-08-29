# Fresh Aurora phase-1 baseline

This compact bundle preserves the exact-current-build baseline used to decide
whether Ref-GH RHS staging is required.  It contains no field dumps or restart
files.

## PVC evolved gate

- PBS job: 8790322, `debug`, one Aurora node, eight MPI ranks/PVC tiles
- result: `PASS_BOUNDED_EIGHT_TILE_DYNAMIC_Q_CYCLE`
- one/eight conditioned Linf: `3.88980825583101983e-14`
- unchanged tolerance: `5e-12`
- executable SHA-256:
  `5d56f36afa384694fa55704affd08e478c637b9e6425113df79fbc25fd9a166f`
- primary analytic scalar-source/Pi compiler pressure: SIMD16, 256 registers,
  1279 spilled Reals for FDNG 4/3 and 1282 for FDNG 2

See `pvc_8790322/gate_status.txt`, `numerical_comparison.txt`,
`rank_gpu_mapping.txt`, `compiler_pressure.tsv`, and the compact run logs.

## Matched warmed benchmark

- PBS job: 8790348, `debug`, one Aurora node
- matched active grid: 64 cubed
- dynamic/static complete-stage ratio: `1.0069059971`
- Ref-GH/Z4c complete-stage ratio: `9.2448863831`
- Ref-GH/Z4c RHS ratio without dissipation: `10.9028283081`
- Ref-GH/Z4c RHS ratio with dissipation: `10.4228120090`
- q-control fraction: `0.0060727679`
- analytic-reference fraction: `0.0004752617`

The controlling values in `performance_8790348/analysis.json` subtract the
20-cycle warmup from each 100-cycle measured run and aggregate the separately
fenced kernel profile.  The raw `performance_summary.tsv` is retained but is
not the controlling warmed aggregate.

Full outputs remain at:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_staged_rhs_20260828_baseline_a09caf_phase1`
