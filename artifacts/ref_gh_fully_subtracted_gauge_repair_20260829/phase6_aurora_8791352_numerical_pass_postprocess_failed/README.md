# Aurora Phase-6 numerical pass with postprocessor failure

Aurora debug job `8791352` used source commit
`4ec57dd25358ac0d774f4cb5f1c7a89b041dac67`, project
`CompactBinaryMerger`, and one node. PBS ran it on `x4516c7s4b0n0` and
reported exit status 1 after 00:24:37.

The numerical gates completed before that nonzero exit:

- the executable compiled with IntelLLVM 2025.3.2 and Kokkos 4.7.2
  `SERIAL;SYCL` for PVC;
- 12 ranks mapped to distinct Level Zero PVC tiles `0.0` through `5.1`;
- the one-tile source-unit/device suite passed, including the moving-reference
  mixed-jet gate and all 4320 compatible/STANDARD all-61 comparisons;
- the all-61 conditioned maximum was `5.46091e-14` under the unchanged
  `256*epsilon` tolerance;
- the 12-tile 96^3 STANDARD exact matched fixed point completed;
- fixed-point reproduction and production-rerun errors were exactly zero;
- actual Hhat/theta/Upsilon, the ordinary-gauge Pi increment, driver
  Hhat/theta/Upsilon, and KO Hhat/theta/Upsilon were exactly zero.

The launcher failed only after both numerical runs. Aurora's default Python
3.6.15 rejected the diagnostic-only f-string debug syntax `{reproduction=}`
while parsing the already-written fixed-point TSV. The preserved raw
`fixed_point_gate.txt` is therefore empty. The unchanged parsing logic was
rerun independently with Python 3.12.3 and is preserved in
`fixed_point_gate_recovered.txt`; it passes every declared criterion. The
launcher follow-up rewrites only the error message in Python-3.6-compatible
syntax. No formulation, numerical algorithm, tolerance, or runtime task graph
changes.

This evidence runtime-qualifies the named Phase-6 source-unit/all-61 and 96^3
exact fixed-point workloads on PVC. It does not qualify a positive-time
evolution, the 64/96/128 stationary ladder, trumpet stability, convergence, or
production readiness.

Complete Aurora location:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_fully_subtracted_gauge_repair_20260829_4ec57dd2_phase6_v4/runs/phase6_device_8791352.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`
