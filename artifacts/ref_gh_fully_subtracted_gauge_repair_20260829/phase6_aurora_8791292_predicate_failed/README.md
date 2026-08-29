# Aurora Phase-6 per-kernel image: preserved predicate-test failure

Aurora debug job `8791292` used source commit
`28eee9d8efaab0d567fcbeda5ae549210c87d9f8`, project
`CompactBinaryMerger`, and one node.  PBS ran it on `x4516c7s4b0n0` and
reported exit status 1 after 00:17:45.

The source-unit-only `-fsycl-device-code-split=per_kernel` override was present
after Kokkos' global `device-code-split=off` flag.  The executable compiled
successfully, and 12 ranks mapped to distinct Level Zero PVC tiles `0.0`
through `5.1`.

The one-rank source-unit process then failed before its first oracle kernel:

```text
### FATAL ERROR: exact matched q=1 predicate admitted a controlled, moving,
or nonidentical reference.
```

The false-case probe used `std::numeric_limits<Real>::denorm_min()` for a
nonzero rate/acceleration.  Aurora's active Intel floating-point mode flushed
that subnormal to zero, so the test accidentally supplied the exact accepted
state.  The follow-up changes only the test probes to the smallest positive
normal value.  The production predicate and formulation are unchanged.

This job establishes successful PVC compilation and distinct tile mapping for
the per-kernel image, but not execution of the all-61 oracle or the production
fixed point.  No evolution was attempted.

Complete Aurora location:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_fully_subtracted_gauge_repair_20260829_28eee9d8_phase6_v3/runs/phase6_device_8791292.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`
