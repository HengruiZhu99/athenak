# Aurora Phase-6 device gate: preserved build failure

Aurora debug job `8791211` used source commit
`ef130480bfdbecc5b7d8f21169085be5da6c8cc4`, project
`CompactBinaryMerger`, and one node.  PBS ran it on `x4613c0s1b0n0` and
reported exit status 2 after 00:10:41.

Configuration passed with IntelLLVM 2025.3.2, Kokkos 4.7.2,
`SERIAL;SYCL`, and `Kokkos_ARCH_INTEL_PVC=ON`.  Compilation then failed in
`src/pgen/ref_gh/source_unit.cpp.o` while lowering the monolithic all-61
device oracle:

```text
IGC: Internal Compiler Error: Segmentation violation
Build failed with error code: -11
icpx: error: gen compiler command failed with exit code 245
```

The job stopped before rank-to-tile mapping or any device kernel executed.
It is therefore a failed compile gate, not PVC runtime or numerical evidence.
No evolution was attempted.

The equation-preserving follow-up stages the independent generic and compact
61-component evaluations in separate device kernels and performs the same
conditioned comparison in a third reduction.  It retains all 4320 samples,
both Phi orderings, the generic oracle, and the unchanged `256*epsilon`
tolerance.

Complete Aurora location:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_fully_subtracted_gauge_repair_20260829_ef130480_phase6_v1/runs/phase6_device_8791211.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`
