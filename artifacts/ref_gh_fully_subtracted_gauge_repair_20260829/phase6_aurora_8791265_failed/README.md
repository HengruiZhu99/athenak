# Aurora Phase-6 staged-oracle gate: preserved build failure

Aurora debug job `8791265` used source commit
`7838324471d0ecb2fe7592bd497230fc5a7d4c40`, project
`CompactBinaryMerger`, and one node.  PBS ran it on `x4516c7s2b0n0` and
reported exit status 2 after 00:10:01.

This source separated the legacy-generic, fully-subtracted compact, and
conditioned comparison portions of the all-61 oracle into three kernels.  The
local Serial regression retained the exact prior result.  Aurora configuration
again passed with IntelLLVM 2025.3.2, Kokkos 4.7.2 `SERIAL;SYCL`, and PVC
architecture enabled, but IGC still failed while lowering the complete
`source_unit.cpp` device image:

```text
IGC: Internal Compiler Error: Segmentation violation
Build failed with error code: -11
icpx: error: gen compiler command failed with exit code 245
```

The staged arithmetic was insufficient because Kokkos 4.7.2 supplies
`-fsycl-device-code-split=off`, leaving every independent kernel in the large
validation translation unit inside one SPIR-V module.  The job stopped before
rank-to-tile mapping or device execution.  No numerical or evolution claim is
supported.

Complete Aurora location:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_fully_subtracted_gauge_repair_20260829_78383244_phase6_v2/runs/phase6_device_8791265.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`
