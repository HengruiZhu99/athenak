# Diagnostic-only reference Ricci gate

This directory preserves compact Aurora PVC evidence for commit
`338b18f0e075b1ddf842acf67aca8f12762958b4`.  Job `8773745` built distinct
checked and bounds-off SYCL/PVC executables, passed the stationary full-output
cycle and time-dependent stage-time tests, and measured `3.550192e5` active
zone-cycles/s.  Its numerical work completed, but its final postprocessor
returned status 1 because it incorrectly required zero Ricci kernels even
though the stationary problem generator requests an initial diagnostic pass.

Job `8774035` reused those exact binaries and profiled a validation-off,
time-dependent evolved cycle.  The production provider rebuilt five times,
while reference Ricci ran only twice for initial and terminal history.  The
checked oracle case intentionally ran both five times.  This proves that RK
production cache updates no longer construct diagnostic-only reference Ricci.
The evolved field error remained `9.992007e-16` and no PVC fault, NaN, Inf, or
bad state occurred.  Job `8774035` likewise returned status 1 only because its
first offline assertion expected one rather than AthenaK's two history calls;
`diagnostic_split.tsv` is the corrected offline evaluation of the completed
profile.

`SHA256SUMS` covers every compact file in this directory except itself.  Full
outputs remain on Aurora at:

- `/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_pvc_performance_convergence_20260821/runs/diagnostic_split_8773745.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`
- `/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_pvc_performance_convergence_20260821/runs/diagnostic_split_followup_8774035.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`

The throughput increase over the preceding `3.033790e5` measurement is
`1.1702x`, but this change removes setup/diagnostic work rather than the timed
scalar-source hotspot.  Treat that difference as cross-job variability, not a
causal production-stage speedup.
