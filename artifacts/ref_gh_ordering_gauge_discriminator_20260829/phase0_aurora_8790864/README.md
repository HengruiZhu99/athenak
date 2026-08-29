# Phase 0 bounded Aurora PVC gate

Aurora debug job `8790864` ran on node `x4216c4s4b0n0` from exact source
commit `39069a16d2d36e1bf5d124f7f274382eca4cd441`.  PBS reported exit status
zero after 11 minutes 25 seconds.

Eight ranks mapped to eight distinct Intel PVC tiles, `0.0` through `3.1`,
using Level Zero and GPU-aware MPI.  The one-rank and eight-rank evolved
dynamic-q cycles passed with conditioned history Linf difference
`3.88980825583101983e-14`, below the unchanged `5e-12` tolerance.  The
production allocation reported zero generic reference-cache bytes and the
expected 12-Real static / 8-Real stage analytic views.

The job intentionally did not run a benchmark or a long evolution.  This is
the frozen Phase 0 PVC gate only.

Full scratch location:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_ordering_gauge_discriminator_20260829_39069a16/runs/analytic_q_pvc_8790864.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`
