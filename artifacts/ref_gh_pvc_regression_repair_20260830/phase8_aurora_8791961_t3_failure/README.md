# Aurora Phase 8 Gate 8A failure

Aurora job `8791961` ran the repaired, fully-subtracted 96^3 STANDARD Ref-GH
case on 36 PVC tiles across three nodes. The target-scoped SYCL per-kernel
build correction remained effective: the logs contain no Level Zero error.

The scientific gate failed before its requested `t=3M` endpoint. History
output reached `t=1.280090959752213M`, after which the run stopped on an
invalid effective timestep. The old fast inner mode recurred with Hhat RHS
e-folding time `0.0382808M` and R-squared `0.998635`. The RHS maxima localized
at `r=0.14878M`. This bundle supports neither stationary-trumpet stability nor
convergence.

Contents include exact provenance and mapping, compact histories, maximum
locations, logs, the analysis JSON/TSV, and checksums. Large restart files are
not committed. They remain on Aurora at:

```text
/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_pvc_regression_20260830_wr_v2/runs/phase8_t3_8791961.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov/run/rst
```

`restart_sha256.txt` records all three restart hashes. Later 5M,
three-resolution, 20M, and 100M gates were not run.
