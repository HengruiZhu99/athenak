# Aurora Phase-7 fully subtracted fixed-point ladder

Aurora debug job `8791429` used source commit
`3c9a34c8c3123c2570eb33e8ec77368feb1f1c61`, project
`CompactBinaryMerger`, and one node. PBS ran it on `x4219c2s3b0n0` and
reported exit status 0 after 00:11:58.

The production SYCL image compiled and 12 ranks mapped to distinct Level Zero
PVC tiles `0.0` through `5.1`. The exact matched STANDARD cycle-zero fixed
point passed at 64^3, 96^3, and 128^3 with the full three-cell FD4+KO puncture
stencil excluded. The first included radius moved from `0.2231696384M` to
`0.1115848192M` at fixed `r/h=sqrt(12.75)`.

At every resolution the following are bitwise zero:

- stored delta Hhat and delta theta;
- production `deltaF_A`, delta conformal Gamma, and delta shift;
- actual Hhat, theta, and Upsilon RHS;
- driver Hhat, theta, and Upsilon RHS;
- ordinary-gauge Pi increment;
- KO Hhat, theta, and Upsilon.

The old moving-`r/h` gauge envelope is therefore absent. The remaining total
RHS is entirely the covariant-vacuum Pi source and ranges from `2.07099e-13`
to `3.80172e-13`. The legacy independently reconstructed full-target
diagnostic still grows from `3.03757e-13` to `1.49782e-11`; this is retained as
direct evidence of the cancellation that the production residual avoids.

The raw `compact_sha256.txt` was generated before the final bootstrap line and
included itself, so exactly its bootstrap and self entries do not validate.
All scientific files in that raw manifest validate. The committed
`SHA256SUMS` is generated after retrieval, excludes itself, and validates the
complete preserved bundle. The launcher ordering is corrected for future
runs; no numerical rerun or reinterpretation is involved.

This is cycle-zero algebraic/fixed-point evidence only. It does not establish
positive-time stability, evolved convergence, 20M robustness, or production
readiness.

Complete Aurora location:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_fully_subtracted_gauge_repair_20260829_3c9a34c8_phase7_v1/runs/phase7_ladder_8791429.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`
