# Aurora Phase-6/7 completion, job 8790947

Job `8790947` completed on `x4610c7s7b0n0` with exit status zero and wall time
`00:12:47`.  Twelve MPI ranks mapped to twelve distinct PVC tiles.

Completed evidence:

- fresh Case A (compatible, gamma2=0, gauge off) through `t=5M`;
- fresh Case B (compatible, gamma2=1, gauge off) through `t=5M`;
- cycle-zero stationary gauge-driver residual decompositions at 64^3, 96^3,
  and 128^3 on the same `[-2M,2M]^3` physical domain.

All three sector decompositions reproduced the production RHS below the
unchanged `5e-13` conditioned tolerance and an immediate production rerun was
bitwise identical.  The global residual maxima move inward at fixed `r/h` and
grow with the singular trumpet coefficients; they are not a fixed-coordinate
convergence sequence.  The companion analysis preserves the measured scaling.

No large CBIN or restart payload was produced by this job.  The complete
remote directory is:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_ordering_gauge_discriminator_phase23_20260829_2c848055/runs/ordering_gauge_phase67_8790947.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`

The in-job `compact_sha256.txt` has the expected self-referential entry created
while that file was open. `verified_compact_sha256.txt` excludes both manifests
and verifies the transferred compact evidence plus this README and final qstat
record.
