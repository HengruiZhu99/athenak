# Aurora Phase-6 discriminator, job 8790932

PBS job `8790932` completed on `x4300c2s0b0n0` in `00:06:37` with exit
status zero.  Twelve ranks mapped to twelve distinct PVC tiles.

Case E (standard Phi ordering, gamma2=1, gauge driver and reference gauge
subtraction disabled) completed a fresh evolution through `t=3M`.  This is the
prescribed discriminator: together with the matched Case-D failure it shows
that the standard Einstein/reference system is healthy and that the rapid mode
requires the evolved gauge sector or its coupling.

The attempted Case-A and Case-B checkpoint continuations failed immediately
with an invalid effective timestep.  They produced no accepted continuation
history and are restart-launch failures, not evidence that the fresh A/B
evolutions became unstable.  Fresh A/B `t=5M` runs are required.

Large CBIN snapshots and restart files are intentionally excluded.  They
remain at:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_ordering_gauge_discriminator_phase23_20260829_2c848055/runs/ordering_gauge_phase6_8790932.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`

The compact files here are byte-for-byte copies of the remote logs, histories,
tables, status, mapping, and provenance. `compact_sha256.txt` was generated in
the job and contains the expected self-referential entry created while that
file was open. `verified_compact_sha256.txt` excludes both manifests and
verifies all copied evidence plus this README and final qstat record.
