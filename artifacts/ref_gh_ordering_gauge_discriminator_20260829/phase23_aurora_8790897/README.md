# Aurora Phase 2/3 ordering-gauge discriminator

Aurora debug job `8790897` ran on node `x4610c7s7b0n0` from source commit
`68bc1ee30c0ac64c3afe8d3961bead65efce205a` and exited zero after 31 minutes
19 seconds.  The executable SHA-256 was
`eb189146c38de649cb9038f57211bb8bb33c2853f7af9537d8ca99e1dad50440`.

The full 96^3 fixed-point sector decomposition passed, then A and B reached
3M while C and D reproduced the invalid-timestep failure at 1.123484M.  See
`phase3_status.tsv`, each case history/max-location table, and the repository
analysis directory.

Large binary64 snapshots and restart files remain only on Aurora under:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_ordering_gauge_discriminator_phase23_20260829_2c848055/runs/ordering_gauge_phase23_8790897.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`

No cbin or restart payload is committed.  The committed logs, histories,
tables, provenance, scheduler record, and hashes are compact evidence only.
