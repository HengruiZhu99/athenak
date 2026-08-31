# Aurora power-lag discriminator 8792940

This directory contains the compact evidence for the single-resolution,
fresh-data, compatible-Phi frozen/prescribed/feedback discriminator.  See
`docs/ref_gh_relative_damped_power_lag_20260831.md` for the claim-limited
scientific interpretation.

- Frozen reached `5.2M` finite.
- Prescribed was killed by the one-hour scheduler limit at `4.96109M`; it did
  not complete the requested endpoint.
- Feedback failed closed at stage `3.48005M`; it neither completed nor reached
  a fixed-`xi` dwell.
- Direct paired mismatch precedes constraint growth in both moving cases, but
  mismatch also grows more mildly in the finite frozen control.

`compact_sha256.txt` authenticates the in-job compact files as originally
written, including its empty restart-manifest placeholders.  The
`three_case_power_lag_reanalysis*` files are the expanded post-job, per-shell
analysis made from those authenticated histories.  The separately named
`restart_*_postprocess` files contain the real checkpoint hashes and sizes;
`restart_manifest_sha256.txt` authenticates them as produced by compute-only
job `8792997`.

The approximately 297 GB of restart files remain only under:

```text
/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_relative_damped_power_lag_20260831_9eff5b52/three_case_8792940.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
```

No restart or field dump is committed.
