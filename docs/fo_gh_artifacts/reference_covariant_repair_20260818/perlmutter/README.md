# Perlmutter compact evidence

Allocation `57256611` used four ranks on node `nid001000`.  The captured GPU
mapping in `rank_gpu_mapping_verified.txt` records four distinct A100 UUIDs.

- `n{64,96,128}_t01.ref_gh.hst`: four-rank exact stationary-trumpet histories
  through `t=0.1` for `dx=1/16,1/24,1/32`.
- `n{64,96,128}_t1.ref_gh.hst`: corresponding histories through `t=1`.
- `restart_direct.ref_gh.hst`: newly generated `dx=1/16` direct run through
  `t=0.2`; it wrote three Ref-GH state checkpoints, each about 205 MB.
- `restart_resume_t01.ref_gh.hst`: continuation from the nonzero-time
  checkpoint (`t=0.1023608699`) to `t=0.25`.
- `n64_t20.ref_gh.hst` and `n96_t20.ref_gh.hst`: completed stationary
  `t=20` histories from allocation `57261202`.
- `n128_t4_paused.ref_gh.hst`: the fine stationary history stopped at
  `t=4.00184777` when allocation `57261202` was explicitly cancelled; raw
  restart checkpoints remain on Perlmutter and are not committed.
- `stationary_t20_pause_summary.tsv` and `PAUSE_AND_REVIEW.md`: compact final
  rows and the fail-closed review handoff.

The direct and resumed rows over their common times are identical to the
printed precision.  No raw checkpoints or large field dumps are committed.
