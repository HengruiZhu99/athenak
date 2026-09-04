# PC-GH AMR transfer discriminator

This directory preserves the one-step experiment documented in
`docs/pc_gh_qualification_log.md` under “2026-09-04 FO-GH damping and
post-projection transfer audit.”

- `baseline/` uses the production hard projections and the retained seven-operation
  reduction/curl monitor.
- `no-q-projection/` is a temporary diagnostic build that skipped only the hard `Q`
  reset while retaining the `p`, `L`, and `B` resets. That switch worsened both
  `curl(Q)` and `R_Q` and was removed from the source.

Both runs used one OpenMP rank with 12 threads, 456 MeshBlocks, no AMR changes, and
terminated normally after one RK3 step at `t=0.01082531755M`. The raw logs and
boundedness tables are retained; no restart or waveform from this discriminator is
used as qualification evidence.

`transfer_monitor.png` is the plotting-script smoke output from the baseline table. It
also verifies that the plotting path recognizes the new per-operation `curl(Q)`
columns.
