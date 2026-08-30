# Phase 1 Aurora W/R control

Aurora job `8791722` executed the frozen 96^3, 216-MeshBlock, 12-rank
one-cycle PVC control on node `x4216c4s7b0n0` with both sources built in the
same job and bounds checking disabled.

- W `68bc1ee30c0ac64c3afe8d3961bead65efce205a`:
  `PASS_ONE_CYCLE_PVC`, latest history time `0.003404497M`.
- R `3c9a34c8c3123c2570eb33e8ec77368feb1f1c61`:
  `FAIL_LEVEL_ZERO`, latest history time `0M`.

The complete run, build, and fence logs are retained here in deterministic
gzip form. `REMOTE_LOCATIONS.txt` identifies the full Aurora trees and
binaries. This evidence establishes a source regression only; it is not a
scientific stability or convergence result.
