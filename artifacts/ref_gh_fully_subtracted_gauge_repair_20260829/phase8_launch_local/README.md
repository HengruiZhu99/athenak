# Phase-8 launch/analyzer local checkpoint

This compact bundle validates the Phase-8 analyzer before any repaired
positive-time Aurora run.  The analyzer was replayed without altered inputs or
tolerances against two frozen committed trajectories:

- old gauge-enabled STANDARD Case D: correctly rejected at
  `t=1.103057103616269M`, with `old_mode_recurrence=true` and the exact prior
  GH e-folding time `0.037516548684379786M`;
- completed gauge-off Case A: correctly accepted through `t=5M`, with
  `old_mode_recurrence=false`.

The Aurora launcher is bounded to the repaired 96^3 STANDARD Case-D setup
through 3M.  It does not automatically continue to 5M.  This directory proves
only analyzer and launch-path preparation; it contains no repaired evolution
result.
