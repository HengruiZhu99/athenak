# FO-GH validation artifacts

These are captured outputs for commit
`1e4c62ee25dd443334f8ddab1a27ad4d697e21d7`.

- `perlmutter_20260817/` contains one-A100 build/runtime provenance, raw logs,
  history files, checkpoint summaries, convergence summaries, and GPU
  telemetry from allocation `57187283`.
- `local_boundary_control/` contains the same-spacing doubled-domain control.
- `local_parameter_sweep/` contains bounded `N=16`, `t=5M` diagnostic runs.

The Perlmutter source and executable hashes are recorded in the evidence
directory.  Large binary restart checkpoints are not copied into Git; their
hashes and retained remote location are recorded instead.

These artifacts document both passing checks and the failed `5M` convergence
gate.  They must not be interpreted as long puncture, SMR, four-GPU, trumpet,
or production qualification.
