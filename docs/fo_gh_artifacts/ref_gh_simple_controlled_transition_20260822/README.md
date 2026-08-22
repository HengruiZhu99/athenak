# Simple controlled-transition evidence bundle

This directory contains compact, reviewable evidence for the 2026-08-22
Ref-GH simple dynamic-reference campaign. Large checkpoints and binary field
outputs remain on Aurora under:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_simple_controlled_20260822_1162bab4_v1/runs`

The final source-qualified PVC gate is `aurora_gate_8775444`. Its PBS job
finished with exit status zero on node `x4217c7s0b0n0`; the gate status records
the evolved and restarted controller generations. The checkpoint itself is not
committed, but its path and SHA-256 are retained.

Production directories are copied here only after a case completes. History,
mesh, provenance, performance, convergence, and hashes are retained; restart
and field files are intentionally excluded.

Controlling result: `SIMPLE DYNAMIC REGULARIZATION NOT ESTABLISHED`. The
open-loop T5 transition failed at all three resolutions near `t=1.62--1.65M`,
before the fit-shell safety condition permits feedback. Consequently no T6 or
T7 run was launched. See the top-level report and `first_bad_state.tsv`.
