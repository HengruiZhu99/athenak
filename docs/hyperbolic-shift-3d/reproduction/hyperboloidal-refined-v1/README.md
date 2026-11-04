# Hyperboloidal refined: integration and separate formulation experiments

Working branch: `hyperboloidal_refined`. Base: normalized-shift `fc9990546`;
merged performance/Cartesian-AMR head: `a2d648962`. The independent BN1 campaign,
its source/binary/launcher snapshots, submissions, checkpoints and monitor are
not modified. Campaign records under `boosted-normalized-shift-v1` remain frozen.

## Merged baseline gate

The normalized-shift driver retains BN1, the exact boosted initial data, native
C6 areal 4M--12M slicing, normalized inner Gamma coefficient, original gauge
and restoration transitions, floors, signs, tracker, horizon policy and scri
operators. Direct halo construction removes redundant setup. The original AMR1
control remains independently reproducible, including its original gauge.

The local Serial/MPI checks passed before adding a new formulation:

* The two-rank BN1 N16 two-step checkpoint exactly matches the frozen one-rank
  checkpoint (SHA256 `8d78ce6cec0bab7c2edf5763b2d0e2300f5845747f5045dd88ba6f1fe15f44ee`).
* The nontrivial normalized-inner-driver oracle passes.
* Dynamic-wave and fixed-boost AMR checkpoints match prior optimized results;
  matched-rank histories also match exactly. Expected dynamic-boost rejection
  preserves its history, failed stage, rejected transfer and failure record.
* Native AMR transfer tests pass at one, two and four ranks; mapped AMR operators pass.

The first history comparison used different rank counts. Its constraint norms
show reduction rounding, and the existing history's CFL field is rank-zero local.
That mismatch is preserved in the evidence; matched-rank reruns pass. Complete
checkpoint diagnostics are collectively reduced. This is not a trajectory change.

See [completion](merged-baseline/completed.json), [bindings](merged-baseline/bindings.json),
[comparisons](merged-baseline/comparisons.json),
[matched-rank comparisons](merged-baseline/matched-comparisons.json), and
[commands/logs/raw histories](merged-baseline/raw-evidence.tar.gz).
These bounded checks establish merge compatibility, not long-time robustness
or convergence. New formulation validation and Aurora work follow separately.
