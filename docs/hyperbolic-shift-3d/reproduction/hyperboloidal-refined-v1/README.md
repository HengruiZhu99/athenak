# Hyperboloidal refined: integration and separate formulation experiments

Working branch: `hyperboloidal_refined`. Base: normalized-shift `fc9990546`;
merged performance/Cartesian-AMR head: `a2d648962`. The independent BN1 campaign,
its source/binary/launcher snapshots, submissions, checkpoints and monitor are
not modified. Campaign records under `boosted-normalized-shift-v1` remain frozen.

The integration and staged experiment are complete. The optimized halo path
is the default; the flat conformal frame remains a separate selectable
experiment. [Aurora comparisons](aurora-comparison/README.md) contain the final
measurements, plots and reproduction commands, with verified raw evidence
from jobs 9037790 and 9037836. Both jobs completed on one node with 12 GPU
tiles/ranks, and their repeated baseline checkpoints are byte-identical.
The final task footprint is 23.87 GB, below the 50 GB allowance.

At N16 through t/M=0.15, the flat formulation worsens constraints. Refining
both transition layers lowers final exterior H by 63% relative to flat alone,
but it remains 3.26 times legacy, with worse momentum and some scri residuals.
All recorded horizons pass the quality threshold; that early agreement does
not establish convergence or long-time robustness. The final placement is
available for controlled research, not promoted as an accuracy improvement.

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


## Separate flat-profile formulation

The parameterized RF1 experiment is described in [the formulation and parameter
audit](formulation.md). Its initial flat sphere is derived from the Cartesian
bounds, the old profile remains selectable, and shell interfaces are configurable.
The physical native spatial map is retained while the conformal frame, geometric
sources, initial fields and gauges change consistently. Explicit inner source
adjustments are necessary for nontrivial agreement with reference Cartesian Z4c.
The new formulation is isolated from legacy GPU compile-time constants.

[Local gates](local-gates/README.md) pass, including physical-frame constraint
invariance, derivative/map oracles, actual reference Z4c, device parity, MPI/AMR
restart and collective configuration rejection. A separate mixed-level flat
boosted AMR test also completes two steps. N16 two-step comparisons show a
constraint regression for the selected flat profile; radial redistribution
in the first Omega-focused trial reduces that error but does not recover the legacy result. No robustness or
convergence improvement is inferred from these short runs.

## Performance scope

The integrated default constructs the final halo plan directly and removes the
redundant discarded setup path. Earlier repeated measurements in the merged
[performance study](../performance-scaling-v1/README.md) found 1.627x fixed and
1.710x dynamic Aurora process speedups in short setup-heavy fixtures. Uniform
steady strong scaling reached 3.86x at 12 ranks and 4.08x at 24 ranks versus one
rank; the extra node reduced steady time by only 5.5%. These are the measured
performance-fixture results, not claimed speedups of the new flat formulation
or a long BN1 campaign. The scientific targets and frozen reproduction records
remain available independently.
