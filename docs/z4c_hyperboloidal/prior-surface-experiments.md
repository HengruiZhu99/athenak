# Prior standalone surface experiments

These retained N24 experiments predate the native AthenaK migration. They are
useful evidence for choosing later experiments, not native N12 performance or
correctness results. Their original source, reports and machine-readable results
are under the chat's `scaling/normal-matrix` and `scaling/normal-ablation`
directories. Exact paths and hashes are recorded in
`evidence/prior-standalone-surface-experiments.json`.

The direct normal-compatibility matrix was already derived and tested. It
replaced three perturbed full boundary RHS evaluations while retaining one full
base evaluation, the pivoted 3x3 elimination and the final corrected surface RHS.
The prior implementation already ran its four trial evaluations as independent
GPU work items. The analytic replacement passed its original declared numerical
checks but was slower in the paired complete-RK4 measurements:

| N24 implementation | 2 nodes, seconds/step | 8 nodes, seconds/step |
| --- | ---: | ---: |
| Four independent full-RHS cases, v8 | 0.33707804 | 0.17069231 |
| Analytic matrix plus full base RHS | 0.33912717 | 0.17406450 |

The local oracle covered 2,560 nonspherical states and 23,040 matrix entries;
actual GPU audit cases also checked every matrix against full-RHS differences.
Analytic-vs-control complete arrays were not bitwise equal, while candidate
repeats, poison cases and cross-rank states were bitwise. These claims belong to
the original experiment's validation policy; they do not bypass the native N12
gates.

A subsequent isolated experiment reused the inverse-metric contraction
`Q = x^i g_inverse_ij x^j` already produced by the base boundary evaluation.
It reduced full-step time by 0.716% at two nodes and 1.135% at eight nodes versus
that experiment's analytic control. Moving elimination into its own kernel
then changed time by +0.009% and -0.300%, respectively. The small eight-node
reduction was comparable to the observed repeat variation. Those variants were
bitwise equal under the original full-state checks. The conclusion was to keep
elimination fused unless a new measurement justified splitting it.

This last `Q` reuse is distinct from the earlier, broader reuse of a complete
conformal-metric Geometry across bulk RHS, damping and gauge calculations. Do
not apply the small normal-kernel improvement to the bulk reuse experiment, or
combine timings from separately controlled allocations.

Native `SurfaceExchange::Solve` currently performs four full evaluations
serially within each point's kernel body. Its scheduling and ownership differ
from the old v8 control: N12 has six block-local surface owners with144points
each at both24 and96ranks. First measure the native implementation. The previous
result neither proves that analytic assembly will be slower here nor justifies
expecting a fourfold whole-step speedup.

The old derivation also identifies an unimplemented option: compute the base
compatibility defect from only chi_t, the contracted metric rate Q_t, and the
shift gauge/restoring terms. That could avoid curvature and unused tensor rates
in the compatibility solve. The complete corrected final surface RHS remains
necessary to evolve the other fields. Any native specialization still needs its
own derivation, invalid-state behavior checks, full-state comparison and paired
timing. No such specialization is introduced by this evidence note.
