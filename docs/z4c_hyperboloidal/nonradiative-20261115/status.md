# Completed nonradiative validation — 16 November 2026

**Unchanged scheme A is not validated at the tested resolutions.** See the
[final report](report.md) for the test matrix, results, plots and limitations.
Exact-derivative analytic extraction recovers zero news, and smooth rotating
and retarded-bouncing donor replays converge. Literal gauge-wave donor replay
is underresolved and fails; native Schwarzschild/gauge runs also fail to
establish a convergent zero-news/full-regularity limit. This does not prove
that continuum scheme A has no regular solution.

All task jobs have finished. The principal control/gauge runs at N12/N16/N20,
N12 half-timestep run, cadence/angular controls, analytic follow-up, and
zero-step initial-constraint checks are complete. Fifty-seven analytic case
products are retained, along with four H1 spacelike-slice rejection attempts.
Reduced coverage and interrupted/recovered cases are explicit in the report.

Every task allocation used two Aurora debug nodes, sequentially; total
PBS-reported consumption is 2.66444 node-hours, including failed attempts.
No unrelated jobs or files were changed, and no binary/radiative campaign ran.

The 14 focused CTests and existing frame/news regressions pass. Diagnostics
off/on leave 24,704,784 checkpoint scalars bitwise equal. Scheme A's equations,
gauge parameters, projections, penalties, lifts and full overlap are unchanged.

Raw local results: `/Users/hz0693/research/results/nonradiative-20261115`.
Raw Aurora results: `/lus/flare/projects/CompactBinaryMerger/hzhu/nonradiative-20261115`.
Compact results and source/build/job receipts are in
[the evidence directory](../evidence/nonradiative-20261115/README.md).
[reproduce.md](reproduce.md) gives commands and provenance limits.
