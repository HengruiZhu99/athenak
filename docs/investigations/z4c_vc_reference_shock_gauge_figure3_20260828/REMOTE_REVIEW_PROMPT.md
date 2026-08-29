# Read-only review prompt: same-tree Brill Figure-3 resolution campaign

Please perform a skeptical, source-and-artifact-only review of this AthenaK
campaign. You have read access only; do not assume you can build or run it.

Repository: <https://github.com/HengruiZhu99/athenak>

Branch: `codex/z4c-vc-reference-shock-gauge-figure3-20260828`

Immutable evidence/report commit:
`c122cc3717cb1ec3a954d62b90e3b01a4b66251b`

Qualified source commit:
`f8303c6be7eb214fa1e91b646123ee0d434b3698`

## Start here

- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/REPORT.md`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/REPORT.pdf`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/EVIDENCE_MANIFEST.json`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/N512_REPLAY.md`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/CONVERGENCE.md`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/analysis/aurora_n256_n512/final/comparison_summary.json`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/analysis/aurora_n256_n512/final/location/constraint_location_summary.json`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/analysis/aurora_n256_n512/field_patch/comparison/field_patch_comparison.csv`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/analysis/aurora_n128_n256_n512/final/three_resolution_summary.json`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/evidence/aurora/authority/n256_reference_shock_authority.jsonl`

The manifest SHA-256 is
`8f4790e76eca126070c6231c4c4998b7394e14ae270d1efbf31c37bdc7492360`.
It indexes 205 final reports, analysis products, compact histories, and execution
records. Intermediate partial-analysis directories are intentionally excluded.

## Controlled experiment

N128, N256, and N512 use the same physical MeshBlock bounds and complete N256
LogicalLocation history. Only cells per physical MeshBlock change: `16x16`,
`32x32`, and `64x64`. N128 and N512 accepted both replay events exactly; the
final tree has 212 leaves at physical level 5. The hierarchy authority SHA-256
is `7055de601e6181e5ad7e1432b5c20a111b0ba67e0e8d5377c170ea80e7bedcde`.

Fixed numerics are native vertex-centered Cartoon SO(2), O4, q6, RK4,
CFL `0.15`, KO `0.50`, shock-avoiding Bona--Masso lapse with `kappa=1` and
unit initial lapse, prescribed zero shift, telegraph lapse off, Z4c
`kappa1=kappa2=0`, and outer boundary `128 M`. No curve fitting or tuning was
used.

## Evidence to explain

| case | final proper time | peak proper time | peak log10 abs(Kretschmann) | max C squared integral |
|---|---:|---:|---:|---:|
| N128 | 19.33240 | 10.31396 | 4.29765 | 107.608 |
| N256 | 11.28631 | 10.30333 | 5.01349 | 48.2330 |
| N512 | 14.98253 | 10.30811 | 5.38112 | 4.09930 |
| published | -- | 10.30683--10.31384 | 5.47778--5.48688 | -- |

N512 directly resolves the deep minimum at proper time `12.62280`,
log amplitude `-6.07875`, and rebound at `13.21629`, log amplitude `-2.81849`,
inside the published timing/amplitude ranges. Nevertheless, C and H peak near
proper time `10.29711`, so the curvature peak is still constraint invalid.

Central curvature/lapse median Richardson orders are approximately `4.86/3.93`
for proper time 0--8, `3.34/3.36` for 8--10, and `2.10/1.40` for
10--11.286. Constraint-amplitude orders are positive but inconsistent between
the N128/N256 and N256/N512 pairs; full convergence is not established.

All 25 evolved fields were stitched over the finest `4<=rho<=6`,
`-2<=z<=2` patch. Shared vertices agree to roundoff, and no measured rho near 5
high-frequency branch strengthens with resolution. N256/N512 C/H maxima are
axis-adjacent and far from a coarse-fine interface; this is only a geometric
classification. The history norm already uses the proper axisymmetric ring
measure, so the growth is not a fictitious collapsed-y normalization effect.

N512 ran in six two-node Aurora `debug-scaling` segments. Segment 000's Athena
process exited cleanly and wrote a usable restart, but PBS exited `-29` during a
redundant second artifact-hash pass. Segments 001--005 and both N128 segments
are sealed with scheduler exit zero. Preserve this limitation in your review.

## Review questions

1. Does the same-tree N128/N256/N512 evidence justify the inference that
   bulk/parent under-resolution is a major driver of the N256 failure? What
   alternative explanation remains compatible with the monotonic improvement?
2. Why does the first-peak time converge closely while its amplitude and
   constraint contamination remain strongly resolution dependent?
3. Is the reported early O4-compatible behavior statistically/numerically
   meaningful, and is the stated loss of uniform convergence through collapse
   correctly bounded by the common N256 proper-time endpoint?
4. Inspect the native-VC Cartoon RHS, active-axis closures, shared-vertex
   ownership, same-level synchronization, and coarse-fine ghost/interface code.
   Is there a concrete source defect consistent with axis-adjacent C/H maxima,
   or only plausible mechanisms? Cite exact source locations and equations for
   any bug claim.
5. Could a stable interface error and bulk under-resolution coexist in these
   results? Explain which current artifact would discriminate them most sharply.
6. Is the proposed next step the smallest decisive one: a bounded shared-RHS
   audit at the retained N512 peak state, with separate active-axis,
   same-level-seam, coarse-fine ghost/interface, and clean-interior
   accumulations while all numerics and the exact tree remain fixed? Refine its
   decision table or propose a narrower source-level diagnostic.

Return a concise verdict with observations, inferences, and hypotheses clearly
separated. Identify the smallest decisive next diagnostic and its outcomes.
Recommend a correction only if the existing source and artifacts establish it.

Do not recommend floors, clipping, weakened chi/admissibility gates, broad
gauge/KO/CFL/AMR parameter sweeps, or another long production run as the first
step. Do not claim a constraint-qualified Figure-3 reproduction, uniform
convergence through collapse, horizon formation/absence, critical behavior, a
unique source bug, or exclusion of all axis/interface mechanisms.
