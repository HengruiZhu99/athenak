# All retained exterior-constraint histories: N16, N24, N32

`exterior-constraints-full.png` / PDF show all saved exterior measurements over
0–20M. `exterior-constraints-early.png` / PDF enlarge0–6M. Lines break at missing
values; the coverage strip distinguishes measured and missing diagnostics.
N16 completed20M; N24 failed after its last accepted time5.869M (last recorded
history row5.85M). The N32 running-job snapshot reaches3.6M. Its capture UTC and
hash are in n32-snapshot.json; this is not a final worker-verified completion.

All data sources, hashes, merged histories and finite-row counts are retained.
N16 and N24 through3M use the verified diagnostic replay with complete candidate
horizons. Other samples retain the frozen evolution's accepted-horizon rule.
In particular, N24 did not save any exterior norms after3M before its failure.
There is no way to recover those missing time samples from the existing scalar
histories; doing so requires a separate diagnostic replay. They are not replaced
by full-domain norms. Missing values are never inferred or interpolated.

The time series show strong refinement improvement at1.5M, but little exterior
H/Mr separation by2.7–3M. Spatial localization uses the matched3M checkpoints
with one common cutoff, documented in ../spatial-audit/README.md. The histories
use each run's own radius and therefore do not alone provide a common-domain
convergence-order measurement. All norms use compactified-coordinate volume,
not physical-volume weighting. Invariant and radial momentum are separate norms.

Run `python3 plot.py` to regenerate. N32 data are a static snapshot so that the
plot remains reproducible while the remote run continues.
