# Few-row localization of inherited amplification

The geometry-only standalone probe examines points0,6,126,14657,17922 of the
frozen N6/refinement3/seven-shell research mesh. It builds no evolution engine,
reads no physical field values and invokes no positive-field chart, logarithmic
reconstruction or scaling. Runtime is1.54s after a4.34s one-worker compile.
All original whole-grid amplification failures remain unchanged.

At the worst Cartesian point17922,
`x=(.0104166666667,-.0104166666667,.239583333333)`, the raw fourth-order
Cartesian stencil has scaled gradient/Hessian L1 of1.5/5.3333. Composing the
ordinary existing halo operator raises these to224.069/461.276. The
Cartesian-receives-shell contribution alone is223.319/457.360; same-level
Cartesian halos contribute0.75/1.6875. The axis-permuted point14657 gives the
same result.

The largest individual halo has L1=351.061 and216 shell donors. Its query
radius is0.260833, below every donor radius0.272946..0.444700. The actual
global radial-shell interpolation rule therefore extrapolates from shell-only
radial data across the Cartesian/shell junction. It does not use nearby
Cartesian active data that could bracket this query. The frozen source capsule
retains `CompositeDifference::InterpolateShellRadius` and the owner-tensor
interpolation implementation, so these claims can be checked against the
operator that produced the weights.

The shell-side failure is also interface dominated. At point6,
`x=(.256059310377,-.128029655188,-.213382758647)`, raw mapped L1 is1.7965/6.3666,
but ordinary composition gives64.777/156.531. Shell-receives-Cartesian terms
contribute63.390/151.765, while shell-to-shell terms are0.463/0.621. Its largest
Cartesian halo L1 is94.9324; every donor is from refinement-level1.

None of the ghost points read by these five query stencils belongs to the
selected radius0.24 positive-log halo override set. Thus their ordinary
composition is exactly the baseline positive-field operator at these points;
the failure predates positive reconstruction or the new rank-one correction.
Constants remain preserved within3.2e-17 in scaled derivative units, and seeded
1e-12 nodal noise obeys the exact functional bounds. Constant preservation alone
therefore does not establish good conditioning.

The next isolated hypothesis is a mixed Cartesian/shell physical-space P5
value interpolation for these same halo points, with complete polynomial
moments and unchanged rank, reproduction and16/32/64 physical amplification
guards. It would leave the mapped first/second derivative operator and its full
Jacobian/Hessian chain rule intact. No such replacement is adopted by this
localization experiment.

Run `python3 tst/elliptic_shift_3d/run_baseline_amplification_localization.py
--output <new-directory>` from the repository root after restoring the recorded
source capsule. Evidence is in `evidence/baseline-amplification-localization-20261027`.
