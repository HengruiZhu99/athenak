# Complete transition derivative qualification

This standalone qualification covers all 432 cells on the first two radial
planes of all six cube-to-sphere faces, on the N6/refinement3, seven-shell,
47,088-cell research grid. Both log chi and log(alpha/Aref) retain their actual
stored values. Only derivative jets change. The physical equations, selected
M=1/3 initial slice, Minkowski reference and lapse sources remain unchanged.
There is no evolution or elliptic-solve acceptance in this experiment.

The original complete preflight failed 12 of 72 scalar plans. Euclidean nearest
125/175-point supports lacked one P5 direction at the second radial-plane
centers; the 256-point supports had gradient weight L1 about 58.24, exceeding
the unchanged cap 32. Those failures remain in
`evidence/transition-complete-qualification-20261027`.

The geometry diagnostic uses the same physical normalization
`h=1/(N max_A ||J_A||)` for every functional. At point127, the original mapped
fourth-order stencil has composed gradient/Hessian L1 of 2.1834/6.4118 and
Jacobian anisotropy 3.1133. The rejected 58.24 is therefore not explained simply
by noncubic shell cells. At a shell corner the original composed stencil reaches
18.4219/54.1258 with anisotropy 11.7016; physical cell shapes are not assumed cubic.

An explicit geometry catalog first tries Euclidean supports 125,175,256, then
target-Jacobian logical-cell distance with the same support counts. Every donor
is an actual global active Cartesian or shell cell. There is no lower-degree,
source-error or field-value fallback. All P5 columns and the freely fitted,
orthogonalized initial-background residual remain in the same 57-column span.
Condition<=1e8, reproduction<=1e-10 and physical value/gradient/Hessian L1
caps 16/32/64 are unchanged. Background coefficient normalization is recorded
separately; a coefficient cap that changes under B rescaling is not used as an
intrinsic physical stability claim.

All 864 scalar plans pass: 720 use Euclidean donors and 144 use the Jacobian
metric; 432/384/48 use 125/175/256 donors. Every face has complete coverage.
The independent 96/192-point background quadratures differ by at most 2e-15 in
value. Separate background nodal and query-jet perturbations, polynomial/log
reproduction, nonspherical profiles, and unknown noise are retained in the
face logs. Maximum conservative background-uncertainty physical-source change
is 3.92e-8 under the predeclared 1e-7 uncertainty budget. These are numerical
qualification controls, not weakened scientific tolerances.

The all-cell comparison retains positive Penrose-volume weights and every
active cell, including core, seams and scri. It uses an independent exact radial
shift in both variants solely to isolate geometry/source derivatives. Interface
H RMS falls from 0.0267026 to 1.17e-13. All-cell H RMS falls from 0.00299250 to
0.00188540; unnormalized L2 falls from 0.0190810 to 0.0120218. Momentum RMS stays
0.03284268. Other-shell H max 0.0290585 now dominates; core and scri norms are
unchanged. All 24 physical source channels and overlapping regional norms are
recorded. Initial physical rates remain nonzero; no RHS residual is subtracted.

Evidence and inspected figure are in
`evidence/transition-complete-analysis-20261027`; raw all-cell data are retained
losslessly in its NPZ and the source experiment CSV gzip. Each experiment has a
hashed source capsule and lossless raw-data compression manifest. The original
failed preflight is preserved. The corresponding runner scripts are
`run_complete_transition_qualification.py`, `run_transition_donor_probe.py`,
`run_complete_transition_balanced.py`, `run_transition_initial_norms.py`, then
`analyze_complete_transition.py` under `tst/elliptic_shift_3d/`.

This phase is ready for an optional discretization pilot, but it is not wired
into production. It does not establish coupled accuracy, arbitrary evolving
puncture regularity, moving-puncture safety or late-time convergence. A separate
global free-background low-rank prototype may offer lower memory cost; it must
be qualified independently before replacing these local plans.
