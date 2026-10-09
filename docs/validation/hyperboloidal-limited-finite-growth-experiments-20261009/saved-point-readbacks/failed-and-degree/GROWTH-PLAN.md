Source-only failed-payload physical constraint readback
====================================================

Status: HELD pending exact root authorization. No query, compilation, generator
spectrum, exponential, RK stage or time evolution is performed by this adapter.
The unchanged physical8 point maps were admitted independently in the analytic
projected-point stage. Both prior ordinary-FD attempts remain failed and the full
nongauge continuum comparator remains unresolved.

Input binding
-------------

`readback_growth_payload.py` pins the original failed growth payload
`J0-N8-rb98-growth001/failed-partial-payload.npz` (1f9036ed...), its receipt2774bc40,
admissionfa1f07b8, and the augmented sector-readback operator8b4b9a5b. It does not
silently follow later growth reruns. The preserved receipt failed inside SciPy
`expm` at t=.25 with a floating-point matmul error. This adapter requires that
failure and only the saved t=0 state. A readback pass never changes the failed
propagation classification.

The map/index/analytic receipt are pinned by `point_constraints.py`:
index970a117d, map44a6e015, analytic receipt98f18f39. The small NPZ is read locally;
all NPZ/NPY in this fresh stage are `large_payload` regardless of size. No frozen
index or verifier is modified.

Physical binding and algorithm
------------------------------

At21 points (r=.15,.30,.50,.70,.90,.96,.975; directions ex,(1,2,3)/sqrt14,
(2,-3,1)/sqrt14), reconstruct channel envelopes W,Wrho,Wrhorho using the original
normalized J0 Jacobi basis, N8/rb=.98. Channel order is alpha,metric_trace,P,
Theta_physical,beta,Lambda,metric_STF,independent_A, with L=0,0,0,0,1,1,2,2.
The solid angular basis already carries r^L. Do not multiply by another r^L.

Apply the saved real complete-lift constraint maps to each real and imaginary
column separately. Output raw physical Cartesian H,Mx,My,Mz,Zx,Zy,Zz,
Theta_physical, without Omega rescaling or rotating a local frame. Reference
S1/a.5/geometry .05-.95/C0/kinput10/physicalP/spatial-norm gauge xi2 is inherited
from the frozen map/source admission. Complete reference and angular coefficient
jets are already included by that map; this stage does not approximate them.

Read back eight saved physical seed columns, their saved-matrix J action, four
selected saved modal modes, the four independently saved actual Jv columns,
saved t0 seed states and their J action. Preserve seed amplitudes and mode
normalization. ActualJv is checked against the saved total matrix acting on v,
never replaced with lambda*v. The additional C(Jv)-lambda*C(v) residual is only a
finite-algebra diagnostic using saved lambda; it does not classify subsidiary
eigenmodes or continuum growth.

Gates and outputs
-----------------

Use allow_pickle=False and explicit finite/shape checks. Recheck EJbulk=Kweak,
EJsat=SATload, analytic nodal_from_modal T, and saved lower Cholesky LL^T=E.
Inherited algebra tolerance2e-9; analytic T and independent scalar-loop point
contraction tolerance5e-11. Saved seed nodes/t0 and actualJv use2e-9. Independent
scalar contractions use math.fsum and no einsum/BLAS. Gauge initial point vectors
(alpha,beta) must be exactly zero. No smallness condition is imposed on C(Jv) or
on the mode's constraints. No spectral or semigroup operation is called.

Output840 column/point records plus real/imag NPZ arrays, sample-coordinate
Euclidean H/M/Z/Theta summaries, checks, source/data before/after hashes, original
growth failure and launch HEAD. Coordinate sample summaries are not an integrated
energy or geometric covector norm. No conclusion about continuum, nonlinear,
exact-scri, CPBC or native stability is admitted. Later single-BH
wormhole-to-trumpet formation with the Minkowski hyperboloidal reference remains
an unresolved project requirement.

Authorization schema: failed_payload_point_readback_admitted=true and exact
driver_sha256,helper_sha256,plan_sha256,payload_sha256,operator_sha256. A fresh
output directory is required. Exceptions are retained in receipt.json and do not
overwrite any prior output.
