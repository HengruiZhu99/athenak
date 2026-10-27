# Global free-background derivative prototype

This separate, host-only prototype uses actual global active scalar values on
the frozen v15 research grid (N6, three central refinement levels, seven shells,
47,088 cells). It changes no production header. The selected M=1/3 isotropic
wormhole, Minkowski reference, lapse sources and positive log storage are fixed.
The measured initial background is a freely fitted spatial basis function,
not a prescribed evolving solution or a fixed puncture exponent.

Let X contain all 56 Cartesian P5 monomials, log(rho), and rho. Use positive
Cartesian coordinate-volume weights W, normalized over every active cell.
Remove the weighted X projection from known initial background B twice, obtaining
residual r. The free coefficient functional is

    a(q) = (r^T W q)/(r^T W r).
    D_new q = D_base q + a(q) (D_exact B - D_base B).

The actual stored q value is retained verbatim. D_base includes the same mapped
fourth-order stencils, global ghost interpolation, enriched logarithmic core
halos, supported58 puncture derivative rows and actual scri surface values used
for q. The background uses that same linear operator. No physical RHS residual
is subtracted. All coefficients are free at each instantaneous spatial
evaluation; this operator does not evolve a time-dependent field chart.

The normalized X matrix has condition2 799.6. Coefficient functional L1 is
16.1697 for log chi and 11.0661 for log(alpha/Aref). Constants, all X modes,
changing log powers and rho have coefficient magnitude below 3.74e-14; a(B)
agrees with one to about 1e-14. A stronger even nonspherical cosine control has
nonzero coefficient, so this is not a symmetry-trivial test. The first NumPy
matrix-multiply attempt emitted floating-status warnings despite finite values;
it is preserved. The qualified scalar repeat uses explicit reductions and is
warning-free.

The seven-query full derivative/source probe includes core, transition, another
shell and scri-neighborhood cells. Initial-B derivative errors fall to about
7e-17 in scaled units. It retains base truncation on P2/log/rho modes and does
not establish generic profile improvement: at point126, the changed profile
0.2 log(r+0.17) has H error -0.00882696 before and -0.00936730 after, a 6.12%
worsening. Angular and fractional-profile errors remain visible.

The complete physical effective weight calculation includes all active and
surface unknowns. Sparse base weights plus the rank-one functional are combined
exactly; no dense N-by-N operator is stored. The common normalization is
`h=1/(N max_A ||J_A||)`, not the Cartesian-only `J[0][0]` expression. Full-grid
gradient/Hessian L1 maxima are 224.069/461.276 for the base and 224.085/461.308
after correction. The same 5,200 scalar rows exceed the unchanged 32/64 caps:
5,056 Cartesian and 144 transition rows. There are no new failing rows, but
this remains a failed whole-grid amplification gate. The peak is Cartesian
point17922, h=1/48 and J diagonal8, so the dominant failure is not a shell
rotation normalization artifact. Seeded 1e-12 active/surface noise stays within
the exact functional bounds everywhere (maximum ratio0.9373); constant leakage
is below 9.8e-15. Noise control does not erase the amplification failure.

An independent full-grid repeat scales each derivative with physical-direction
spacing `h_i=1/(N sqrt(sum_A J[A][i]^2))` and each Hessian entry with `h_i h_j`;
it also checks common minimum-singular physical spacing. Both retain exactly
the same 5,200 failures under the literal 32/64 caps. Cartesian spacings agree
exactly, and the six face-axis permutation defects are below 1.33e-15. Raw
physical gradient/Hessian L1 maxima, 10756.1 and 2.32371e6, are retained rather
than hidden by unit changes. The peak Cartesian cell is near r=0.240036,
close to the outer region of the existing core value-halo treatment.

Only chi/lapse derivative jets are corrected in the all-cell initial comparison;
physical values remain bitwise unchanged, and every cell retains its positive
Penrose-volume weight. H RMS falls from 0.00299250 to 2.51e-15, but M RMS changes
from 0.03284268 to 0.03285356. With an independent analytic shift used only for
source isolation, Axx/Khat source error RMS worsens from 0.18277/0.16777 to
0.27037/0.24594. This exposes cancellation with the remaining numerical signed
field derivatives rather than proving all initial sources accurate.

The separate no-solve control retains reference beta=-x. It is not an actual RK
stage source gate: production solves the elliptic shift before evaluating RHS.
The subsequent actual-stage experiment performs exactly one frozen old-v15
vector solve after correcting scalar jets, with no analytic beta substitution.
It converges in 26 iterations, 10.29s, with original-equation residual RMS
2.69243e-11 and maximum1.27571e-9. Vector beta error RMS is 4.45277e-4 (max
0.00435970), radial error RMS3.93039e-4, and tangential beta RMS2.09264e-4 against
independent initial radial data. Actual solved-source RMS errors remain
Axx0.270763, Khat0.245928, lapse0.00126467, chi0.00393950 and Lambda-x0.0516287.
The old vector discretization and signed derivatives therefore remain limiting.

Evidence directories start `global-background-` under `evidence/`. The analysis
figure and JSON retain the failed profiles, full-region norms and all 24 source
channels. Compile-only failures, the scri audit-coordinate mistake, and the
first value-preservation rejection are kept separately; successful repeats
use distinct directories. Frozen source capsules and lossless raw-data copies
make the standalone commands reproducible without mutable production headers.

This prototype has not been adopted. Full generic profile accuracy, evolving
wormhole-to-trumpet regularity, nonflat nonspherical metrics, coupled RK
convergence and boundary closure remain unproved. Possible separately qualified
signed K/A background modes must retain independent unknowns and full equations;
they cannot be used as constraint projection or to hold the wormhole fixed.
