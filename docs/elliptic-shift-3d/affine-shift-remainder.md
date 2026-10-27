# Isolated affine-plus-remainder shift representation

`AffineShiftRemainder` and `AffineShiftMoments` are new host-only prototype
classes. No current solver, pilot, boundary implementation, field storage or
checkpoint has been changed. They supply an invertible coordinate decomposition
and local arithmetic tests, **not an augmented elliptic solve**.

Write the global Cartesian shift as

    beta^i(x) = b^i + B^i_a (x^a−X^a) + u^i(x) = (A c)^i + u^i.

In matrix notation use normalized c=(b,RB) and A=(1,(x−X)/R) per
component. All three constant and nine anisotropic linear coefficients are free. There is
no beta=0 or u=0 puncture condition, parity, singular factor, conformal-flatness
assumption or regularity exponent in the representation. Its affine basis is
global, not multiplied by a compact cutoff. The compact support below selects
only the coefficient convention. It does not localize the physical affine field.

## Unique decomposition from conformal-volume moments

At a fixed elliptic stage, use the known evolved conformal metric to define the
local measure

    dmu = tau(|x−X|/R) sqrt(det(g_tilde)) d³x,
    tau(q) = (1−q²)^4 for 0≤q<1, and 0 otherwise.

This is **conformal** volume; neither chi^(−3/2) nor Omega^(−3) appears. R is a
positive support/grid scale, not a radius floor. With dimensionless basis
phi=(1,(x−X)/R), normalized discrete weights w=dmu/sum(dmu), define

    (M u)^i_a = sum_p w_p phi_a(x_p) u^i_p = 0,  a=0,1,2,3.

Let G_ab=sum w phi_a phi_b. Whenever the support contains four independent
Cartesian affine modes, G is positive definite. For any nodal beta the unique
canonical coefficients in normalized units are

    c^i = G^(−1) M beta^i,    u^i = beta^i − phi c^i,
    b^i = c^i_0,             B^i_a = c^i_(a+1)/R.

Conversely beta=u+A c, so the twelve moment equations remove exactly the twelve
redundant decomposition parameters without constraining the physical shift.
They are representation constraints, not differential Z4 projections. Compact
moment weights can vanish outside support, but **all physical u samples remain**
and are reconstructed. The test retains all 125 samples even where moment
weights vanish.

The implementation uses dimensionless G, a Cholesky factorization, explicit rank
and condition checks, and anchored moment right-hand sides. Normalization avoids
a misleading R² condition number caused solely by coefficient units; it does
not remove derivative amplification, continuum degeneracy or the conditioning of
a future full operator. Accumulation uses the C++ `long double` type, whose actual
mantissa precision is recorded by the test: on this Mac, both Real and long
double have 53 mantissa bits. No extra precision is assumed.

If u is already stored separately, `Rebalance` fits only that small field, adds
its affine projection to b/B, and subtracts it from u. It never reconstructs
large nodal beta to compute the new remainder. `FromNodal` is also provided, but
cannot restore information previously rounded out when a small u was added to
a larger nodal beta. Its limitations are explicitly tested.

## Derivatives, recentering and compatible boundaries

A caller differentiates **the stored u samples** and supplies their jet. The
full Cartesian jet is then

    beta = b+B(x−X)+u,   partial_a beta^i = B^i_a+partial_a u^i,
    partial_ab beta^i = partial_ab u^i.

The large reconstructed beta values are never fed to a finite-difference
stencil. Full metric-dependent operator terms still use the physical beta value
and its complete jet through native `ApplyEllipticShift`; the affine term is not
silently assumed to solve the curved-metric equation. In these tests its curved
operator reaches .0420, while its flat principal operator is zero.

For a new center X'=X+DeltaX, the exact coordinate change is

    b' = b+B DeltaX,   B'=B,   u'(x)=u(x).

The global affine field is unchanged. Moving the compact moment support changes
its coefficient convention, so a new moment plan and a rebalance follow this
recenter operation. The old plan rejects changed position, velocity, time, stage
or revision. This is an instantaneous spatial shift representation; there is no
extra evolution time source. Any later evolved-chart use would need its own
transport derivation.

For the current complete vector-value scri data beta=D, the compatible boundary
is

    u_boundary = D − b − B(x−X).

The helper supplies exactly this value conversion. Independent homogeneous u
data would instead impose beta=A c and generally change the physical boundary
problem. The tests reconstruct the native future-scri null/fixed-angle boundary
at all six coordinate-axis points with unit lapse clock. They do not solve a
local manufactured source together with this boundary, or establish a full BVP.
No puncture endpoint condition is supplied by the decomposition. In particular,
canonical b is not generally beta(X), because u(X) is unconstrained. A tracker
must use full reconstructed beta(X), not b alone.

## The necessary solve is still future work

If N denotes the total number of physical vector degrees of freedom, a real
integration must solve for N remainder unknowns plus 12 affine coefficients,
with the original physical equations and 12 moment constraints. Schematically,
including the original boundary rows in K,

    [ K   K A ] [ u ] = [ f ],
    [ M    0  ] [ c ]   [ 0 ].

Here K uses the full variable-metric L in the interior and the actual vector
boundary operator at scri. In curved geometry K A has nonzero interior terms;
a preconditioner cannot omit them merely because a flat affine field has zero
second derivatives. A block/Schur or equivalent augmented Krylov/multigrid
preconditioner, physical residual and boundary monitors, hierarchy transfers and
coarse treatment would be required. **None is implemented here.**

Warmstarts and restart must preserve u, b/B and the center separately; rebuilding
u from rounded physical beta can forfeit the benefit. Mesh changes require
consistent transfer of u and the affine representation and rebuilding moments
from the new known geometry. Stage-dependent conformal volumes can change c/u
without changing beta. This prototype implements no state migration, AMR,
restart or moving-puncture evolution.

## Bounded local falsifier

Reproduce after the existing CPU build is present:

    python3 docs/elliptic-shift-3d/evidence/affine-shift-remainder-20261027/reproduce.py

The test uses 5³ stencils at h=1e−2 through 1e−9, centers X=0 and
X=(.037,.013,−.021), flat and nonflat SPD metric jets, and queries at
(.5,.5,.5)h and (8.5,.5,.5)h relative to X. It builds 128 original/recentered moment
plans and runs 256 local vector cases. No entire fine grid is allocated. The
manufactures include nonzero constant shift, anisotropic linear shift, and

    beta^i = b^i + B^i_a y^a + (i+1)/16 rho^nu y^i
             + nonradial polynomial terms,   nu=.5,1.09.

These are representative powers, not universal hyperboloidal trumpet exponents.
The same native full vector operator acts on all approximate/analytic jets.
The manufactured right side is obtained from the analytic physical beta jet;
this tests representation/arithmetic, not an independent proof of that kernel.

The direct nodal control uses anchored fourth-order differences and wide-type
accumulation, rather than deliberately inaccurate unanchored constant
subtraction. The separate control differentiates u before composing its affine
jet. A third control decomposes already rounded nodal beta. The reported
truncation/coordinate control differentiates independently evaluated small
remainder data plus the analytic original affine part. It is not assumed to be
an arbitrary-precision oracle: world-coordinate subtraction, sampling and
finite differences still have their own errors. World points and remainders
use the same finite Cartesian coordinate representation.

All algebraic controls pass: maximum G condition is 519.74, normalized moments
are at most 1.03e−14 of the stored remainder, physical reconstruction differs by
at most 8.33e−17, recentering the affine field by 5.56e−17, and subsequent
canonical recenter/rebalance changes full nodal values by at most 1.12e−16.
Constant and anisotropic affine modes have exactly zero original-operator error
when stored as analytic affine coefficients. Native full scri boundary
reconstruction differs by at most 1.12e−16. Rank-deficient support, invalid
radius/volume, stale metadata and shape mismatch reject explicitly.

A representative translated nonflat result at h=1e−9 and the 8.5h query is:

| nu | direct nodal residual | separate residual | late nodal decomposition | truncation/coordinate control |
|---|---:|---:|---:|---:|
| .5 | 64.306 | .373353 | 65.847 | .373347 |
| 1.09 | 60.469 | 6.5132e−6 | 49.383 | 6.5131e−6 |

For nu=1.09, the direct arithmetic defect relative to the small-remainder
control is 60.469, versus 1.21e−10 with separate storage. Late decomposition is
still inaccurate because the nodal addition lost information earlier. At h=.01,
both representations agree to truncation accuracy; this experiment does not
attribute the current affordable pilot's core error to roundoff.

The physical residual need not improve in every case. For nu=.5 at the near
query, fractional discretization error dominates: at h=1e−9 the separate
residual is about 789, despite reducing the additional arithmetic defect from
about 91 to 1.1e−6. Some direct rounding errors happen to cancel truncation error;
that is not improved accuracy or robustness. Recenter/rebalance also introduces
finite arithmetic changes in the computed operator (maximum 2.16e−8 in this
family), retained in the evidence rather than claimed exactly invariant.

The affine split does not regularize rho^nu y, resolve a singular endpoint,
ensure uniform ellipticity, improve the underlying finite-difference truncation
order, or demonstrate safe wormhole-to-trumpet collapse. It is promising as a
nonsingular storage/preconditioning coordinate for future translating shifts,
provided the augmented solve and full coupled scientific gates are implemented
and validated. All current solver/pilot code remains unchanged.
