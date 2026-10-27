# Selected wormhole continuation

This continuation uses the existing `IsotropicWormhole` initializer with
M=1/3 and its derived k=.37754694621653295. It does not replace the slice by
CMC Schwarzschild, a trumpet, or time-symmetric data on the entire domain.
Only the initial data are sampled radially. The evolution keeps all six metric
and curvature components, three Lambda components, Theta, lapse, chi and three
flux components independent on the composite 3D mesh.

## Lapse and telegrapher

For z=|x|², Omega=(1-z)/2, A=(1+z)/2, T=(1-z)^4, the selected physical-log
restoring strength is xi=1+(3 nu-1)T, with nu=1. The complete lapse equation is

    alpha_t = beta^i alpha_i + S/Omega + chi partial_i(w B^i),
    S = -(alpha²+4 nu Omega² alpha)(Khat+3)
        -2 xi alpha(alpha-A) + alpha beta^i x_i + z
        + T z (alpha-A)/A.

The Cartesian telegrapher uses tau=.02, kappa=.005, componentwise beta transport,
and the previously audited C7 window w=1 up to r=.72 and w=0 beyond r=.8.
It is the reference divergence of a Cartesian vector, including the taper's
gradient. The radial 2B/r term is a consequence of that divergence.

For z<1/4 the code evaluates the algebraically identical combined source

    S = -alpha[(alpha+4 nu Omega²)Khat + (3+2 xi)alpha + D(z)]
        + alpha beta^i x_i + z[(1-T)+T alpha/A],
    D = z[(3nu-4)+(2-3nu)z+(2-6nu)z²+(9nu-3)z³+(1-3nu)z⁴].

The zero constant term of D and `1-T=z(4-6z+4z²-z³)` are canceled analytically.
This avoids losing the leading collapsed-lapse source by subtraction. The
independent spherical gauge audit covers r=1e-8 through .99, including both
sides of the arithmetic branch at .5. It measures a maximum relative lapse
error 5.53e-15. This is a gauge source change already selected in the spherical
configuration; no Einstein RHS or differential constraint is projected.

At scri, T and its first three derivatives vanish. Consequently the existing
tensor-pole normal derivative applies with multiplicative restoring and xi=1.
The lapse clock alpha=1 is still prescribed, and the unforced clock rate remains
a reported compatibility diagnostic.

## Positive coefficients and their limits

The generic Cartesian chart at a puncture position X uses rho²=|x-X|²:

    C = rho⁴/(rho⁴+Omega⁴), W = rho²/(rho²+Omega⁴), L = A W,
    chi=C exp(c), alpha=L exp(a), A_ij=W a_ij,
    Khat=W(p-3), Theta=W t.

Metric components and Lambda are unweighted and independent. Full analytic
first/second Cartesian derivatives of the factors enter the tensor jets.
The chart is positive at sampled physical points; it does not interpret the
missing rho=0 physical point as a regular center. No parity reflection or
beta=0 datum is imposed there. The generic reference supports a moving X by
explicit reference time derivatives; a separate finite-difference check gives
2.37e-12 absolute error in those derivatives. Moving-puncture evolution itself
is not validated.

An additional, explicitly selected initial-reference control replaces C and L
below r=.2 by the known isotropic inner expressions

    C_iso = k² r⁴/[Omega²(r+Mk/2)⁴],
    L_iso = Omega r²/(r+Mk/2)².

A compact C7 rational blend returns to the generic reference between .2 and
.3, before the physical initial-slice transition r=.3667508103655886. This
factorization retains the same physical fields and equations. It is not a
subtraction of the analytic initial RHS. Both choices and their failures are
retained as separate matrices. The matched reference is currently stationary
and specific to this benchmark; its parameters must be generalized before
using it for translating or multiple punctures.

The compact blend is **rejected** on these coarse grids: N6 initial H is about
1e-14 inside r<.1, but the unresolved blend produces a point maximum exceeding
1000 and full-domain H RMS near 100. It is retained as a failure control, not
selected to improve an accepted norm. A third, smooth factorization uses

    C_s = k² r⁴/[16 Omega⁶(r+Mk/2)⁴+k²r⁴],
    L_s = A r²/[r²+16 Omega⁴(r+Mk/2)²].

This captures the leading AF radial cusp without a compact collar and retains
the required scri jets. Its N6/8/12 initial H RMS is .388919/.199049/.047587,
while the coupled pole gates still fail; it is exploratory, not accepted.

**Minkowski reference equivalence:** the above factors are only coefficient
charts. The reference connection remains Cartesian flat, the reference lapse
and shift remain A and -x, and `WormholeLapseRate` keeps the selected Minkowski
reference sources. `UnpackComposite` reconstructs full physical jets before
every free tensor RHS and elliptic-source evaluation. No reference/background
RHS is subtracted. The separate chart-equivalence audit reconstructs 18
nondiagonal, off-constraint jets across all three charts, including scri. It
compares every regular/pole tensor component, the unchanged Minkowski gauge,
original vector source and null data. Maximum errors are 1.94e-12 relative,
3.09e-16 absolute and 2.23e-16 absolute respectively. The r=1 test compares
undivided sources and boundary data; it never evaluates an interior 0/0.

The coefficient functions can have radial cusps and angular limits. Cartesian
polynomial interpolation across the unresolved puncture is not an evolved
angular AF coefficient endpoint. Neither positivity nor survival of these
short pilots establishes changing wormhole/trumpet regularity. A complete 3D
AF endpoint treatment remains required. The finite-grid physical-volume
quadrature also changes as the minimum cell radius shrinks: that end has
infinite proper volume. RMS, point maxima, region diagnostics and the minimum
sampled radius must therefore be read together; no core cells are excluded.

## Budgets and evidence

The related even resolutions N=6/8/12 use eight Cartesian blocks and twelve
shell blocks, at 4,320/10,240/34,560 active cells. The two shell radii are .65
and 1. The minimum radii are .03608/.02706/.01804, compared with the initial
throat radius .06292449. These are exploratory coarse grids, not accepted AF
resolution. The duration is fixed at t=.0002 with dt=.0001; the middle-grid
dt/2 and elliptic-tolerance controls are separate. Each case has a 180-second
timeout, one Serial worker and no remote allocation.

`validate_wormhole.py` records all exits, complete stdout/stderr, commands,
source/executable hashes, exact initializer provenance and source stability.
The first rational-reference matrix has six component/initializer successes
and nine failed scientific gates; it has no timeout. A mistyped argument index
in the first runner attempt is preserved separately and corrected in the
reproducible script. Those invalid-argument exits are not numerical results.

At rational-reference N6/8/12 the initial H RMS is 1.21136/.892530/.546013;
M RMS is .903947/.516219/.133138. Initial tensor poles are
2.68721e-4/4.82300e-5/4.76446e-6, all above the unchanged 1e-7 target.
The t=.0002 pilots also fail that target. Actual ghost-reconstructed clock,
future-null and fixed-angle residuals remain at roundoff, showing that imposed
elliptic boundary data alone do not establish regular coupled evolution.
The N8 tighter-tolerance control changes H RMS at about 1e-15 and leaves the
tensor pole unchanged to about 2e-14. No late-time or accepted wormhole claim
follows from these runs.

Checkpoint version 6 records the positive-chart and reference choices, all
surface/bulk/flux state and the shift warm start. Readers retain explicit
versions 2–5 with their original meanings and validate source binding. The
runtime reconstructs geometry, interpolation and solver hierarchies on restart.
Native AthenaK restart/task integration is separate outstanding work.
