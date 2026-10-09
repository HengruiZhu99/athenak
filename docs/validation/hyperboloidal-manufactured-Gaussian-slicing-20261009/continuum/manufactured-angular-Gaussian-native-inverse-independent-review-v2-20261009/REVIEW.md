The source/math and saved-data review pass for the declared finite native-time
Gaussian inverse/value consistency screen. This does not certify timelikeness
between samples, embedding jets, a full RHS, black-hole suitability, or stability.
No candidate import, multiprecision solve, CAS, kernel, compiler or evolution was
used in this review. The independent readback streams saved JSONL using only
stdlib JSON/Decimal arithmetic at 170 digits.

The fixed physical map is Y0=T+epsilon F(T,X), Yi=Xi, with
F=partial_X partial_Y[(f(T-R)-f(T+R))/R] and
f=sigma^4 exp(-T^2/(2 sigma^2)). The native inverse equation is
T+epsilon F=t_native+H(R), so reference/physical T is not native time.
The source uses the actual wide Minkowski layer (.05,.95), S=1, a=.5,
and keeps its prescribed reference throughout. J=1+epsilon F_T>0 is the
monotone time-map condition; D=J^2-|grad H-epsilon grad F|^2 is the separate
spacelike-levelset condition. The analytic Gaussian J lower bound and the
factored compact c_ret equation have the expected signs. The regular-origin
C coefficient is -2 f^(5)/15; the retarded/advanced radial formulas retain
both terms and the compact outer source retains the advanced tail.

The compact factors FT+FR=p z^2 Aplus and FT-FR=p z Aminus,
with z=1/R, give D=z^2 Delta without a cancellation-prone subtraction near
scri. The source's ADM values agree algebraically with the pulled-back
Minkowski metric: alpha_bar=Omega/sqrt(D), physical spatial metric
I-w w^T/J^2, beta_Q=-w/D, with the stated compact radial Jacobian. They are
exported only where D>0. No K/P/A/Lambda jets or evolved time rates are claimed.
For fixed p=n_x n_y, choosing n_x^2+n_y^2=1 gives the most negative angular
D term. This does not minimize D over continuous p, radius or time.

All 54,432 saved rows are present exactly once: four levels, each with
13,608 records, 21 radii, nine native times, nine p values, two sigma values,
and four amplitudes. All 32 level/profile summaries contain 1,701 events.
Saved initial/final numerical root signs, ordering, iteration caps, future
T>=0, J>0, original-map residual, finite positive-domain ADM scalars, and
all original precision/height/root/identity thresholds pass independently.
The reported brackets are numerical sign/width checks, not rigorous interval
enclosures; a collapsed bracket is bookkeeping rather than an interval proof.

There are exactly five negative-D events per level (20 saved records total),
all sigma=.5, epsilon=.75 and p=.5: (r,t_native)=(.65,.5), (.75,0),
(.75,.05), (.75,.1), and (.75,.2). The worst sampled D/reference is
-.25785359294197895 at (.75,.1,.5), despite positive J. Sigma=.35,
epsilon=.75 is positive at every declared sample, with minimum
.16161994896724030 at (.45,0,-.5). Sigma=.5, epsilon=.5 has sampled minimum
.15328765981288172 at (.75,.1,.5). These are sampled-domain statements only.

The maximum saved convergence discrepancy is 1.84378645e-55; the compact/direct
outer identity reaches 6.09583634e-45, the inverse/map residual 1.93063611e-56,
and width 5.00000000e-56, within unchanged 1e-30, 1e-50 and 1e-55 gates.
Decimal reanalysis reproduces all profile minima and event locations exactly.
It differs from the owner comparison maxima by at most 5.90e-112, due to
finite decimal serialization/arithmetic, without changing the conclusions.

The first independent reader checked all rows but failed its own exact
printed-maxima equality assumption. This failure is preserved in history.
The width discrepancy was 3.48e-137, and direct-identity discrepancy
2.48e-125. V2 uses explicit decimal serialization envelopes (ERRATUM.md),
while keeping every scientific threshold unchanged. That envelope checks
saved formatting consistency and is not a theorem about multiprecision
function-evaluation errors. All 145 original source/runtime/output pins are
unchanged before/after readback. The 199,965,069-byte samples JSONL remains
external large_payload metadata; it is never copied into this capsule.
