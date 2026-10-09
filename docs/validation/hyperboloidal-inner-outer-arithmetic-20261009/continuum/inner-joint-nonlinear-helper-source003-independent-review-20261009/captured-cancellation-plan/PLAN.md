# Held correction plan after the actual source002 oracle failure

No source/helper edits, scientific imports, numerical target recomputation,
compiler, query, Debug retry or evolution are authorized by this pencil.
The source001 compile failure and source002 genuine Release oracle failure
remain unchanged. This fresh plan requires root mathematical review before
implementation in another sibling and a separate exact execution release.

The saved source002 association summary records all15740 expected records,
3236 oracle failures (1926 source parts and1310 assembled source RHS),
with first/worst matches in the fixed small-alpha-large-chi family,
alpha=1e-150,chi=1e60 at radii.1,.5,.65. This is an association from saved
results, not an independently recomputed target or proof of unique cause.
Coefficient/principal/ordinary dual successes do not override that failure.

## Primary narrow correction to be reviewed

Use the exact same analytic gauge, coefficient k, C0 geometry/P/Theta,
frozen full reference connection, exact W=1 RWM short circuit and finite
contracts. Change only the algebraic evaluation of three field differences:
dA=alpha^2 chi-alpha_hat^2 chi_hat, the chi log-gradient difference dc,
and the lapse log-gradient difference dal.

Define Near(x,xhat) by1/2<=x/xhat<=2, testing positive finite PRIMAL values
without forming a ratio. With frexp(x)=m*2^e and frexp(xhat)=mh*2^eh:
if e-eh=0 it is true; if+1 require m<=mh; if-1 require m>=mh; all other
exponent differences are false. This exactly implements the closed interval
using normal mantissas/comparisons, with no underflowing half-products.
The branch is for fixed-reference field duals; each selected algebraic
formula must retain both dual components. No position/cutoff derivative
backend is introduced.

* dA: retain the existing factored deviation expression only if BOTH
  Near(alpha,alpha_hat) and Near(chi,chi_hat). Otherwise subtract the two
  separately exponent-scaled products Product(alpha,alpha,chi) and
  Product(alpha_hat,alpha_hat,chi_hat).
* dc_i: retain the existing deviation expression only if Near(chi,chi_hat).
  Otherwise use chi_i-Product(chi,chi_hat_i/chi_hat).
* dal_i: retain the existing deviation expression only if Near(alpha,alpha_hat).
  Otherwise use alpha_i-Product(alpha,alpha_hat_i/alpha_hat).

The branch criterion must not use only A0/Ahat. Anti-correlated alpha/chi
can keep A0 near Ahat while the factored summands are individually enormous.
Exact reference values select the original near branch, preserving its exact
zero value and its nonzero live-field dual tangent. The closed branch
thresholds alter no continuum formula; representational equality is not a
claim of universal correctly rounded finite-state source arithmetic.

Use this narrow three-difference correction first against the entire
UNCHANGED source002 suite, same15740 records, fixed cases, MP precision
assignments and all thresholds. In particular keep full source parts/RHS
entrywise-scaled2e-10, dual MP2e-10, principal2e-11, reference exact0,
core nonzero relative2e-10 and the three declared FD levels/final5e-7 plus
convergence/floor gates. Do not accept the previously failing family by
RHS-norm scaling, target cancellation, a case/radius drop or a relaxed gate.
The independent literal unfactored MP oracle remains byte-identical.

Record branch counts for dA/dc/dal and retain exact reference/W1 outcomes.
Any audit-output additions must be separately diffed; old queries/cases are
not overwritten. Unique new Release/Debug executables/dependencies/logs and
outer failure captures are required. No compiler/query runs under this plan.

## Explicit metric-contrast limitation and a separate coherent extension

The fixed high-contrast families keep metric geometry at the reference;
the separate SPD-tensor family varies it only moderately. Passing those
unchanged cases would NOT establish arbitrary simultaneous field/metric
contrast. The existing

    dV=dA*gInv+Ahat*dgi, dgi=gInv-gHatInv

can itself cancel enormous terms when A0 is tiny and gInv is huge. The
primary three-difference patch does not repair this untested regime and
must not claim it does. A separately reviewed extension is derived in
ALGEBRA.md. It would use a direct live-minus-reference tensor product far
from the reference and coherently regroup reference-gradient terms through
that same dV. Merely changing the dV assignment while leaving cancelling
dA/dgi gradient summands elsewhere is not complete hardening.

For a future additive coefficient-only witness, take the exact flat reference
alpha_hat=chi_hat=1,gHatInv=I, alpha=2^-200,chi=2^-100,
gInv=diag(2^501,2^-501,1). Then A0=2^-500 and the exact dV_11=1.
The existing large-term difference can lose that unit. This witness is an
analytic proposal, not a measured result. It need not solve any constraint
or call the full actual geometry/RHS; its role is a clearly named arithmetic
tensor check. A later actual nonlinear metric-contrast suite needs its own
representability/SPD/jet/source gate, complete dependency pins and release.

No additional witness or coherent dV extension is automatically appended to
the primary fixed suite by this plan. Root review decides that distinct
scope before implementation. No arbitrary-state accuracy, BH-data,
scri-regularity, global discretization or evolution acceptance follows.
