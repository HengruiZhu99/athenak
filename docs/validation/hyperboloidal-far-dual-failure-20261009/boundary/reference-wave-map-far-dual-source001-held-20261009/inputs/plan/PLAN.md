# Additive complete field-dual RWM validation — plan only

Status: HELD. No probe/oracle implementation, compiler, scientific import,
numerical/CAS evaluation, source query or evolution has been performed for
this plan. Root must separately review/release any implementation and execution.
The new outer001 source identity and the original15,740-record registry/oracle
remain byte-exact. Original source003 FAIL and its legacy W1 failures remain
protected; a new arithmetic suite cannot reclassify them.

The object tested is **direct `rwm::Gauge` from outer001**, with its unchanged
complete reference connection and assembled R+S/Omega, plus explicit old
LegacyGauge comparisons. It is not the compound inner gauge. In particular,
direct RWM queries at W<1 validate the underlying far arithmetic only; the
inner source still uses its source003 body there. Main reference/principal,
actual22, ordinary dual/FD and near-outer bitwise gates stay unchanged.
No far C0 geometry RHS/full22 acceptance is inferred: extreme live lapse/chi
can make unrelated geometric intermediates unrepresentable even when the
four RWM gauge rows and their derivatives are finite.

## Fixed positive-field base registry

Use the existing source003 State constructors without formula changes, with
a in{.5,2}, radius in{.025,.1,.5,.65,.84,.95,.995}, and directions
(1,0,0),(.36,-.48,.8). Use exactly the four existing high-contrast families:
collapsed,small-alpha-large-chi,large-alpha-small-chi,chi-gradient-contrast.
This is112 base states. Reference geometry remains S1,.05/.95, with complete
consumed reference jets; the gauge cutoff W(.45,.85) is reported as context.
All base states are joint-alpha/chi-far, which must be checked from actual
exported primal fields and the exact frexp predicate. The direct RWM itself
does not depend on G0; use the existing G0=.375 registry copy as provenance,
without implying a G0-dependent RWM equation.

No region, family or radius may be dropped following a failure. In particular,
r=.84 may already have W exactly1 in binary64; retain the actual stored W
rather than substituting a mathematical endpoint label.

## Complete field-dual seeds

Keep the reference position/parameters/coefficient jets fixed (zero field
dual), and register the same complete D primal+derivative scalar adapter as
outer001. Let e=(1,-1/2,1/4). Define independent seed parameters by

    adot=xi_a*a,       chidot=xi_chi*chi,
    (grad a)dot=xi_a*grad a+a*zeta_a,
    (grad chi)dot=xi_chi*grad chi+chi*zeta_chi.

The17 named seeds are fixed in CASE-REGISTRY.json: zero; alpha;chi;joint;
A0-balanced(1,-2);a*chi-balanced(1,-1);independent alpha/chi gradient seeds;
balanced A0 plus gradients;beta value;beta derivative;Lambda;physicalP;
metric STF;Theta-only;all-used-fields mixed;unconsumed-jet-only. The metric
seed is gdot=gE+Eg with E=diag(1,-1,0)/32, so tr(gInv*gdot)=0. No reference
derivative is silently varied. The mixed seed includes all gauge-consumed
live field groups simultaneously. Physical P and Theta stay independent;
the Theta-only tangent must give zero gauge derivative at fixed P.

Unconsumed-jet-only varies A and metric spatial first/second jets at finite
dyadic scales while keeping metric value unchanged. This checks the source
claim that only the inverse metric is used after Geometry.valid. It is not a
nonlinear Einstein/constraint-tangent or full22 gate. All complete input jets
must remain finite and Geometry.valid must be recorded; no NaN poisoning.

The17 seeds on112 bases give1,904 rows. Four additional controlled variants
per base set BOTH live alpha and chi gradient primals exactly0, then use
alpha-gradient-only,chi-gradient-only,balancedA0+gradients and all-used mixed
seeds. The resulting zero-primal/nonzero gradient duals are explicit and must
not vanish through a value-only Product shortcut. These448 rows are named
additive variants, not claimed to be unchanged original main primal states.
Total direct physical-reference field-dual rows:2,352.

## Independent targets and declared gates

The new target will be built independently from the literal physical-P RWM
live/reference rows, with a separate scalar MP dual implementation and
independently differentiated determinant/cofactor inverse. It will NOT reuse
GaugeFar, its scaled Product, NearFieldValue, its deviation graph or the old
inner-k correction. Exact binary64 exports are lifted to MP without decimal
parameter reinterpretation. The full raw regular/pole rows are given in
ORACLES.md. Every reference coefficient stays fixed in this local field dual.

Predeclare480/560 decimal digits for all2,352 direct dual rows; require
primal+dual entrywise scaled agreement <=1e-220. This is a numerical precision
cross-check, not an interval/error proof. It is intentionally more generous in
working precision than the original240/280 contrast oracle: live alpha²P
terms can be O(1e200), reference-deviation alternatives can be larger than the
finite answer, and balanced derivative directions can cancel leading terms.
No precision escalation or target omission is admitted by this fixed plan.

For all eight split parts and four assembled rows compare both components
against the independent target with the unchanged2e-10 entrywise scaled gate.
The scale is max(1,abs(native entry),abs(target entry)) for that single
primal or dual entry; no norm of another row is used. Record absolute errors
and a separately labeled sum-of-absolute-literal-terms diagnostic as well;
the latter must not replace the acceptance scale. Expected nonzero targets below binary64 range
are explicitly recorded; this scaled gate does not certify their relative
accuracy. The closed normal witnesses below provide a separate relative gate.
Any nonfinite target/output, invalid route, precision disagreement, unexpected
unrepresentable final target or declared comparison failure stops the gate and
is preserved. No floors, clipping or metric repair are allowed.

For every direct row record new and LegacyGauge primal+dual outputs separately,
old/new bit comparisons and old/target errors. Old far disagreement is expected
evidence, not a reason to alter the target. This supplement does not replace
the main near-state bitwise oracle. It does not establish universal conditioning
for arbitrary simultaneous metric and field contrast or absolute unit seeds.

## A small independent relative-alpha FD readback

Use only alpha-relative seed on16 already selected representatives: a=.5,
direction(.36,-.48,.8), all four families, r=.025/.5/.84/.995. Query the same
full field-linear direction u(s)=u0+s*udot at the five fixed relative steps
1e-3,5e-4,2.5e-4,1.25e-4,6.25e-5, centered plus/minus:160 additional direct
RWM double evaluations. Reference remains fixed; positive lapse/chi and SPD
must be checked for every side. Compare the full12-output derivative with
the actual D tangent, retain all five entrywise sequences, require final
scaled error<=5e-7 and first>=2*final or all levels<=5e-9. This is the same
FD acceptance form as the main gate, on a separately fixed direction family.
Reference and xyz stay zero-tangent. Branch derivatives mean derivatives
within the fixed primal far branch, with no claim of floating C1 continuity.
Main near-state value-zero parts can have a nonzero tangent; neither this plan
nor the existing Product contract equates value-zero with derivative-zero.

Do not impose a naive binary64 FD gate on every balanced seed. For example,
a(s)=a(1+s),chi(s)=chi(1-2s) has
A0(s)=A0(1-3s²-2s³): its exact first derivative is0 but centered FD gives
-2A0*s². At A0~1e50 this finite truncation term cannot meet a small absolute
zero-derivative gate. Chi-only source differentiation can also be hidden below
an unrelated O(a²P) constant in a double FD subtraction. The full independent
MP dual and closed targets still test those seeds; this limitation is explicit,
not a post-failure removal of any listed FD case.

## Closed normal flux and legacy negative controls

Add18 positive rows: three arithmetic contexts in ORACLES.md times the six
fixed relative seeds(0,0,0),(1,0,0),(0,1,0),(1,1,0),(1,-2,0),(0,0,1).
These use complete field duals and exact powers; independent Fraction closed
targets give primal1,-1,1 and the listed derivatives. Nonzero normal targets
use relative2e-10; exact-zero targets use absolute2e-10. Record all12 outputs
and require unused outputs exactly zero only where the supplied exact context
implies it. These are synthetic arithmetic contexts, NOT nonflat Minkowski
reference states or new physical-reference gates.

From the zero-seed18-row payload, emit three additional LegacyGauge negative
control records without another helper call. Exact old graph source lines and
their prerequisites must be byte-bound to the retained LegacyGauge. The legacy
primal results are0,0,2^301, while the closed targets are1,-1,1. Their failure
must be detected and labeled expected negative controls, never reclassifying
the old actual source003 FAIL or changing the new source targets.

The complete planned record count is2,373 (2,352 directD +18 closedD +3
negative references). Each positive D row records a new and old helper call;
the3 negative records reuse old exports. With160 FD-side evaluations this is
4,900 helper evaluations. No query has been performed.

## Provenance and later admission

Prepare a fresh probe/oracle/runner only after root plan review. Keep old001
headers/main registry/oracle untouched and bind their exact source index plus
independent review. Any compiled support must capture complete compiler flags,
dependency hashes, exact registered D/header binding, unique Release/ASan-UBSan
executables and pre/post inputs. Preserve actual main run status literally;
this separate plan does not invent a prerequisite main PASS or authorize any
execution. Root may release exact local stages after review. No operator,
spectrum, propagation, native option, scri class or BH claim follows.
