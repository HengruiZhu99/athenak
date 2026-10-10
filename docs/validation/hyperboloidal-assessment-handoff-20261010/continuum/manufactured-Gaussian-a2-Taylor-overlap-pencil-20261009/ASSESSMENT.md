# K32 Taylor overlap for the a=2 Gaussian slicing certificate

This is a source and pencil assessment of the exact v6 numerical bodies. It does not change any source, method boundary, recipe, threshold, active attempt, or failure status. No candidate was imported and no interval, Gaussian, target, unit, certificate or replay arithmetic was executed. Active outputs and the certificate JSONL were not read. The captured v6 index is `cafc4df240ba9988919f2f4d97522bda2f71a19f1899a625d2be98afa2144397`; the numerical producer and replay hashes are recorded in capture.json.

The Taylor proof extends beyond R=sigma
-------------------------------------

Write f(T)=sigma^4 exp(-T^2/(2sigma^2)), x=T/sigma, z=R/sigma and rho=R^2. The exact smooth integral is

C(R,T)=-(1/8) integral[-1,1] (1-v^2)^2 f^(5)(T+Rv) dv.

CT differentiates f^(5) once. For R>0,

Crho(R,T)=-(1/(16R)) integral[-1,1] (1-v^2)^2 v f^(6)(T+Rv) dv,

with its regular limit at R=0. The Gaussian Fourier moment bounds are global in T: the dimensionless even derivative bound is M_(2m)=(2m-1)!! and the odd bound M_(2m+1)<=2^(m+1)m!. Thus neither the Taylor theorem nor these derivative bounds require z<=1.

For C and CT, Taylor through degree 2K+1 and integrate the odd retained terms to zero. The remainder uses derivative orders 2K+7 and 2K+8. For Crho, Taylor f^(6) through degree 2K; the retained even terms vanish after multiplication by v, including degree 2K. The remainder uses derivative order 2K+7. The exact absolute moment

integral[-1,1] (1-v^2)^2 |v|^n dv = 16/[(n+1)(n+3)(n+5)]

therefore gives, with dK=(2K+3)(2K+5)(2K+7),

|error(sigma C)| <= 2 M_(2K+7) zhi^(2K+2) / [dK (2K+2)!],
|error(sigma^2 CT)| <= 2 M_(2K+8) zhi^(2K+2) / [dK (2K+2)!],
|error(sigma^3 Crho)| <= M_(2K+7) zhi^(2K) / [dK (2K+1)!].

These are exactly the expressions in producer_bounds.regular(), including use of the exact box Rhi rather than an outward-rounded endpoint for the remainder radius. The dimensional replay expresses the same theorem with an independently grouped Hermite polynomial and rho Horner expansion. The series coefficient identities and parity proof do not change when the permitted zhi is increased. In the current source, the z=1 restriction occurs only in coefficients()/lower_bound() dispatch and in root construction, not in regular() or the remainder proof.

Exact K32 bounds at zhi=2
------------------------

At K=32, M71=2^36 35!, M72=72!/(2^36 36!), and d=67*69*71. The three exact scaled remainder radii are

eC = 2^103 35! / [d 66!],
eCT = 2^31*68*70*72 / 36!,
eCrho = 2^100 35! / [d 65!].

No floating evaluation is needed for coarse useful bounds. Since d>2^18, every factor from 36 to 66 exceeds 2^5 and every factor from 36 to 65 exceeds 2^5,

eC < 2^(103-18-155) = 2^-70,
eCrho < 2^(100-18-150) = 2^-68.

For eCT, 68*70*72<2^21. Grouping the factors of 36! as 2..3,4..7,8..15,16..31,32..36 gives 36!>=2^(2+8+24+64+25)=2^123. Therefore eCT<2^(31+21-123)=2^-71. These are deliberately coarse exact inequalities; no numerical remainder evaluation or timing inference was performed.

For both admitted profiles sigma>=7/20>1/3, the corresponding ordinary dimensional remainder radii satisfy |error C|<2^-68, |error CT|<2^-67, and |error Crho|<2^-63. These bounds concern Taylor truncation only. They do not bound the width introduced by interval evaluation of Hermite polynomials, correlations among C/CT/Crho, the angular coefficients, or outward arithmetic. The tiny remainder alone is not a positivity or termination proof.

Two mathematically valid fresh policies
--------------------------------------

A minimal candidate is a fresh fixed method boundary at 2sigma. For each profile, closed roots [1/50,2sigma] and [2sigma,4], with the same T interval [0,4+8sigma], cover exactly the old compact domain. Their shared boundary is duplicate coverage, not a gap. Use the same K32, 256/384-bit producer/replay arithmetic, exact positivity test lower>0, and reviewed regional arguments. The producer and replay dispatches and root reconstructions must be changed coherently in a fresh source version. Current v6 cannot be reinterpreted as having this boundary, and its replay does not already admit the new version.

An alternative is an explicitly declared overlap sigma<=R<=2sigma. Both regular and separated formulas enclose the same exact functions there. For a cell wholly within that overlap, componentwise intersection of their C,CT,Crho intervals is valid; an empty intersection must reject the gate rather than choose one silently. The angular expression can be evaluated on the intersections. Another valid policy is to retain each complete lower bound and take their maximum: the maximum of two proved lower bounds remains a proved lower bound. A producer can try the regular enclosure first and compute the separated enclosure only if needed, but that is a new prescribed policy requiring exact transcript/method semantics and its own reviewed replay implementation.

A selected producer method must still be independently reconstructed by the replay's differently structured formula and yield a strictly positive replay bound. An overlap/intersection result is not an independent interval-library proof: both constructions still share the trusted primitives. If one implementation cannot reproduce a positive enclosure at the fixed budgets, the outcome remains unresolved. New method labels, witness interpretation, root construction and coverage checks require a coherent fresh admission, not a patch to a live certificate.

Practical recommendation and limits
-----------------------------------

The fixed 2sigma boundary is the simpler first source proposal: it avoids the inverse-power cancellation near the old join without necessarily paying for both formulas on every overlap cell. It is justified by the exact global Taylor remainder above. An overlap can be useful if Taylor dependency becomes large at some T, but adds cost and replay complexity. No speedup, leaf-count reduction, or successful certificate follows from this pencil. The Hermite recurrence and polynomial interval ranges can become broad over T boxes, especially away from the small-T region; enlarging the regular domain is not automatically monotone in runtime or enclosure quality.

The parent-reported negative interval lower bounds near R just above sigma do not demonstrate a negative physical D, and this review did not inspect those active cells. The physical domain, epsilon endpoint reduction, a=2 CMC endpoint, actual-layer concavity transfer, regional core/high-T/exterior arguments, and unchanged positivity thresholds would remain separate obligations. Only an accepted complete producer and independent replay, joined with those reviewed regional proofs, can support the manufactured slicing certificate. There is no native, wave-map PDE, original pulse, global evolution, or black-hole claim here.
