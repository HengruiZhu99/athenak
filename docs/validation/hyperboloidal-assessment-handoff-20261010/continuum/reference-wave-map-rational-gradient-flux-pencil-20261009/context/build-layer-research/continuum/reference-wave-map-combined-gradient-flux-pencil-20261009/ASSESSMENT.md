# Combined gradient flux: a bounded arithmetic proposal

This is a source/pencil proposal, not an implementation or a claim that the completed far-dual failure is repaired. The parent reports 63 failed tangent comparisons in the completed Release gate, with primals and the independent two-precision targets passing. The separately frozen saved mapping identifies metric-STF seeds in all 63 failures, including eight zero targets and several pole components. Thus the gradient flux below is a concrete candidate site; determinant/inverse tangents, dV/dL and later contractions remain independent possible sources. No failed gate is relabeled and no actual payload is decoded here.

Let a be the live conformal lapse, x the live chi, G the inverse conformal metric, and h,y,Ghat their fixed reference counterparts. Write a_j and x_j for the submitted ordinary spatial derivatives, independently seeded in the field dual. The current far regular shift includes

    sum_j [(a^2/2) G^ij x_j - a x G^ij a_j
           -(h^2/2) Ghat^ij y_j + h y Ghat^ij h_j].

The exact algebraic grouping is

    L_j = (a^2/2) x_j - a x a_j,
    Lhat_j = (h^2/2) y_j - h y h_j,
    gradient_row_i = sum_j [G^ij L_j - Ghat^ij Lhat_j].

This uses no lapse/chi division, log-gradient quotient, field floor, on-constraint substitution or reference subtraction beyond the existing expression. Physical P, reference P, Theta, advection, Lambda, the nonflat reference connection, all pole terms and the legacy-near branch are unchanged as real equations. A future source change must state separately which rounded operations change; the current scaled Product bodies cannot be called exact mantissa products.

For an arbitrary local field direction denoted by a dot,

    dot L_j = a dot a x_j + (a^2/2) dot x_j
              -dot a x a_j - a dot x a_j - a x dot a_j,
    dot gradient_row_i = sum_j [dot G^ij L_j + G^ij dot L_j].

The reference tangent is zero in the accepted supplement; a more general adapter must differentiate it too. For the metric-only direction, dot L=0, so the live tangent contracts dot G with the already combined L. This exposes the cancellation before the large metric tangent multiplies the two contributions. An exactly zero primal gradient or L must not erase a nonzero seeded tangent. Merely replacing the current expression by ordinary binary64 `G*(.5*a*a*x_j-a*x*a_j)` does not suffice: its inner products can lose the residual, overflow/underflow individually, or produce a rounded zero that subsequently multiplies a large G.

## Exact bounded specification

A CPU-only signed-sum-of-products primitive is a clear correctness specification for this site. It accepts a short fixed list of signed monomials in finite binary64 atoms and returns the correctly rounded exact sum. Each atom is decoded as a signed integer M times 2^e, with at most 53 significand bits; powers of two such as one half are exact exponent shifts. Multiply integer significands, add exponents, align the finitely many terms to the smallest exponent, and sum signed integers. Only then round once to binary64, ties to even. For this gradient contraction each full monomial has at most four nonconstant atoms. The exponent range and integer workspace are therefore bounded by the binary64 format and fixed arity, rather than by a data-dependent precision loop. An implementation may use an explicitly bounded limb accumulator; importing a multiprecision header would be a new reviewed dependency, not an implicit permission.

The forward-dual primitive constructs the exact product-rule list from all factors and sums it with the same accumulator. Repeated a factors supply both terms, or may be combined to the exact coefficient two. A zero primal factor does not discard its differentiated product. No division by a primal factor is used. The strongest version expands `G*L` and its derivative into the accumulator before rounding, instead of rounding L to binary64 first. Per row this is a small fixed set (six live primal monomials over three spatial indices, and at most twenty-one live tangent monomials). The reference terms can enter the same accumulator when cancellation against reference is relevant. A rounded L-only implementation has the weaker claim of grouping improvement and must retain the propagated rounding bound `sum_j |G^ij| error(L_j)` plus contraction errors.

For the final integer N times 2^E, ordinary normal output is rounded to its 53-bit significand by quotient/remainder and even-significand tie handling. Subnormal output is rounded on the exact 2^-1074 grid, including carry to the minimum normal. The implementation must distinguish exact cancellation (+0 by declared convention) from nonzero negative values rounding to -0. Exact half-minimum-subnormal and normal-boundary ties need explicit cases. An exact out-of-range result is rejected with a recorded overflow status; a small legitimate result is rounded, not floored. Invalid/nonfinite input is rejected. Counts and exact rational/exponent bounds record rounding-to-zero and subnormal output without a global warning suppression. The ordinary branch, if retained, needs a proved condition for its claimed error or exact-result agreement, not a primal-nearness heuristic.

## FMA alternative and its limits

An efficient alternative is a power-of-two scaled floating expansion: split products with TwoProductFMA, accumulate residual components with TwoSum, and keep exponent-tagged components until their final rounded sum. On a safe normal scale, `p=RN(u*v)` and `e=fma(u,v,-p)` recover the product as p+e; repeated expansion multiplication handles the three/four-factor monomials. Exponent bins or an exact fallback are still required when aligning widely separated terms would discard a component which later survives cancellation, or when a residual/product is not representable on the chosen normal scale. Fixed twofold precision has an error bound, not unconditional correct rounding for arbitrary cancellation. A lone `fma` around one already rounded product also does not recover all missing product errors.

This distinction is supported by [Ogita, Rump and Oishi, Accurate Sum and Dot Product (2005)](https://ogilab.w.waseda.jp/ogita/math/doc/2005_OgRuOi.pdf), especially Algorithms 3.1/3.5 and the underflow qualifications in section 3. Their error-free transformations motivate the expansion route; they do not prove this new multi-factor/dual implementation or guarantee its exponent handling. No source quotation or algorithm execution was used here.

## Admission sequence and unresolved geometry

First require the owner's pinned 26-context stage diagnostic, with literal input atoms, the returned determinant/inverse primals and tangents, each live/reference gradient monomial, L, dV/dL, and pole contractions. Compare each stage independently to the Fraction/rational input interpretation. This separates a lost difference of products from an already inaccurate inverse tangent and from later row cancellation. It must retain all eight exactly zero targets. The exact accumulator specification is exact only for its submitted binary64 atoms; it cannot recover information already rounded out of Geometry or a previous dV/dL.

For example, an exact metric-STF seed H=gD+Dg with trace D=0 satisfies dot det(g)=0 and dot G=-G H G in real arithmetic. A tiny nonzero computed determinant tangent can be amplified by extreme coefficients. Conversely, using rounded G in a locally exact contraction still differs from the rational inverse of the original g. These are not resolved by regrouping L. Any geometry change must have its own exact rational oracle, determinant/positive-definite validity rules and independent source review; no determinant tangent may be forced to zero by recognizing a test seed.

A first arithmetic unit gate should cover exact equal products, a residual many ulps below either product, all sign patterns, exponent gaps, individual-product overflow with finite combined result, individual-product underflow with a representable combined result, normal/subnormal ties, signed zero, invalid input and genuine result overflow. Forward-dual controls must include zero primal/nonzero tangent, repeated factors, cancellation in the tangent only, metric-only, value/gradient-only and mixed seeds. An independent bit-pattern/rational rounding oracle should differ structurally from the candidate's integer rounding routine. FMA-specific controls need explicit contraction/compiler flags and residual-normality evidence; failed conditions go to the exact path and are counted.

Only after separate source/admission review should the original 2,373-record/4,900-call gate be rerun in a fresh attempt with the same targets and tolerances, the 16 FD representatives and closed witnesses, exact near-legacy controls and all raw failure history. Keep stage counters and report zero-target errors separately. A local pass would establish only the declared finite-Omega fixed-field arithmetic coverage. It would not establish compound-inner/full22 behavior, uniform conditioning for arbitrary data, native stability, scri continuation, or wormhole-to-trumpet black-hole adoption with the retained Minkowski reference.
