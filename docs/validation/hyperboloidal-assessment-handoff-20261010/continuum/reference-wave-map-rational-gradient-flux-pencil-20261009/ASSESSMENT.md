# Retaining the inverse rationally through a complete gradient row

This is a mathematical contract proposal, with no implementation, evaluation, compilation, or gauge replacement. The completed 63 far-dual failures remain failures. The frozen 26-context diagnostic distinguishes contamination introduced by the inverse from further cancellation after holding the native inverse fixed. Correctly rounding each inverse entry and then summing products exactly addresses only the latter submitted-atom problem; it does not generally give the correctly rounded contraction of the inverse of the original metric.

## Proposed arbitrary-input contract

Let `A` and `Ahat` be two nonsingular real 3x3 matrices whose entries are finite binary64 atoms. Let `a,chi,a_j,chi_j` and their hatted counterparts be arbitrary finite binary64 atoms, and let every atom have an independently specified finite first-directional derivative. In the actual gauge, `A` is the submitted conformal metric, `a` is conformal lapse, and `a_j,chi_j` are submitted ordinary spatial derivatives. The reference direction is normally zero, but the general contract does not require it. Nonsymmetry, non-unit determinant, nonzero determinant derivative, zero primal gradients, repeated factors, and mixed seeds are allowed. A separate physical caller may impose its existing metric-validity conditions. No seed, trace-free identity, on-constraint relation, or reference equality may be recognized to change the algebra.

For each output row, define the exact real functions of the submitted atoms

    L_j = (a^2/2) chi_j - a chi a_j,
    Lhat_j = (ahat^2/2) chihat_j - ahat chihat ahat_j,
    Phi_i = sum_j [(A^-1)_ij L_j - (Ahat^-1)_ij Lhat_j].

The proposed outputs are the binary64 round-to-nearest, ties-to-even of `Phi_i` and of its exact analytic first derivative. The latter is not the derivative of an already rounded inverse or rounded primal result. An exact zero result has the declared `+0` convention; a nonzero negative result rounding to zero yields `-0`. Nonfinite atoms, exact singularity, or an exact result beyond the declared maximum-finite contract produce explicit rejected status. There is no determinant floor or condition-number cutoff. Near-singular nonsingular inputs may still have enormous outputs; valid input does not guarantee a representable output.

The product rule retains

    dot L_j = a dot(a) chi_j + (a^2/2) dot(chi_j)
              - dot(a) chi a_j - a dot(chi) a_j - a chi dot(a_j),
    dot Phi_i = sum_j [dot G_ij L_j + G_ij dot L_j
                      - dot Ghat_ij Lhat_j - Ghat_ij dot Lhat_j],
    dot G = -G dot A G.

A zero `L`, metric entry, or primal factor does not discard any differentiated product. The proposed contract is the complete live-minus-reference gradient contraction in the far regular shift, not its individual large terms rounded separately.

## Exact common-scale representation

Every finite binary64 atom, including every directional atom, is an integer times `s=2^-1074`. Its integer magnitude is strictly less than `Bmax=2^2098`. Write

    A=s B,       Ahat=s Bhat,
    D=det B,     Dhat=det Bhat,
    C=adj B,     Chat=adj Bhat.

Here `adj` already includes the cofactor transpose. Thus `G=s^-1 C/D`; no rounded inverse entry is formed. With integer lapse/chi/gradient atoms `b_a,b_chi,b_aj,b_chij`, form

    ell_j = b_a^2 b_chij - 2 b_a b_chi b_aj,
    N_i = sum_j C_ij ell_j,
    Nhat_i = sum_j Chat_ij ellhat_j,
    P_i = N_i Dhat - Nhat_i D,
    Q = D Dhat.

Then the exact complete row is

    Phi_i = 2^-2149 P_i / Q.

Differentiate these integer polynomials with the complete unsimplified product rule. The common scale `s` is constant, so

    dot Phi_i = 2^-2149 (dot P_i Q - P_i dot Q) / Q^2.

This postpones both inverse-entry rounding and cancellation against the reference until one final rational rounding per output. Normalize the primal denominator sign once; `Q^2` is positive. Exact `D=0` or `Dhat=0` is rejected before division. The derivative formulas are valid for arbitrary directional matrices and scalar/gradient directions, including nonzero `dot D`.

## Explicit finite bounds, without selecting a new implementation

Treat a repeated atom as a repeated factor, with integer coefficients represented exactly. A determinant has degree 3 and absolute coefficient sum 6. A cofactor has degree 2 and coefficient sum 2; `ell` has degree 3 and coefficient sum 3. Consequently each `N_i` has degree 5 and coefficient sum at most 18. The following conservative strict magnitude bounds follow solely from `|integer atom| < 2^2098`:

| Polynomial | Degree | Absolute coefficient-sum bound | Integer magnitude bound |
| --- | ---: | ---: | ---: |
| `D` | 3 | 6 | `<2^6297` |
| `N_i` | 5 | 18 | `<2^10495` |
| `P_i` | 8 | 216 | `<2^16792` |
| `Q` | 6 | 36 | `<2^12594` |
| `dot P_i` | 8 | 1728 | `<2^16795` |
| `dot Q` | 6 | 216 | `<2^12596` |
| `dot P_i Q-P_i dot Q` | 14 | 108864 | `<2^29389` |
| `Q^2` | 12 | 1296 | `<2^25187` |

For a degree-`d` polynomial, differentiation replaces each of its `d` factors in turn, so the coefficient-sum bound is multiplied by at most `d`; directional atoms have the same format bound. The tangent numerator bound uses `1728*36 + 216*216 = 108864`, retaining both differentiated-denominator terms. These are format/arity bounds, not measured conditioning or numerical error bounds.

Both the primal and tangent have the form `2^-2149 U/V`, with positive denominator after sign normalization. For subnormal rounding, the exact number of minimum-subnormal units is `|U|/(V*2^1075)`. Quotient, remainder, and ties-to-even therefore suffice; the half-minimum-subnormal comparison is exact integer arithmetic. For a normal exponent `e` between `-1022` and `1023`, the significand ratio is `|U|/(V*2^(e+2097))`; the required denominator shift lies between 1075 and 3120. Overflow under the strict exact-result contract is the exact comparison

    |U| > V (2^53-1) 2^3120.

The tangent numerator needs at most 29,389 bits; its denominator with the maximum significand-grid shift at most 28,307 bits, and the strict overflow comparator at most 28,360 bits. An eventual implementation must separately prove every multiplication, sign, carry, quotient, remainder, and exponent-selection workspace. No limb allocation or rational implementation is authorized here. In particular, the already reviewed four-factor short signed-product primitive is not silently extended to degree 14 or to a rational denominator.

## Comparison with rounding the inverse first

Let `K=RN(G)` and `dot K=RN(dot G)` entrywise, and suppose a short-sum primitive subsequently contracts these atoms with the exact submitted scalar/gradient atoms. Its real target differs by

    sum_j [(K-G)_ij L_j - (Khat-Ghat)_ij Lhat_j]

in the primal, and by

    sum_j [(dot K-dot G)_ij L_j + (K-G)_ij dot L_j
           -(dot Khat-dot Ghat)_ij Lhat_j -(Khat-Ghat)_ij dot Lhat_j]

in the tangent, before final contraction rounding. Entrywise half-ulp bounds do not control relative error when the exact complete row cancels. Even a correctly rounded inverse can therefore lose a small or zero final tangent. The rational route has the stronger final-row target because it retains those cancellations before rounding.

It still cannot recover information already rounded out of the supplied metric or gradient atoms. It does not automatically repair the other `dV`, `dL`, Lambda, reference-connection, physical-P, advection, or pole contractions. Those parts of the actual gauge must remain explicit independent contracts and diagnostics. No claim follows that replacing this one gradient row would pass all 63 failed comparisons.

## Later tests and scope

A separately reviewed unit source could use exact Fraction dual Gauss-Jordan elimination with nonzero primal pivot selection, then contract the full rows in Fraction dual arithmetic and use a structurally independent rational-to-binary64 oracle. It should cover general nonsymmetric and traceful seeds, zero-primal/nonzero-direction factors, dense reference/live cancellation, exactly zero targets, near-singular nonsingular metrics, finite outputs despite individual overflowing/underflowing terms, normal/subnormal ties, signed zero, invalid inputs, exact singularity, and genuine final overflow. The old inverse-entry-plus-short-sum result can be retained as a weaker comparison, not treated as the primary oracle.

Any later gauge proposal needs its own source/admission review and fresh original 2373-record/4900-call gate with unchanged targets, tolerances, 16 finite-difference representatives, closed witnesses, and near-legacy controls. Neither a standalone rational unit pass nor a local gauge pass establishes continuum/native stability, scri continuation, or black-hole adoption. The eventual wormhole-to-trumpet requirement retains the Minkowski hyperboloidal reference. All current failed source gates and archives remain unchanged.
