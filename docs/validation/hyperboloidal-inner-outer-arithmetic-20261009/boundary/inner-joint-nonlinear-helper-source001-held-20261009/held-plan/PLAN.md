# Held nonlinear inner gauge helper gate

This is source-only preparation. No helper implementation, numerical import,
compilation, query, matrix, spectrum, propagation, native evolution or black-hole
initial data is admitted by this plan. All old sources and results remain intact.
The completed 118-case actual20 plus 18-case exact-scalar principal gate is a
prerequisite, not a nonflat nonlinear gauge test.

The target is precisely the candidate in the pinned `gauge_proposal.hpp`, with
G0 in {3/8,3/4}, no additional shift restoring term (eta=0), and the complete
frozen reference-wave-map (RWM) source. P remains the stored independent
Kphysical-2Theta variable. This gate does not replace P by K, change geometry or
damping, or prescribe a puncture reference. The gauge reference is the analytic
nonflat Minkowski layer throughout.

## Binding and exact target

A future fresh helper will expose `inner::Gauge(p,u,xyz,Parameters{G0})` and a
separate bounded coefficient function for k. It will take the actual explicit
Cartesian xyz, call the unchanged `rwm::ReferenceConnection(p,xyz)`, and retain
all its spatial and temporal-index coefficient slots. W is evaluated directly
as `SmoothCutoff(p.radius,.45,.85).value`. Calling `LayerCoefficients` merely to
obtain W is forbidden: it also forms 1/alpha, which is unnecessary here and can
overflow for a positive tiny lapse. The planned helper is a private CPU generic
scalar/dual implementation, without a GPU portability claim.

The first candidate branch after ordinary input validation is `if (W==1)
return rwm::Gauge(p,u,connection)`. It occurs before A0, B, k, inner gradients or
new coefficient products are evaluated. The complete returned regular/pole
parts, validity and assembled rows must then match the frozen RWM bit for bit.
The inherited RWM arithmetic is not asserted robust for arbitrary high-contrast
outer states; any declared outer failure remains a failure of this gate.

For W<1, write c=1-W, A0=alpha^2 chi, B=c G0+W A0, k=B/(A0+B). The mathematical
correction to RWM is exactly

    Delta pole.alpha = -2 c alpha (P-Phat)
    Delta regular.beta^i = (B-A0)(Lambda^i-Lambdahat^i)
        + alpha^2 (2 k^2-1/2) gInv^{ij}
          [partial_j chi-chi partial_j chihat/chihat].

No mu=B/A0 is formed by the runtime helper. The coefficient oracle may use
arbitrary-precision A0 and B to define the mathematical target. `ALGEBRA.md`
specifies an equivalent grouped nonlinear implementation, including every
reference connection and reference derivative. It avoids the destructive
addition of A0 deltaLambda to (B-A0) deltaLambda, and the corresponding
chi-gradient cancellation. It introduces no floor, clipping, assumed falloff,
reference counterterm or new source subtraction.

## Bounded arithmetic contract

The coefficient function represents positive A0 and X=c G0 using frexp/scalbn
mantissas and integer exponents. It normalizes both by their maximum exponent
and evaluates k=(v+W u)/(v+(1+W)u). The dominant normalized mantissa stays
positive. It never forms alpha^2 chi, B/A0, a huge power-of-two constant, or a
zero-over-zero ratio just to compute k. Exact W=1 is handled separately.
Generic dual scaling must scale both primal and tangent by the same power of
two; a value-only double fallback is a failing negative control. Relative dual
seeds are used in extreme cases. A finite primal does not imply that every
arbitrary derivative or RHS is representable. Genuinely nonrepresentable
declared results must be reported and rejected, not floored or silently omitted.

For the full-source cases, preserve the displayed grouped operations and use
scaled products where an intermediate overflows or underflows while the final
declared target is representable. Record every scaled/fallback path and its
case. There is no blanket claim covering all finite positive input jets.

## Fixed cases and independent oracles

`cases.json` fixes the complete grid and deterministic state families. The main
grid has 4 curvature radii x 14 radii x 3 directions x 2 G0 = 336 labeled
reference/G0 cases. Direction duplicates at the origin remain labeled
duplicates. Fourteen fixed state families give 4704 full-source cases. These
include finite high-contrast positive lapse/chi, independent P and Theta,
off-reference Lambda, gradient and SPD tensor variations. The five radii at or
above .85 provide 1680 exact outer short-circuit comparisons. The three exact
core radii provide 1008 main core comparisons.

Independent connection checks compare every slot of Omega times the physical
reference Christoffel symbol against both literal ADM4 construction and the
independently differentiated reference embedding, using the pinned frozen
audit definitions. In the flat core the independent connection is exactly zero.
Nonflat transition and collar cases must exercise the actual nonzero reference
connection; no zero-connection stand-in is allowed.

At an exact reference input every returned regular/pole entry must be exactly
zero, not merely small. A separate literal unfactored RWM-plus-correction oracle
uses exact binary64 inputs lifted to multiprecision, retaining Phat, chihat,
reference gradients and all connection slots. It checks all eight parts and
four assembled gauge rows. It does not derive its target by calling the new
grouped helper. Ordinary families use 80/110 decimal digits; high-contrast
families use 240/280 digits. Precision agreement is a prerequisite to a native
comparison. Exact inner additions have no independent preferred-Box claim.

At W=0, Omega=1 the independent core target is

    alpha_t = beta.grad(alpha)-alpha(alpha+2)P
    beta_t^i = beta.grad(beta^i)+G0 Lambda^i
        -alpha chi gInv^{ij} partial_j alpha
        +2 alpha^2 k^2 gInv^{ij} partial_j chi.

This retains off-constraint P. Separate connection-only and chi-gradient-only
core cancellation witnesses require entrywise relative accuracy for each
nonzero normal target; a zero result cannot pass by normalization with one.
The coefficient-only extreme grid is separate from the full-source grid: it
contains minsubnormal through maxfinite inputs and tests k without requiring
an unrepresentable full RHS to be finite. Isolated huge-A core witnesses are
also separate, with exactly vanishing unused gradients, connections and P.

The generic-dual gate uses the frozen full20 algebraic tangent lift and compares
the actual changed gauge rows against independent directional differences and
the literal target. All other geometry rows are outside the helper and remain
unchanged by source identity. The principal attribution compares the actual
Lambda and chi-gradient coefficients and lapse P coefficient against the
passed proposal, not only a sampled eigenvalue. The finite-difference gate
retains its full fixed epsilon sequence and accepts only its final level plus
the stated convergence/floor rule. It never takes the best level over epsilon.

## Final predeclared thresholds

All absolute errors and entrywise scaled errors abs(error)/max(1,abs(target),
abs(result)) are retained, including near-zero fields. No threshold is changed
in response to output. The exact case lists, counts and thresholds are in the
recipe. They include:

- coefficient k absolute error <=2e-14, finite 0<=k<=1; exact unit/zero and
  correctly rounded tiny results where explicitly declared;
- full connection scaled error <=2e-11, complete source parts/assembled rows
  <=2e-10; reference zero and W=1 branch are exact;
- independent precision agreement <=1e-65 ordinary and <=1e-180 contrast;
- core nonzero normal cancellation-witness relative error <=2e-10;
- dual oracle entrywise scaled error <=2e-10; final ordinary FD <=5e-7,
  with error decreasing by >=2 from first to last level or every level<=5e-9;
- proposal principal coefficient scaled error <=2e-11;
- invalid inputs are rejected, and nonrepresentable expected outputs are
  explicitly classified without a finite-output acceptance claim.

The future implementation source, compile recipes, generated dependencies and
all fixed query payloads require a fresh source checkpoint and root release.
Release and ASan/UBSan builds must retain distinct as-built executables, complete
commands, compiler flags, dependency hashes, stdout/stderr, return codes and
before/after input guards. Full sequences and every failed attempt are retained.
No broad native adoption, black-hole run, spectrum, global stability, continuum
well-posedness or preserved scri class follows from this local nonlinear gate.
