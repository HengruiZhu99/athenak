# Source-only grouping identities

Use a=alpha, h=alphahat, chi=chi, ch=chihat, c=1-W, gi=gtildeInv,
ghi=ghatInv, da=a-h, da2=(a+h)da, dchi=chi-ch. Every h/ch quantity and
derivative is the complete analytic Minkowski reference, not a constant core
replacement. All products below are mathematical identities; their grouped
binary64 evaluation is the subject of the proposed gate.

Define

    dgi = -gi (g-ghat) ghi
    Ahat = h^2 ch
    DeltaA = da2 chi + h^2 dchi = a^2 chi-Ahat
    Dchi_j = (chi_j-ch_j)-dchi ch_j/ch
    Dalpha_j = (alpha_j-h_j)-(da/h)h_j
    dV = DeltaA gi + Ahat dgi
    db = beta-betahat
    Lhat = Ahat ghi-betahat betahat^T
    dL = dV-db beta^T-betahat db^T.

The complete new regular beta row is the sum of:

    B (Lambda-Lambdahat) + DeltaA Lambdahat
    beta^j (partial_j beta^i-partial_j betahat^i)
        +db^j partial_j betahat^i
    2 a^2 k^2 gi^{ij} Dchi_j
        +1/2 [(DeltaA/ch)gi^{ij}+h^2 dgi^{ij}] ch_j
    -a chi gi^{ij} Dalpha_j
        -[(DeltaA/h)gi^{ij}+h ch dgi^{ij}] h_j.

The new pole-alpha row is

    -a[(a+2c)(P-Phat)+da Phat+db^j Omega_j]
        -a dL^{ij} connection.scaled[0][i][j].

Regular alpha is exactly the frozen RWM regular-alpha expression. Pole beta is
exactly the frozen RWM pole-beta expression, using the dV/dL identities above:

    2 dV^{ij} Omega_j
      -dL^{jk}(connection.scaled[i+1][j][k]
               +beta^i connection.scaled[0][j][k])
      -Lhat^{jk} db^i connection.scaled[0][j][k].

The Lambda identity follows from A0 deltaLambda+c(G0-A0)deltaLambda
=[cG0+W A0]deltaLambda. For the chi-gradient identity, expanding Dchi leaves
the residual reference-gradient coefficient

    a^2(chi/ch-1)+da2 = DeltaA/ch.

For the alpha-gradient identity, the residual reference-gradient coefficient is

    a chi da/h+da chi+h(chi-ch)=DeltaA/h.

These identities show that the proposed grouping retains every frozen RWM
reference derivative and connection term. When u equals the reference, da,
dchi, db, dgi, DeltaA, Dchi, Dalpha and deltaLambda all vanish exactly.
The added P term vanishes independently of the reference Einstein equation.
For W=1 the helper does not evaluate these regrouped formulas: it returns the
unchanged frozen RWM implementation, preserving the exact outer arithmetic.

For coefficient-only arithmetic let X=cG0. Write A0=mA 2^eA and X=mX 2^eX
using positive finite normalized mantissas. With e=max(eA,eX), u=mA2^(eA-e)
and v=mX2^(eX-e),

    k = (v+W u)/(v+(1+W)u).

The larger normalized term is nonzero and positive, so this avoids both a raw
A0 overflow and a B/A0 singular ratio. A ratio that rounds below minsubnormal
may correctly become zero; that is not a floor. The implementation must not
replace a derivative by zero merely because it branches on a primal exponent.
Relative-seeded generic dual tests bind this requirement.
