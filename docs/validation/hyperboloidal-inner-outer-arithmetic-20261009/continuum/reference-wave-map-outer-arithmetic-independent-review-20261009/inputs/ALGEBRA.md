# Complete far rows and comparison to the legacy graph

Let a=live alpha,h=reference alpha,x=live chi,y=reference chi,
G=gtildeInv,Gh=reference gtildeInv,betaHat=reference beta,
db=beta-betaHat,A0=a²x,Ah=h²y. The reference position and its complete consumed
coefficient jets are fixed during field-dual differentiation. Let C^a_ij be
the unchanged scaled physical reference connection and O_i=Omega_i.

The far graph forms each tensor entry as complete scaled products

    dV_ij=Product(a,a,x,Gij)-Product(h,h,y,Ghij),
    Lhat_ij=Product(h,h,y,Ghij)-betaHat_i betaHat_j,
    dL_ij=dV_ij-db_i beta_j-betaHat_i db_j.

No underflowed scalar A0 is multiplied later by a large G entry. Product
retains every registered field-dual product-rule term separately, including
a zero primal/nonzero derivative factor. This does not mean a genuinely
ill-conditioned sum of large physical terms is universally accurate.

The complete alpha rows are

    Ralpha=beta.grad a-(a/h)betaHat.grad h,
    Salpha=-a² P+a h Phat-a db.O-a dL:C0.

Expanding -a[a(P-Phat)+(a-h)Phat+db.O] gives the displayed pole exactly.
Ralpha follows by expanding the legacy deviations and its scaled reference
advection. There is no division by the live lapse. All factors of a reach a
scaled Product before multiplication; the reference h inverse remains fixed.
P is the original stored physical-trace variable, not replaced with P+2Theta
or an Einstein-constraint relation.

For beta component i, the far regular row is

    Rbeta_i=A0 Lambda_i-Ah LambdaHat_i
      +sum_j[beta_j (beta_ji-betaHat_ji)+db_j betaHat_ji]
      +sum_j[.5 a² Gij x_j-.5 h² Ghij y_j
              -a x Gij a_j+h y Ghij h_j].

The Lambda identity is the complete live/reference contraction. The advection
line is the unchanged stable deviation graph. The two derivative groups equal
the legacy chi- and lapse-gradient graphs respectively, including every
reference derivative. Treating them as complete groups removes artificial
O(a² y_j) or O(h x h_j) reference terms that cancel in real arithmetic but
destroy the fixed high-contrast fields in the old graph.

The pole beta row retains the same full connection and lapse/shift coupling:

    Sbeta_i=2 sum_j dVij O_j
      -sum_jl[dLjl C^(i+1)_jl+dLjl beta_i C^0_jl
               +Lhat_jl db_i C^0_jl].

This is exactly the legacy expression after distributing its connection
bracket. The original Assemble still returns R+S/Omega once. No reference RHS
subtraction, pole suppression or new geometric equation is introduced.

At exact reference all rows vanish through the unchanged near branch and all
old reference/algebraic/principal identities are preserved there bitwise.
At far fields the real expressions are the same equations but their floating
graphs deliberately differ. The old literal bitwise-outer contract and its
actual failed outputs cannot be claimed to pass; a new source identity and
new explicit comparison contract are mandatory. No proof of arbitrary-state
accuracy, scri class preservation, global stability or native acceptance follows
from these identities or from the source-only prototype.
