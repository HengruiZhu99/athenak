# Exact identities and their conditioning regimes

Write a=alpha,h=alpha_hat,x=chi,y=chi_hat,
A0=a^2x,Ah=h^2y. The current factorization is exactly

    A0-Ah=(a+h)(a-h)x+h^2(x-y).                              (1)

When a/h is tiny and x/y huge, the two displayed terms can be of order
h^2x with opposite sign while the desired answer is approximately-Ah.
For a=1e-150,x=1e60 and a finite O(1) reference h,y, the real A0 is of
order1e-240, while (1)'s terms are of order1e60. Binary64 can discard the
O(1) reference contribution before their sum. This diagnoses a real
conditioning defect of that expression; it does not quantify an actual
saved row beyond the parent's association summary.

If1/2<=a/h<=2 and1/2<=x/y<=2, each factored term is bounded by a fixed
multiple of Ah and its deviations vanish exactly at reference. If either
individual closeness test fails, separate scaled A0 and Ah products remove
the artificially much larger h^2x intermediates. A direct A0-Ah subtraction
can still be genuinely cancellation-sensitive if its true value is tiny;
the original entrywise-scaled oracle gate must measure the finite outcome.
Anti-correlation is why A0/Ah alone cannot select the near branch.

For each Cartesian derivative direction,

    dc=(x_i-y_i)-(x-y)(y_i/y)=x_i-x(y_i/y),
    dal=(a_i-h_i)-(a-h)(h_i/h)=a_i-a(h_i/h).                  (2)

The left forms preserve exact-reference deviations. Far from reference the
right forms avoid cancelling y_i or h_i against products of a rounded
deviation approximately equal to the negative reference field. Scaled
products must use the live fields and their complete field-dual product
rules. A zero primal must not erase its nonzero dual derivative. Reference
log-gradients are fixed coefficients in this local field-dual contract.

## Why dV needs a coherent separate repair

Let gi=gInv,ghi=gHatInv,dgi=gi-ghi (the actual inverse-difference identity
may remain factored). Then

    dV=(A0-Ah)gi+Ah*dgi=A0*gi-Ah*ghi.                        (3)

For A0 tiny and gi huge, the first form can cancel two huge terms and lose
a finite live/reference tensor difference. The future direct form should
evaluate each component from Product(a,a,x,gi_ij) minus
Product(h,h,y,ghi_ij), not from a previously underflowed A0 times gi.
This also avoids erasing a representable composite product after an
unrepresentable intermediate A0. The current primary helper still requires
its other explicitly formed coefficients to satisfy its finite contract;
this observation does not silently broaden that contract.

At the exact core reference h=y=1,ghi=I, choose
a=2^-200,x=2^-100,gi=diag(2^501,2^-501,1). Real arithmetic gives
A0=2^-500 and dV_11=2-1=1. In the first form, A0-1 can round to-1 and
gi_11-1 can round to gi_11, producing cancellation of opposite2^501 terms.
This is an exact analytic lost-unit mechanism, not a numerical execution.

The current regular-beta reference-gradient terms must be regrouped with
(3) if that tensor is hardened. Exactly,

    (1/2)(dA/y)gi*y_j+(1/2)h^2*dgi*y_j
       =(1/(2y))*dV*y_j,
    -(dA/h)gi*h_j-h*y*dgi*h_j=-(1/h)*dV*h_j.                 (4)

Consequently the complete displayed gradient group becomes

    2a^2 k^2 gi*dc+(1/(2y))*dV*grad(y)
      -a*x*gi*dal-(1/h)*dV*grad(h).                          (5)

Together with unchanged dL=dV-dbeta*beta-beta_hat*dbeta, this binds all
reference-connection and Omega-gradient appearances to the same tensor
difference. The identities preserve the full nonflat reference connections,
P/off-constraint dependence and the outer exact W1 branch. They do not by
themselves prove finite-state accuracy, supply a general compensated sum,
or justify a universal dual/metric-contrast contract. That broader change
and additive witnesses remain a separate held scope, not the primary
three-difference correction.
