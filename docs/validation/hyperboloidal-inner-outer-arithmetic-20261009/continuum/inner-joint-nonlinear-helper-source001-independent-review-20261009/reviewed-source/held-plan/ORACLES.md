# Independent target definitions for the held gate

All inputs to these targets are the exact real values of the recorded binary64
inputs. No target calls the new grouped helper. Multiprecision is an oracle,
not a runtime requirement. Let C^a_ij be the returned full reference scaled
connection, V^{ij}=alpha^2 chi gInv^{ij}, L^{ij}=V^{ij}-beta^i beta^j,
Bomega=beta^i Omega_i. Hatted expressions use the recorded reference jets.
Define the literal unfactored expressions

    Ralpha = beta^i partial_i alpha
    Salpha = -alpha^2 P-alpha Bomega-alpha L^{ij} C^0_ij
    Rbeta^i = V^{ij}(gtilde_jk Lambda^k)
        +beta^j partial_j beta^i
        +(alpha^2/2)gInv^{ij}partial_j chi
        -alpha chi gInv^{ij}partial_j alpha
    Sbeta^i = 2V^{ij}Omega_j-L^{jk}(C^{i+1}_jk+beta^i C^0_jk).

The first Rbeta term is equivalently alpha^2 chi Lambda^i; the target will use
that direct component expression, avoiding an unnecessary inverse contraction.
The split equivalent to the frozen factored RWM helper is

    regular.alpha = Ralpha-(alpha/alphahat) Ralpha_hat
    pole.alpha = Salpha-(alpha/alphahat) Salpha_hat
    regular.beta = Rbeta-Rbeta_hat
    pole.beta = Sbeta-Sbeta_hat.

Add precisely the proposal's Delta pole.alpha and Delta regular.beta stated in
PLAN.md. This unfactored split is an independent algebraic readback of the
deviation implementation. Its reference evaluations cancel identically. It is
not a new runtime reference RHS subtraction. The separate literal ADM source
R+S/Omega and its reference stationary-identity residual must also be reported;
the analytic reference identity has binary64 coefficient roundoff, which must
not be silently assigned to the candidate. Gate the new helper against the
specified split and report the literal-source residual separately.

For the independent core oracle, reference values are alpha=chi=Omega=1,
beta=P=Lambda=0, gtilde=I and all connections/gradients zero. The simplified
equations in PLAN.md follow without cancellation. They independently validate
the collapsed-lapse/off-constraint-P and high-contrast cancellation witnesses.

For relative coefficient-dual seeds delta alpha=salpha alpha,
delta chi=schi chi, fixed W and G0, the exact derivative is

    delta k = -X A0/[X+(1+W)A0]^2 (2 salpha+schi), X=(1-W)G0.

It can be evaluated independently with high-precision products, or with the
equivalent normalized u,v ratio. No runtime mu or division by a tiny live lapse
is required for this derivative. W=1 gives k=1/2 and delta k=0. At tiny results
the oracle explicitly records the correctly rounded binary64 target, sign and
whether the exact result lies below minsubnormal. There is no zero floor.

The full generic-dual target differentiates the literal split and correction
with exact analytic dual arithmetic in multiprecision, including the inverse
metric and all frozen algebraic tangent reconstruction. It is compared to the
new generic helper and to the fixed ordinary centered-FD sequence separately.
It does not assume value-only source columns classify the coupled system.

In particular, no contribution is discarded merely because a primal factor is
zero. Any scaled product or zero shortcut must preserve the full generic dual
factor, so a nonzero tangent of a zero primal cannot disappear. Exact all-zero
reference-connection factors in the core may safely make a full term zero.
