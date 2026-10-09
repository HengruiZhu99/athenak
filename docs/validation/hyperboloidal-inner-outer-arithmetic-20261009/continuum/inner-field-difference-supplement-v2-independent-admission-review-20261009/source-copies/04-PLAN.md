# Held additive arithmetic supplement for source003

No imports of this probe/oracle, compiler, numerical query or gate execution
are admitted by this source preparation. The actual unchanged15740-record
source003 Release gate must first PASS; root must then separately release
this exact supplement. Both runner and Fraction oracle enforce that actual
outer child receipt, its source index and successful saved oracle report.
Old source001 compile FAIL and source002 oracle FAIL remain unchanged.

The scientific inputs are byte-copied source003 inner_gauge.hpp plus its
existing dual/reference support. Product/ProductValue bodies are verified
byte-identical to source002. All reference fields have zero dual derivative;
live scalar/gradient variables have the following independent relative seeds:

    (xi_a,xi_chi,xi_grad)=(0,0,0),(1,0,0),(0,1,0),
                         (1,1,0),(1,-2,0),(0,0,1).

Use18 witness rows, three witnesses times six seeds. All inputs and targets
are exact binary powers. With h=chi_hat=1:

1. a=2^-300,chi=2^601. Named FieldSquareDifference gives1, with
   tangent4xi_a+2xi_chi. The gradient seed is unused.
2. a=2^300,chi=2^-600,chi_i=0,chi_hat_i=1. Gradient dual is
   chi*xi_grad despite zero primal. Product(a,a,FieldLogGradientDifference)
   gives-1, tangent-2xi_a-xi_chi+xi_grad.
3. a=2^-300,chi=2^600,alpha_i=0,alpha_hat_i=1. Gradient dual is
   a*xi_grad despite zero primal. Product(-1,a,chi,FieldLogGradientDifference)
   gives+1, tangent2xi_a+xi_chi-xi_grad.

The three additional zero-seed negative controls evaluate the exact original
near-only source002 expressions through unchanged Product. Each is expected
to lose the normal target and return primal0. Exact original expression
lines and Product bodies are byte-bound by preparation evidence. These are
new scalar-expression controls, not a rerun or relabeling of the old actual
nonflat source002 failure.

The108 Near boundary rows use reference h=2^-400,1,2^400; primal ratios
.5-2^-40,.5,.5+2^-40,2-2^-40,2,2+2^-40; and all six seeds. Inputs a=h*ratio,
chi=1,gradient=a/8,reference_gradient=h/4 have independent tangents
(a*xi_a,xi_chi,a*xi_grad). Check NearFieldValue against the exact Fraction
inequality1/2<=a/h<=2, and both generic log-gradient audit kinds using the
same scalar x=a. Also check FieldSquareDifference and all returned duals:

    dA=a^2-h^2, dA_tangent=a^2(2xi_a+xi_chi),
    dl=-a/8, dl_tangent=a(xi_grad-xi_a/4).

This covers branch endpoints and each side without a claim of universal
floating C1 continuity. Every input/output remains finite and representable
in this fixed family; artificial scalar gradients are arithmetic contexts,
not complete nonflat Minkowski states. Each called dA/dc/dal branch counter
must agree with the exact primal criterion. Negative controls use no new
branch helpers. Expected total129 records, unchanged main oracle/cases.

The independent oracle uses only stdlib Fraction on exact hexadecimal
binary64 exports and fixed closed targets, never source helper results as
expected targets. Nonzero normal primal/tangent targets use relative2e-10;
exact zero targets use absolute2e-10. Required negative-control primal0 and
branch/count/label/source identities are exact. No arbitrary absolute-seed,
metric-contrast,dV, nonlinear source, gauge principal, scri, BH, grid or
evolution acceptance follows. Unique Release/ASan-UBSan binaries, complete
dependencies, pre/post inputs and stdout/stderr/returncodes will be retained
if root later releases this gate. No old destination is reused.
