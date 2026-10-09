# Independent direct RWM equations and exact arithmetic witnesses

For fixed reference connection C=Omega*Gamma_ref and O_i=Omega_i, write
G=gtildeInv, a=alpha,x=chi,beta=b. Lambda_i below denotes the stored
Cartesian contravariant Lambda component with label i, not a lowered covector.
The literal physical-P rows are

    R0=b.grad a,
    S0=-a²P-a b.O-a sum_ij[(a²xGij-bi bj) C0ij],
    Ri=a²x Lambda_i+sum_j[bj bji+.5a²Gij xj-a x Gij aj],
    Si=2sum_j[a²xGij Oj]
       -sum_jk[(a²xGjk-bj bk)(C^(i+1)_jk+bi C0jk)].

Compute these independently for live and fixed reference fields. The returned
parts are R0-(a/h)Rhat0, S0-(a/h)Shat0, Ri-Rhati, Si-Shati, ordered as
regular alpha/beta xyz then pole alpha/beta xyz. Assemble each R+S/Omega.
There is no added inner-k correction, preferred-source substitution, alternate
P+2Theta storage or live reference subtraction. The target's MP inverse and
dual product/quotient rules must be independent of the C++ helper graph.

All following contexts use g=gHat=I, beta=betaHat=0,Lambda=LambdaHat=0,
P=Phat=0,Omega=1 and supplied zero reference connection, with every unstated
gradient/field tangent0. Reference h=y=1. They deliberately supply synthetic
reference gradients/Omega derivatives, so they test scalar arithmetic only;
no claim that the reference is a stationary Einstein/Minkowski layer is made.
The new route is far in all three cases. The independent oracle uses Fraction
of exact binary64 exports and the closed targets below, not GaugeFar outputs
as expectations. Reference values and gradients always have zero field dual.
Set both p.dalpha and p.state.alpha.d to the stated reference alpha gradient,
p.beta=p.state.beta.value=0 and p.k_physical=p.state.trace.value=0. Supply
Connection.valid=true with all scaled entries exactly0 directly; do not call
ReferenceConnection on these deliberately synthetic contexts. Metric spatial
jets are0, so Geometry.valid has a finite exact identity-metric prerequisite.

1. Tensor/pole flux: a=2^-300,x=2^601,all field gradients0,O_x=1/2.
   Then A0=2,dVxx=1 and Sbeta_x=1. With adot=xi_a*a,xdot=xi_chi*x,
   the exact tangent is4xi_a+2xi_chi; xi_grad is unused. All other parts/RHS
   are zero except assembled beta_x=1 with the same tangent. Legacy dV uses
   two cancelling2^601 terms and returns0 in its zero-seed graph.

2. Complete chi flux: a=2^300,x=2^-600,x_x=-x,y_x=1,all alpha gradients0,
   O=0. Let (x_x)dot=-x*xi_chi+x*xi_grad. The exact regular beta_x is
   .5*a²*x_x-.5=-1, with tangent -xi_a-.5xi_chi+.5xi_grad.
   Assembled beta_x is the same; all other outputs are zero. Legacy's exact
   zero-seed graph is .5[a²(x_x-1)+(a²-1)]; its two opposing rounded2^600
   terms yield0 instead of-1.

3. Complete lapse flux: a=2^-300,x=2^600,a_x=a,h_x=2,all chi gradients0,
   O=0. Let (a_x)dot=a*(xi_a+xi_grad). The exact regular beta_x is
   -a*x*a_x+2=1, with tangent -2xi_a-xi_chi-xi_grad.
   Assembled beta_x is the same; all other outputs are zero. The legacy
   bracket (a-1)x+(x-1) rounds to0, whereas a*x*(a_x-2) rounds to-2^301;
   after its outer minus sign the legacy result is2^301, not1.

The stated legacy zero-seed arithmetic is a source-pencil prediction, not an
executed observation. Its future negative-control exports must match it or
the supplement stops and preserves the discrepancy. No universal correctly
rounded-sum or AD branch-continuity claim follows. The input fields/duals and
all expected normal outputs are finite in these fixed power contexts.
