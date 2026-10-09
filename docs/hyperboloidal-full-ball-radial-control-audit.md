# Full-ball radial representation and scalar energy control

The common-rho polynomial representation passes a scalar-wave discrete energy
gate while retaining regular-origin modes. It also exposes a mass-matrix
error in the simplest diagonal-weight construction. These results admit a
representation and scalar mathematical control, not a Z4c radial solver.
The actual [total-J angular kernel gate](hyperboloidal-total-J-continuum-control-audit.md)
supplies the separate local tensor action.

## Regular representation and mass

For each regular Cartesian solid harmonic, use solid_L(x)*W_L(rho),rho=r^2,
with independent degree-(N-1) envelopes. Common N-point Gauss-Jacobi nodes
for weight rho^(1/2) preserve the independent origin freedoms. Constant
Cartesian vector and STF channels must remain where allowed by total J.
No extra cross-L origin conditions or artificial inner boundary are imposed.
This smooth-origin space is not an eventual puncture/trumpet representation.

The cancellation-safe Laplacian envelope is

```
Delta(solid_L W)=solid_L*[4rho W_rhorho+(4L+6)W_rho].
M_L,ij=(1/2) integral_0^(rb^2) rho^(L+1/2) li lj d rho.
```

For degree-(N-1) envelopes, common N-point weights are exact for mass products
only at L=0,1. An overintegrated base rule needs at least N+floor(L/2) points
for polynomial mass products. Nonpolynomial reference products require
independent quadrature convergence. Dense exact mass matrices permit common
collocation nodes without assuming a universal diagonal mass.

## Scalar whole-ball energy identity

With polynomial trace row t at rb, define

```
D_L=4rho D_rho^2+(4L+6)D_rho,
S_ij=(1/2) integral rho^(L-1/2)*
     [(L li+2rho li')*(L lj+2rho lj')+L(L+1)li lj] d rho.
M_L D_L=-S+rb^(2L+1)*t^T*[L t+2rb^2 t D_rho].
```

S retains both radial and angular physical gradient terms. The origin flux
vanishes for all regular envelopes. At L=0 its extra derivative factor gives
O(r^3); the source uses an integrable rho^(-1/2) quadrature multiplied by
Q=2rho W_rho, equivalent to a rho^(3/2) derivative-energy integral.

The scalar control uses W_t=P and

```
P_t=D_L W-M_L^-1*rb^(2L+2)*t^T*
    [P_b+(L/rb)W_b+2rb W_rho,b],
E=(P^T M_L P+W^T S W)/2,
E_t=-rb^(2L+2)*P_b^2.
```

This is a checked finite-dimensional scalar-wave identity. The full symmetric
energy matrix and eight fixed-seed states per case are tested. At L=0 the
constant W mode has exactly zero energy and zero generator action, so E is
a seminorm on all configurations. That mode is retained.

All50 predeclared cases pass: N4/8/12/16/24,L0 through4,rb=.7/1. The last
radius is a scalar model endpoint, not an exact-scri Z4c evolution. Independent
Jacobi derivative and barycentric actions agree below5.83e-13 scaled. Worst
scaled nodal/modal integration-by-parts residuals are2.89e-13/2.45e-13;
the wave-energy matrix residual is3.82e-13. Largest absolute modal IBP and
wave-matrix Frobenius residuals are1.14e-6/1.61e-6 alongside large derivative
entries. They are retained rather than described as small absolute errors.

Raw mass conditioning reaches2.87e9; diagonal scaling reduces it to2.77e4.
An orthonormal Jacobi modal congruence of the same polynomial space gives
mass condition1+1.1e-12. The nodal-to-modal evaluation condition reaches5.36e4.
Naive diagonal common weights differ from the exact mass by2.86% to17.16%
for L>=2. No tolerance changed to pass these checks.

Historical NumPy-boolean JSON serialization and array-count verifier failures,
their exact sources and saved arrays are preserved. Serialization repairs
leave the original and accepted mathematical matrix bytes identical. Root
and independent source reviews check factors, congruence and energy signs.

## Finite-radius harmonic boundary advisory

At a deliberately finite rb<S in the exact harmonic collar, the actual
normalized twenty-field principal reduction uses ten normal configuration
derivatives and ten momenta (P/Omega,Theta/Omega,A5,Lambda3). These are not
the twenty stored values. The exact harmonic matrix obeys A^2=I and

```
H=I+A^T A>0, H A=A+A^T,
Pplus/minus=(I+/-A)/2, rank(Pplus/minus)=10.
```

It agrees with288 retained actual extraction records within8.88e-16. In
RHS convention y_t=(beta_n I+alpha A)y_r, the incoming coefficient is
k_in=(S-rb)^2/(2aS)>0; the outgoing coefficient is
k_out=-(S+rb)^2/(2aS)<0. Small inward speed does not eliminate incoming data.
P/Omega and Theta/Omega norms are not uniformly equivalent to unweighted
norms as rb approaches S.

A momentum-only frozen boundary penalty has the correct energy sign if its
trace lift supplies the intended work. Its Schur condition is
k_in*||R||^2<=|k_out|,with ||R||^2=2.71142155315. The ratios at rb=.98/.995
are2.77e-4/1.70e-5 for S1,a.5. However H has a rank-six configuration/momentum
cross block. A scalar inverse quadrature weight does not automatically
realize the required coupled adjoint lift.

A separate variational candidate uses the complete state X=(U,V),q=D U,
an SPD finite-dimensional energy matrix E_X including positive U mass,
and trace B X=y_b. Its incoming penalty

```
X_t|SAT=-E_X^-1 B^T H_b k_in Pplus B X
```

has exactly the intended negative incoming work. This algebra does not match
the actual bulk operator automatically. Boundary derivative/momentum traces
are unbounded in the corresponding H1(U) times L2(V) continuum energy;
finite-N SPD alone is not a uniform closed-generator or boundary theorem.
The complete derivative normalization matters: delta(D_s alpha)/alpha_ref
and D_s(deltaalpha/alpha_ref) have different lower-order terms.

## Remaining gates

No Z4c radial matrix, incoming physical/constraint data, SAT adoption,
propagation, eigensolve or finite-pulse result is accepted. Actual regular
core action, variable-coefficient bulk energy/control, full boundary reduction,
constraint coupling, quadrature/degree sensitivity and rb-to-S limits remain.
Interior Gauss nodes alone impose no justified outflow condition. A later
radial solver changes bulk derivatives as well as the boundary and cannot
uniquely attribute earlier Cartesian growth to ghosts.

The subsequent [core and principal-sector controls](hyperboloidal-core-principal-controls-audit.md)
supply an exact division-free total-J core action and a separate harmonic
constraint/gauge/TT classification. Complete finite-radius boundary data and
the variable-coefficient radial operator remain separate validation steps.

Production remains implementation27c19d20696ea6dd4704032c51dfd026218f64f2.
The later black-hole acceptance target includes the inner wormhole-to-trumpet
transition with a Minkowski hyperboloidal reference retained throughout.

The [byte-preserved evidence archive](validation/hyperboloidal-full-ball-radial-control-experiments-20261009/README.md)
contains61 cataloged blobs,870,883 bytes and23 finite JSON files. Catalog
SHA256 is `f6c999d314af215868c2c7c1cf203bc6771095ec33a4e212d07ef518ac34e15f`.
Original frozen advisory/scalar/additive-review index SHA256 values are
`bff27b1f403868a8e081c0cf0b574a7bbcd75df7261bdb62053563b4df9a29fb`,
`2a61499d58ceae69d0259208b790e2688156e9a6e0496f44dcd52bf27665197e`,
and `62d8cea2a833b86bebe0a2a1929c16b921f3f81513c84afcb6692535d335560d`.
The read-only verifier passes without scratch dependencies or scientific
reruns. NumPy binary arrays and payloads larger than1MiB remain metadata only.
