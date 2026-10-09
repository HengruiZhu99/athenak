# Differentiated fixed-Lorentz-ray Kirchhoff oracle

Pencil/source review only, 2026-10-09. No numerical or CAS imports, evaluations,
kernel queries, derivative implementation, or propagation are performed here.
This is a proposed later gate, separate from the already released values-only
prototype `flat_ivp_values.py` SHA256
`89c96ee729b8cc741757740534566c9557dd6fdce2e66213c2fe8ff1c7e8ffe7`.
That prototype and its running attempt are unchanged. This note derives the
first and second **inertial** derivatives of the same full angular integral.
It does not reconstruct native time or diagnose a native abort.

## Conventions and the unchanged integral

Use signature (-,+,+,+), inertial event p=(T,X), initial graph T=H(|y|), and
zero initial scalar field u with future-normal datum n(u)=s(y). Each of the
four fixed target components u^A is a separate scalar. Let

```
v_i = H_i,       c = sqrt(1-H_i H_i) > 0,
n = (1,v_i)/c,   F(y) = s(y)*c(y).
```

Spatial source indices i,j,l run 1..3; event indices a,b run 0..3. All source
derivatives in this note are ordinary derivatives in the graph chart y:
they already include the dependence of the initial data on T=H(y). They
are not derivatives of an arbitrary off-graph extension at fixed inertial T.

Choose a constant proper orthochronous Lorentz matrix B and set
k=B(1,omega), omega on the unit sphere. Thus k is future null, k0>0, and
all four components of k are held constant in every event derivative. The
source point and root are

```
y^i = X^i-lambda*k^i,
g(p,lambda) = T-lambda*k0-H(y) = 0,
K = k0-H_i*k^i > 0.
```

Here lambda is the **boosted affine parameter**, called ell in the pinned
values pencil; it is not the lab radial distance lambda_lab=lambda*k0.
The positive Green sign and angular measure were independently derived in
the pinned KIRCHHOFF-PENCIL:

```
u(p) = (1/(4*pi)) integral_S2 I(p,omega) d omega,
I = lambda*F(y)/K.
```

The source graph is strictly spacelike and H>=0. At a finite future point
g(p,0)>0, g_lambda=-K<0, and lambda<=T/k0 brackets the unique root. The
sphere has fixed limits. At each such point the source intersections are
finite and K has a positive minimum on the compact sphere. Differentiation
under this integral is valid locally for sufficiently smooth H,s and a
future neighborhood inside the graph's domain of dependence. This statement
is not uniform as the event approaches scri. Initial-surface values and jets
may instead be imposed from the exact normal data or taken as future limits.

## Exact implicit root jets

Define the constant event-to-spatial inclusion e_a^i=delta_a^i for a=1..3
and e_0^i=0, and t_a=delta_a0. Then

```
lambda_a = (t_a-H_i*e_a^i)/K,
y_a^i = e_a^i-k^i*lambda_a,
lambda_ab = -H_ij*y_a^i*y_b^j/K,
y_ab^i = -k^i*lambda_ab.
```

The second formula has a minus sign: twice differentiating g=0 gives
`0=-K*lambda_ab-H_ij*y_a^i*y_b^j`. There is no missing k0 derivative and
no independent derivative of a moving sphere. The source time satisfies
`(T-lambda*k0)_a=H_i*y_a^i`; this is also an independent graph-tangency
identity for the computed first root jets. The second version is
`-k0*lambda_ab=H_ij*y_a^i*y_b^j-H_i*k^i*lambda_ab`.

For the denominator,

```
K_a  = -k^i*H_ij*y_a^j,
K_ab = -k^i*H_ijl*y_a^j*y_b^l
       +(k^i*H_ij*k^j)*lambda_ab.
```

The last term comes from y_ab and has a plus sign. These formulas require
H through its third spatial derivatives, not its fourth derivatives.

## Full first and second integrand jets

Use source pullbacks

```
F_a  = F_i*y_a^i,
F_ab = F_ij*y_a^i*y_b^j-(F_i*k^i)*lambda_ab,
N = lambda*F,
N_a  = lambda_a*F+lambda*F_a,
N_ab = lambda_ab*F+lambda_a*F_b+lambda_b*F_a+lambda*F_ab.
```

Then the complete quotient derivatives are

```
I_a  = N_a/K-N*K_a/K^2,
I_ab = N_ab/K-(N_a*K_b+N_b*K_a+N*K_ab)/K^2
       +2*N*K_a*K_b/K^3.
```

Integrate these with the unchanged factor 1/(4*pi) and the same fixed
angular rule. The formulas retain both first-root cross terms and the
second denominator derivative. They never divide by F or s: zeros and
sign changes of individual target components are allowed. Replacing them
by logarithmic source derivatives would introduce an unnecessary singularity.

For completeness, c and F can be assembled from ordinary initial jets:

```
A_i = H_k*H_ki,
c_i = -A_i/c,
c_ij = -(H_ki*H_kj+H_k*H_kij)/c-A_i*A_j/c^3,
F_i = s_i*c+s*c_i,
F_ij = s_ij*c+s_i*c_j+s_j*c_i+s*c_ij.
```

Thus complete H<=3 and s<=2 suffice. These c formulas establish derivative
order; they are not a recommendation to subtract 1-|grad H|^2 near scri.
Root residual control alone does not certify the differentiated integral.

## Cancellation-aware layered evaluation

For this retained Minkowski reference write h=alpha_hat_bar, b>=0 and
nu=y/|y|, so H_i=(b/h)*nu_i and c=Omega/h. The symbol h here is the
reference conformal lapse, **not** the height H. Define

```
D = h*K = h*k0-b*nu.dot(k_space),
khat = k_space/k0,
D = k0*[Omega^2/(h+b)+(b/2)*|khat-nu|^2].
```

This positive sum uses h^2-b^2=Omega^2 and the null identity
|k_space|=k0. It avoids the cancellation in h*k0-b*nu.dot(k_space).
It requires an honestly null k from the fixed Lorentz construction; a
finite-arithmetic null defect must be checked, not silently absorbed into
the identity. No Omega, D, K, or angular floor is permitted.

For any source set G=s*Omega. For the native initial normal data the pinned
values prototype uses Pi=s/Omega^2, so G=Omega^3*Pi. Then

```
I = lambda*G/D,
lambda_a = (h*t_a-b*nu_i*e_a^i)/D,
lambda_ab = -h*H_ij*y_a^i*y_b^j/D.
```

Differentiate the **factored expression** for D in the source chart to
obtain D_i,D_ij, retaining all derivatives of Omega,h,b,nu and khat fixed.
Pull these and G back by

```
D_a=D_i*y_a^i,
D_ab=D_ij*y_a^i*y_b^j-(D_i*k^i)*lambda_ab,
G_a=G_i*y_a^i,
G_ab=G_ij*y_a^i*y_b^j-(G_i*k^i)*lambda_ab.
```

The quotient formulas above apply verbatim with F,K replaced by G,D.
They are an algebraically equivalent evaluation of the full derivatives,
not a truncation of small-Omega terms. Source Pi jets must be derived from
the complete initial lapse/shift data and reference coefficient jets; the
values-only prototype does not supply them. Height integration supplies H
for the root, while analytic H_R=b/h supplies its jets. Differentiating an
insufficiently converged height quadrature or a values-only interpolant is
not an admitted substitute.

Away from the source center, q=|y|, v=H_q=b/h gives explicitly

```
H_i=v*nu_i,
H_ij=v_q*nu_i*nu_j+(v/q)*(delta_ij-nu_i*nu_j),
H_ijl=(v_qq-3*v_q/q+3*v/q^2)*nu_i*nu_j*nu_l
      +(v_q/q-v/q^2)*(delta_ij*nu_l+delta_il*nu_j+delta_jl*nu_i),
v_q=(Omega^2/L)*d_r(v),
v_qq=(Omega^2/L)*d_r[(Omega^2/L)*d_r(v)].
```

Use the exact Cartesian core branch at q=0: H is constant there, so these
height derivatives are zero. Compute the smooth Cartesian normal data there
before introducing nu or 1/q. At both layer endpoints retain the established
exact branches and exp-flat derivative tails. Near an angular cap, evaluate
the small vector khat-nu at the declared high precision before squaring it;
positivity of the factorization alone is not a floating-error bound.

## Conditioning, outer events, and a later derivative gate

A normal-aligned boost chosen at the base reference event can reduce the
lab-frame cap concentration. It must stay constant for every first/second
derivative at that event. The components u^A remain in the original inertial
target basis. Repeating the integral with a second **constant** boost tests
parameterization independence; transforming the target components instead
would test a different quantity. At fixed event, high powers of K^-1 and
the source/ray derivatives can still be large. An angular rule adequate for
values need not resolve Hessians or their signed cancellations.

An outer failure-position reference radius r_e<S has finite
R_e=r_e/Omega_e and T=tau_ref+H(R_e). Every ray then has a unique finite
intersection, and the integral covers all source angular modes without a
finite-l truncation. This makes a separate quadrature oracle feasible in
principle at such **reference events**. Approaching scri sends both the
physical source range and the reference boost upward, and may sharpen the
derivative integrands. No fixed angular order, precision, or existing
values-only tolerance is certified here to resolve those events. Reading a
native abort position does not equate tau_ref with native target time.

A later, separately reviewed derivative recipe should pin complete source
jets and exact events, then include at least:

* Complete root, graph-tangency, and first/second implicit identities,
  positive K/D, finite operands, and two-precision agreement.
* H=0 constant and affine normal data with closed-form first/second jets.
* The pinned pure-CMC l=0,1,2 zero-Dirichlet exact solutions, including an
  oblique harmonic quadratic, testing all ten inertial Hessian entries.
* Independent constant-boost agreement and successive angular rules for
  values, four gradients, and ten symmetric Hessian entries of all four
  target components. Preserve per-component absolute as well as scaled
  error and cancellation evidence.
* The scalar wave trace -u_TT+sum_i u_XiXi=0, source symmetry, and exact
  initial normal-data limits. The wave trace is an additional necessary
  check, not an accuracy proof by itself.

Such a gate would supply J^A_a=delta^A_a+u^A_a and u^A_ab on the chosen
inertial reference events. It would not prove det J nonzero, timelikeness of
the target time gradient, global injectivity, an inverse at a native event,
constraint preservation by a numerical evolution, or any Minkowski/BH
stability result. The later user requirement remains a single hole surviving
the inner wormhole-to-trumpet transition while retaining the Minkowski
hyperboloidal reference; this pencil oracle establishes no part of that
evolution acceptance.

## Provenance and disposition

The unchanged integral, normal-data convention, and fixed-frame measure are
the reviewed derivations in the pinned initial-IVP note and values pencil.
This note's first/second chain-rule formulas were derived independently by
hand here. `source-pins.json` binds their exact source context and the
unchanged values prototype; it does not admit execution. No new primary
literature claim is made. Disposition: pencil formulas consistent; later
derivative quadrature and reconstruction gates remain held.
