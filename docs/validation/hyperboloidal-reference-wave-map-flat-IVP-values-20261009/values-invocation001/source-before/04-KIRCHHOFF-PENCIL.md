# Pencil derivation of the independent retarded-integral oracle

Source-only, no CAS or numerical evaluation. Signature eta=(-,+,+,+). Let the
initial spacelike graph be T=H(R), v=H_R, |v|<1, future unit normal
n=(1,v*nu)/sqrt(1-v^2). The complete layered reference has v=b/h and
sqrt(1-v^2)=Omega/h. Let u|Sigma=0 and n(u)=s. Every target-index component is
an independent scalar obeying Box_eta u=0.

## Cauchy Green formula and ray reduction

Using the retarded wave kernel normalized to the usual T=0 Kirchhoff solution,
the zero-Dirichlet term on Sigma is

```
u(T,X)=integral [delta(T-H(|y|)-|X-y|)/(4pi*|X-y|)]
                s(y)*sqrt(1-H_R(|y|)^2) d^3y.
```

The sign is fixed by H=0: this gives
u(T,X)=T*(1/4pi) integral s(X-T*omega) d omega for T>0. The derivative-of-kernel
term is zero because the initial field is identically zero, not because its
normal derivative is zero. Restrict this representation to the future domain
of dependence of the graph; it is not a global Cauchy formula outside that
domain and does not license dropping a nonzero endpoint/incoming contribution.

Set y=X-lambda*omega. The radial delta argument has derivative

```
f_lambda=-1+v(y)*nu_y.dot(omega)=-D,
D=1-v(y)*nu_y.dot(omega)>0.
```

At lambda=0, f=T-H(|X|)>0. Since the retained H>=0 and |v|<1, f is strictly
decreasing and is nonpositive by lambda=T; the root is unique for a finite
future point. The core uses Cartesian v*nu=0 at R=0. Thus

```
u(T,X)=(1/4pi) integral lambda*s(y)*sqrt(1-v(y)^2)/D d omega,
f(lambda)=0.
```

This formulation retains the entire initial normal velocity, including its
noncompact physical tail and infinitely many angular harmonics. There is no
finite computational incoming boundary. Finite target events have finite
source intersections. A uniform limit towards scri still requires estimates
on the large source radii and on the angular concentration; finiteness at each
point is not a uniform error certificate.

With a fixed lab frame the implicit derivatives include

```
lambda_T=1/D,
lambda_Xi=-v*nu_i/D,
y_T=-omega*lambda_T,
y_Xi=e_i-omega*lambda_Xi.
```

Differentiating these and the weight provides exact first/second inertial jets
once complete H and s derivatives are supplied. D>0 at finite sources is a
mathematical property; an unresolved very small D must trigger a conditioning
or precision stop, not a floor.

## Fixed boosted frame for angular conditioning

Choose any constant proper orthochronous Lorentz matrix B and let
k=B(1,omega), a future null vector with k0>0. Parameterize q=p-ell*k. Relative
to the lab sphere, lambda=ell*k0 and d omega_lab=(k0)^(-2)d omega. Therefore

```
K=k0-v*nu.dot(k_space)>0,
-n(q).k=K/sqrt(1-v^2),
u(p)=(1/4pi) integral ell*s(q)/[-n(q).k] d omega,
q0=H(|q_space|).
```

All factors of k0 cancel. This is the same integral, not a different source or
coordinate pulse. A normal-aligned B at the base query can reduce large
Lorentz factors near the outward angular cap. Hold B constant over the local
jet differentiation; no derivative of a query-dependent chosen boost is then
missing. Independently comparing two choices of constant B tests the result.
The components u^A remain in the original fixed target frame; B changes the
integration parameterization only.

For the boosted root, f_ell=-K, ell_T=1/K and
ell_Xi=-v*nu_i/K. At a future point lambda<=T in the lab parameterization,
which supplies a simple finite root bracket. A transformed bracket can be
obtained by lambda=ell*k0. Root jets must be differentiated from the same
equation used for the root value.

## Coarea in initial source radius and source azimuth

This independently checks the root's proposed efficient reduction. Let
Re=|X|>0, q=|y|, mu=nu_X.dot(nu_y), and
d=sqrt(Re^2+q^2-2*Re*q*mu). In the Green formula,
partial_mu(T-H(q)-d)=Re*q/d. The delta integration therefore gives the exact
weight

```
u(T,X)=(1/(4pi*Re)) integral_allowed q*sqrt(1-H_q^2)
                            [integral_0^(2pi) s(q,nu(mu,az)) d az] dq,
mu=[Re^2+q^2-(T-H(q))^2]/(2*Re*q).
```

The radial measure has an explicit factor q. It is not just dq. The kernel's
1/d has canceled, not been dropped. The allowed radii satisfy
|Re-q|<=T-H(q)<=Re+q. Set u_ret=T-Re, D_h(q)=H(q)-q, w=u_ret-D_h(q). Then

```
mu=1-w/q+w/Re-w^2/(2*Re*q),
q_upper: H(q)+q=2*Re+u_ret,
q_lower: D_h(q)=u_ret for u_ret<0,
         H(q)+q=u_ret for u_ret>=0.
```

Here H+q is strictly increasing, and D_h is strictly decreasing at finite
q>0. Since a future event obeys u_ret>D_h(Re), the negative-u_ret lower root
exists below Re. If u_ret=0, the lower endpoint is q=0. Core center limits
must be taken before divisions by q or Re. No admissible radius satisfies a
nonpositive retarded distance except a degenerate initial endpoint.

In compact initial radius r, dq=L/Omega^2 dr and sqrt(1-H_q^2)=Omega/h. Thus

```
q*sqrt(1-H_q^2)*dq = r*L/(h*Omega^2) dr.
```

For a reference event, Re=r_e/Omega_e and dividing u by Omega_e gives the
regular compact phi formula in PLAN.md. This is a calculation of the same
physical u; no different conformal wave data have been substituted.

The cancellation-free height defect obeys

```
dD_h/dr=(b/h-1)*L/Omega^2=-L/[h*(h+b)],
D_h=-r in the core,
D_h=C+a^2/(sqrt(q^2+a^2)+q) in the exact outer branch.
```

The identity h-b=Omega^2/(h+b) proves the derivative. At the event center the
original ray formula reduces to q_star satisfying H(q_star)+q_star=T,
u=q_star*sqrt(1-H_q^2)/(1+H_q)*sphere_average(s). This is the proper center
formula, not the Re->0 quotient evaluated numerically.

For flat H=0 and constant s=v0, the interval is
[|Re-T|,Re+T]. The coarea expression is
v0/(2*Re)*integral q dq=v0*T, while the center expression is v0*T as well.
This checks the 4pi factor and sign independently. Differentiating the
coarea representation needs moving-endpoint contributions. Near mu=+/-1,
the full azimuthal average has regular even transverse moments; differentiating
the raw square-root direction chart term by term without handling that limit
is not an admitted derivative method.

## Exact oracle families with zero initial field

For H=0, constant/affine s produces the ordinary flat Kirchhoff solution and
tests the overall sign and first derivatives. For a separate pure-CMC graph
H(R)=sqrt(R^2+a^2)+C, write z=(T-C)^2-R^2. If P_l(X) is a homogeneous spatial
harmonic polynomial of degree l, then

```
u=P_l(X)*[1-a^(2l+2)/z^(l+1)],
Box_eta u=0,
u|Sigma=0,
n(u)|Sigma=2(l+1)*P_l(X)/a.
```

The wave identity follows by differentiation:
Box_eta(P_l*f(z))=-4*P_l*[z*f''+(l+2)*f']; z^(-(l+1)) solves this radial ODE.
The normal derivative uses n(z)=2a on the initial graph. Use l=0,1,2 with a
non-axis-aligned harmonic quadratic as independent derivative/angle controls.
These are oracle tests only, not substitutes for the native pulse or for the
layered initial surface. The future region z>a^2 avoids their singular cone.

## Energy and reconstruction limits

At Sigma, grad(u)=-s*n_flat, so the physical Killing-energy density is
(-n.dot(partial_T))*s^2/2. Multiplying by the graph volume
sqrt(1-v^2)R^2 dR dOmega gives s^2 R^2 dR dOmega/2. This is not a pointwise
Jacobian bound. Commuted-energy/Sobolev estimates or validated derivative
quadrature on a stated domain are needed before using it for global claims.

After solving the waves, native coordinates require the full inverse equation
X+u(X)=Yhat(t_native,x_native). The inverse's active linear generator is -u,
and native t is not the reference hyperboloidal evaluation time. A nonzero
Jacobian, target-time timelike gradient, global injectivity and coverage are
distinct conditions. The integral itself decides none of them without the
planned derivative and inverse-map gates.
