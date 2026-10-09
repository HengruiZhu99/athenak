# Regular core action and harmonic principal sectors

The exact flat-core tensor action now has a regular polynomial-envelope
formula at the Cartesian origin. A separate harmonic normal-principal
calculation identifies four constraint, four coordinate-gauge and two
screen-TT directions per characteristic sign. These are local continuum
and algebraic controls. They do not construct a radial Z4c solver or admit
a boundary condition, stable finite pulse or black-hole evolution.

## Division-free exact core action

Use the frozen Cartesian solid-CG basis with independent envelopes W_L(rho),
rho=x.x. For total J=0,1,2 there are8,16,20 amplitudes. At the exact flat
Cauchy core r<=.05 of the pinned reference,

```
Omega=alpha=chi=1, gtilde=I, beta=A=P=Theta=Lambda=0,
W_t=B0(rho)W+B1(rho)W_rho+B2(rho)W_rhorho.
```

Every coefficient is a polynomial in rho of degree at most two. The188
nonzero terms split28/69/91 by J. Exact homogeneous Cartesian reconstruction
checks44 primary columns and156 all-m columns, or468 independent radial-jet
actions. All residuals vanish exactly, including negative m. There are no
r/rho denominators, cross-L envelope conditions or excised origin. This
formula applies only to the exact flat core, not the transition or collar.

The physical metric trace is tau=tr(delta bar-gamma)/sqrt(3), with
delta chi=-tau/sqrt(3),delta gtilde=h_STF. At this reference A_ref=0, so
independent A is STF. The independent frozen Cartesian equations retain
the actual physical-P slicing alpha_t=-3P,shift beta_t=3Lambda/8 and
connection damping -10(Lambda-div h_STF). The trace/Theta terms are
P_t=-Delta alpha+10Theta and
Theta_t=-Delta tau/sqrt(3)+div Lambda/2-20Theta.
The negative-K convention and all STF Hessian factors are checked against
the full22 core oracle. No C1, Q lapse, damping profile or additional source
is combined with this action.

The generated evaluator consumes W,W_rho,W_rhorho and evaluates1,rho,rho^2.
At rho0 it gives the regular envelope action without dividing vanishing
Cartesian fields by r^L. Regular channels whose Cartesian value vanishes
at the origin remain independent.

## Compiled local and held-out checks

The declared19,500 queries cover all nonnegative-m real/imaginary phases,
four oblique directions,the origin,six positive core radii,and five
polynomial envelopes. The reused actual full dual bridge is byte unchanged;
its1055/1057 Release/ASan compiler dependencies and four link archives per
build are pinned. The standalone envelope evaluator is separately built.
Every point has Omega exactly one. Both native and polynomial output pairs
are byte identical between Release and ASan/UBSan; all four processes exit
zero with empty stderr.

| Comparison | Largest scaled action error | Largest absolute entry error |
|---|---:|---:|
| Full22 actual RHS versus polynomial action |3.98240e-16|1.77636e-15|
| Full22 input lift/layout |5.55112e-17|2.77556e-17|
| Held-out Cartesian full22 oracle |4.17262e-17|5.20418e-17|
| Held-out physical eight constraints |2.77556e-17|2.77556e-17|

The held-out driver supplies2800 cases:35 Cartesian monomials through
degree4,times20 physical tangent fields,times four core points including
the origin. Its seeds use no CG harmonics. It checks the unchanged
FlatFormula/FlatConstraints against actual full22 dual RHS and diagnostics;
it does not assert that arbitrary monomials lie in J<=2.

The action error is norm(residual)/max(1,norm(expected)). There are1254
exactly zero polynomial actions and5395 nonzero actions below1e-12; the
smallest nonzero norm is1.90731e-81. These checks do not claim relative
accuracy of each near-zero action. Raw algebraic-normal residuals are
retained before output projection and remain below4.441e-16 scaled.

All18 earlier fitted core blocks at r=.025/.05 also pass. Their worst scaled
discrepancy is8.40735e-11 in J2/B1/r.025,whose maximum entry error is3.30216e-9.
The largest entry error across all18 blocks is5.89108e-9 in J2/B0/r.025.
The older fit there has raw condition3.16742e6. This distinction corrects
the frozen core REPORT's wording of the former value as the overall largest
entry error; no saved result, tolerance or acceptance changes.

The historical first symbolic assertion is preserved: it rejected a
candidate rho degree before checking that its angular coefficient was zero.
Moving the degree assertion after that exact zero test changed no basis,
core formula, coefficient, envelope freedom or tolerance. Exact sources,
logs,receipts and both read-only source reviews accompany the accepted gate.

## Physical constraint and principal-sector normalization

At fixed positive Omega,alpha,chi and SPD Penrose metric,use a Penrose
orthonormal frame (n,T,U). The harmonic W_gauge1 reduction contains ten
normal configuration derivatives and ten momenta/connections. In particular

```
a=D_s(delta alpha)/alpha, c=D_s(delta chi)/chi,
h_ij=D_s(delta gtilde_ij)/chi, b_i=D_s(delta beta_i)/alpha,
p=delta P/Omega, t=delta Theta_phys/Omega,
A_ij=delta Atilde_ij/chi, l_i=chi*(frame delta Lambda)^i.
```

They are not twenty raw stored values. The physical diagnostic definitions
give the independent reduced map

```
Zn=(ln-h_nn)/2, ZA=(lA-h_nA)/2,
Hred=h_nn+2c,
Mnred=A_nn-(2/3)(p+2t), MAred=A_nA.
delta H_pr=Omega^2 D_s Hred,
delta M_Penrose,pr=Omega D_s Mred,
delta M_physical-orthonormal,pr=Omega^2 D_s Mred.
```

Theta_phys=Omega*t and physical-orthonormal Z=Omega*Z_Penrose. The exact
normal symbol has A^2=I, and this rank-eight constraint map satisfies
CA=BC,B^2=I. Four constraint combinations per sign lambda=+/-1 are

```
Hred-2lambda Mnred,
t+Mnred+lambda Zn,
ZA+lambda MAred, A=T,U.
```

Four coordinate rightvectors are independently derived from Lie_xi eta
with contravariant time component xi^tau and Kbar=-Lie_n bar-gamma/2.
Together with two screen-TT metric/A polarizations,they exhaust the
six-dimensional principal Einstein kernel for each sign. The complete
left basis has rank20. Off that kernel,gauge rows can mix with constraint
rows; the displayed classification is not asserted to be orthogonal in
H=I+A^T A. All288 retained harmonic actual-kernel matrices agree with the
exact rational A within8.882e-16; no new kernel is executed for this check.

Hred and Mred use one fewer normal derivative than H and M. Their physical
interpretation uses inverse normal Fourier differentiation at nonzero k,
not a local condition at k0. Tangential derivatives,coefficient gradients,
lower-order sources,reduction constraints and the singular Omega limit
remain outside this normal-principal calculation.

For y_t=(beta_n I+alpha A)D_s y at finite outer CMC radius rb<S,the lambda+
branch is incoming: beta_n+alpha=(S-rb)^2/(2aS)>0 in RHS convention. The
lambda- branch is outgoing. Its formal ten incoming directions split4+4+2;
this does not justify zeroing them,primitive Dirichlet data or CPBC.
The ten-direction count belongs to the full normal symbol; a fixed-J radial
restriction must derive its own restricted boundary ranks and trace space.

## Remaining scope

No radial discretization or global matrix,SAT adoption,eigensolve,
propagation,full constraint hierarchy or finite-pulse stability is supplied.
The origin formula is a smooth Cartesian core control,not a puncture space.
Production remains implementation27c19d20696ea6dd4704032c51dfd026218f64f2.
The later single-BH acceptance target still requires the inner
wormhole-to-trumpet transition with the Minkowski hyperboloidal reference
retained throughout.

The [byte-preserved evidence archive](validation/hyperboloidal-core-principal-controls-experiments-20261009/README.md)
contains118 cataloged blobs,2,334,216 bytes and33 finite JSON files. Its catalog
SHA256 is `a859c59d6fa2b51635728581e2af16207d43e19fb35189a1ea171c673cea4358`.
Original core/principal index SHA256 values are
`b0fde1e0eb95d6660a9fa3d190eda69207e6153c88da35366b038369ac3aa3d4` and
`05d4d7308477efe26d26ec7256fd9ba824040849363ed7f723031e5760adc0fc`.
The read-only verifier passes without scratch dependencies or scientific
reruns. Original reviewed drafts,source/math reviews and historical failures
are preserved. Executables,objects,NumPy binary arrays and payloads larger
than1MiB remain hashes/sizes/metadata only.

The subsequent [configuration derivative and reference/frame control](hyperboloidal-configuration-derivative-audit.md)
validates the actual linearized configuration-source derivative used by the
planned radial weak/strong energy comparison, with its scope and historical
binary preservation limitation stated explicitly.
