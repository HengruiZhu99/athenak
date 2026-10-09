# Harmonic normal principal constraint, coordinate and tensor sectors

This is a fixed-positive-Omega, normal-principal calculation. It derives
the physical diagnostic map independently, checks exact rational matrices
against 288 retained actual-kernel harmonic records, and proves four
constraint, four coordinate-gauge and two screen-TT directions per sign.
It does not execute a tensor kernel or construct a radial PDE operator,
incoming boundary condition, CPBC or energy estimate.

## Normalization and differential order

Freeze a positive lapse alpha, positive chi, Omega>0 and an SPD Penrose
spatial metric bar-gamma=gtilde/chi. Choose its orthonormal frame (n,T,U),
and let D_s denote the derivative along n. All frozen-frame transformations
are constant for this principal calculation. Use the exact ordering of
`tst/hyperboloidal/kernel_symbol.cpp`:

```
y=(a,c,h,p,t,Ann,ln,b,
   hT,AnT,lT,bT,hU,AnU,lU,bU,hplus,Aplus,hcross,Across),
a=D_s(delta alpha)/alpha, c=D_s(delta chi)/chi,
h_ij=D_s(delta gtilde_ij)/chi in the Penrose orthonormal frame,
p=delta P/Omega, t=delta Theta_phys/Omega,
Aij=delta Atilde_ij/chi in that frame,
li=chi*(Penrose-frame contravariant delta Lambda)^i,
bi=D_s(delta beta^i)/alpha in that frame.
```

Here h=h_nn, hA=h_nA, hplus=(h_TT-h_UU)/2, hcross=h_TU, and
analogously for A. The determinant-one and trace-free algebraic tangents
leave five metric and five A components at principal order. Background A
times undifferentiated metric variations enters the exact A-trace tangent
but is lower differential order in this reduction. It must be restored in
a full variable-coefficient lift.

The stored trace is P=K_phys-2Theta_phys, so the physical trace variation
is delta P+2delta Theta_phys. The ten derivative variables are indices
0,1,2,7,8,11,12,15,16,18. The other ten variables are momenta or connections.
These are not twenty raw stored values. With dimensionless alpha, beta,
chi and Omega, every component of y has dimension inverse length.

At W_gauge=1, the normal system is

```
y_t=beta_n D_s y+alpha A D_s y + lower differential order,
A^2=I.
```

The exact matrix A and every row below are saved in report.json. The
preferred source, physical-P/conformal-Q source choices, damping and the
tested algebraic feedback do not change this harmonic principal matrix.
This statement is not a claim about their lower-order equations.

## Physical diagnostic map

The source definitions are `EvolvedConstraints()` and `Z4Constraints()`.
At the frozen frame, D_s(delta bar-gamma)=h_ij-c delta_ij. Linearizing
the scalar Ricci curvature gives

```
delta Rbar_pr=D_s[(h_nn-c)-tr(h-cI)]=D_s(h+2c).
```

The divergence of A and gradient of the physical trace give the momentum
principal part. The connection identity is
Z_i=gtilde_ij*(Lambda^j-Gamma_tilde^j)/2. Thus define

```
Zn=(ln-h)/2, ZA=(lA-hA)/2,
Hred=h+2c,
Mnred=Ann-(2/3)(p+2t), MAred=AnA,
z=C y=(t,Zn,ZT,ZU,Hred,Mnred,MTred,MUred).
```

These reduced quantities retain the following physical weights:

```
delta H_pr=Omega^2 D_s Hred,
delta M_i,pr (Penrose orthonormal covector)=Omega D_s Mired,
delta M_i,pr (physical orthonormal covector)=Omega^2 D_s Mired,
delta Theta_phys=Omega t,
Z_i (physical orthonormal)=Omega Z_i (Penrose orthonormal).
```

Hred and Mred have one fewer normal derivative than the physical H and M.
At nonzero normal Fourier frequency k they mean (ik)^-1 H/Omega^2 and
(ik)^-1 M_Penrose/Omega, respectively, at principal order. The k=0 case
does not admit this inverse-derivative interpretation. This is a
pseudodifferential reduction, not a local eight-field physical boundary
condition. No stronger Theta or Z falloff is imposed.

Terms involving coefficient derivatives, Omega gradients, background
curvature, reference sources and tangential derivatives are not in this
normal principal map. They are needed in a full boundary reduction.
Their omission is justified by differential order at a fixed Omega>0;
it is not a uniform estimate as Omega tends to zero.

## Closed reduced constraint symbol

The exact independently constructed C has rank eight and obeys CA=BC,
where, in the order above,

```
B z=(Zn+Hred/2, t+Mnred, MTred, MUred,
     -2Mnred, -Hred/2, ZT, ZU),
B^2=I.
```

For each lambda=+1 or -1, four independent left characteristic
combinations are

```
cH_lambda=Hred-2lambda Mnred,
cZ_lambda=t+Mnred+lambda Zn,
cT_lambda=ZT+lambda MTred,
cU_lambda=ZU+lambda MUred.
```

Each row has eigenvalue lambda for B and for A after composition with C.
C P_lambda has rank four, P_lambda=(I+lambda A)/2. Multiplication by D_s
would replace Hred/Mred with the physical diagnostics and derivatives of
Theta/Z, with the weights above, but only at this frozen principal level.
It would not supply the missing lower-order, angular, reduction-constraint
or physical boundary data.

## Independent coordinate and screen-TT directions

The six-dimensional kernel of C within each ten-dimensional eigenspace
is not identified solely by dimension counting. Four independent vectors
are derived from a frozen flat-spacetime coordinate pullback
delta g_ab=Lie_xi eta_ab. Write tau for normalized normal time and assume
D_tau xi=lambda D_s xi. The time generator is the contravariant xi^tau;
spatial components are unchanged by raising in this orthonormal frame.
Absorb the common second normal derivative into
zeta=(zeta_time,zeta_n,zeta_T,zeta_U), with zeta_time=D_s^2 xi^tau.
The sign convention is

```
delta alpha/alpha=D_tau xi^tau,
delta beta_i/alpha=D_tau xi_i-D_i xi^tau,
delta bar-gamma_ij=D_i xi_j+D_j xi_i,
delta Kbar_ij=-D_i D_j xi^tau.
```

The last expression follows from Kbar=-1/2 Lie_n bar-gamma, including
the shift terms. The four coordinate vectors in y are therefore

```
a=lambda zeta_time, c=-2zeta_n/3, h=4zeta_n/3,
p=-zeta_time, t=0, Ann=-2zeta_time/3, ln=4zeta_n/3,
b=lambda zeta_n-zeta_time,
hA=zeta_A, AnA=0, lA=zeta_A, bA=lambda zeta_A,
hplus=Aplus=hcross=Across=0.
```

The checker derives them from the four-dimensional Lie derivative and
negative-K formula rather than assuming eigenvectors. Their matrix G
has rank four, AG=lambda G and CG=0. Fixed prescribed-Omega pullbacks
also have terms involving xi.dOmega/Omega; these are lower differential
order at fixed Omega and cannot be dropped in an actual scri timejet
or boundary-gauge analysis.

Two screen-TT directions have one metric polarization equal to one and
the matching A polarization equal to -lambda/2, with all other fields
zero. They have rank two, AT=lambda T and CT=0. This is the usual
normal-principal tensor polarization split; no complete finite-radius
Weyl/radiation boundary prescription is inferred.

The concatenation (G,T) has rank six. The stacked matrix (C;A-lambda I)
has rank fourteen, so these exhaust the principal Einstein kernel for
that sign.

## One complete left basis and incoming convention

Together with the four constraint rows, a convenient choice of four
gauge and two TT left rows is

```
gtime_lambda=p-lambda a,
gnormal_lambda=b-p-t/2+(3lambda/4)ln,
gA_lambda=lA+lambda bA,
TT_lambda=A_TT-(lambda/2)h_TT.
```

The gauge rows annihilate both TT columns and restrict on the coordinate
vectors to diag(-2,2lambda,2,2). The TT rows annihilate coordinate vectors.
Each sign has ten independent left rows; the two signs give rank twenty.
Gauge rows may be altered by constraint rows without changing their
restriction to the Einstein principal kernel. Classification away from
that kernel is consequently not unique. These rows are not asserted to
be orthogonal in H=I+A^T A; an energy-normalized boundary lift needs its
own treatment.

In the RHS convention y_t=(beta_n I+alpha A)D_s y, coordinate propagation
speed is -(beta_n+lambda alpha). At a finite outer CMC radius rb<S,
lambda=+1 is incoming, with beta_n+alpha=(S-rb)^2/(2aS)>0. The lambda=-1
branch is outgoing, beta_n-alpha=-(S+rb)^2/(2aS)<0. Hence the formal ten
incoming directions split into four constraint, four coordinate and two
screen-TT directions. The split alone does not justify setting any of
them to zero, impose primitive Dirichlet data or establish CPBC.

## Reproduction and limits

Run `run_check.py` with the Python executable recorded in receipt.json,
from a fresh working copy/output path. It calls only `check_sectors.py`.
The command uses SymPy1.14.0 exact arithmetic and reads historical actual
principal matrices; it invokes no C++ executable. All exact identities,
ranks and independent pullback checks pass. The 288 retained W=1 matrices
agree with A within 8.881784197001252e-16. Source hashes before and after
the command agree, the command exits zero and stderr is empty.

No finite-frequency roots, constraint-ideal invariance, scri hierarchy,
full angular boundary system, SBP identity, radial closure, propagation,
positivity, compact-pulse acceptance or black-hole transition is proved.
