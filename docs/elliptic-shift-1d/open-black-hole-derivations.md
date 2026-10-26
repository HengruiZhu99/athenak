# Open black-hole derivations (not implemented or admitted)

These are candidate next steps. None is a successful evolution, an initial-data
solver, or permission to replace a failed experiment with a different physical
one. Native constraints and independent convergence remain required.

## A nonlinear restoring-source candidate

The symmetric Minkowski source's conditional stationary-puncture obstruction
is derived in `positive-variables.md`. Multiplicative restoring removes its
constant inner drive but is locally antidamped near zero lapse. One possible
smooth source with stable zeros at alpha=0 and alpha=A is

    S_restore=-2xi alpha(alpha-A)(2alpha/A-1), A=(1+r^2)/2.

There is an intermediate unstable zero alpha=A/2. At alpha=A the derivative
is -2xi A, identical to both existing sources' Minkowski linearization. At
alpha=0 it is -2xi A, instead of the multiplicative +2xi A. Relative to the
multiplicative source the change is -4xi alpha u^2/A, u=alpha-A, so the
Minkowski equations are unchanged to linear order. Its first scri normal
correction also vanishes when u(S)=0. This is a nonlinear gauge modification,
not a constraint projection, and could trap sufficiently large pure gauge
perturbations on its collapsed branch. It needs native/limit audits and pulse,
collapse and wormhole controls before any adoption. It has not been coded.

For an AF puncture alpha=r^2 A0+..., the restoring part of the A0 ODE would
change from +2xi A0 to -2xi A0. The additive geometric +2 term is independent
of that restoring choice. The other independent AF curvature/constraint
coefficients must remain evolved.

## Stationary hybrid-gauge trumpet boundary problem

For a stationary vacuum embedding in Schwarzschild, define H=n(R), h=H/R,
Y=1/(Omega R), k=K+3 and chi=r^2Y^2. In the g=1 isotropic spatial gauge,
R_r/R=alpha/(Omega r), K=-H_R-2H/R. The following equations result:

    Omega r Y_r-(r^2-alpha)Y=0,
    Omega r h_r-alpha(3-3h-k)=0,
    Omega r h alpha_r+F k-r^2(1-alpha h)+R_restore-G=0,
    alpha^2-Omega^2(1-2M Omega Y)-h^2/Y^2=0.

Here R_restore=-S_restore and G is an optional explicitly labeled analytic
lapse-reference forcing. The main hybrid target has G=0. On a constraint-free
stationary slice, Khat=K and
Ar=2[k+3(h-1)]/(3Omega). This reconstructs the CMC expression when k=0.
The geometric and gauge equations can become singular at h^2=F Y^2, the
stationary gauge horizon; a regularity condition is needed there. At scri the
bounded values are Y=alpha=h=1,k=0. At a trumpet alpha tends to zero while
Y and negative h remain finite. These are singular endpoint equations: their
values and normal-jet conditions cannot be counted as ordinary independent
Dirichlet conditions. Rank/indicial analysis and a properly posed numerical
boundary problem remain open. No stationary hybrid data have been solved.

## Scalar perturbation of a CMC trumpet: inner constraint condition

For a physical conformal perturbation psi of vacuum CMC trumpet data, keeping
K=-3 and choosing zero physical scalar normal momentum gives zero scalar
momentum density. In that restricted initial-data family, the radial TT
curvature scales as Ar=Ar_background psi^-6 and
chi=chi_background psi^-4. The scalar Hamiltonian equation still needs to be
derived and solved; this is not the regular-center solver already in the code.

In an exactly vacuum inner neighborhood, write chi_background~X0 r^2 and
Ar_background~B0. Linearizing the Hamiltonian with psi=1+epsilon r^s gives

    delta H=[-2X0(1+s^2)+(9/2)B0^2] delta psi.

Since X0=4/R0^2, B0=-4C/R0^3 and C^2/R0^4=R0^2+1/3,
its indicial roots are s=+/-sqrt(2+9R0^2)=+/-gamma. The bounded branch is
psi-1 proportional to r^gamma. For a fixed finite cutoff, merely imposing
psi=1 is generally inaccurate. An exact local vacuum matching condition,
with R=R_background psi^2 and fixed interior M,C, is instead

    psi_r/psi=[sqrt(f(R_background psi^2))-sqrt(f(R_background))]/(2r),
    f(R)=1-2M/R+(R-C/R^2)^2.

This follows from isotropy R_r/R=sqrt(f)/r and requires an actually vacuum
inner neighborhood. A compact scalar pulse can supply that for a preliminary
control; a Gaussian with a nonzero inner tail requires a separate justification
or a different constraint boundary treatment. No such solver is implemented.

## Scalar momentum conventions must be checked against each source

The code uses f=phi/(Omega sqrt(chi)) and
normal_bar(phi)=sqrt(chi)[Omega P+(K/3-Omega Ar/2)f]. Let
barphi=phi/Omega and Pi_bar=normal_bar(barphi). Then

    Pi_bar=sqrt(chi)[P+(K/3-Omega Ar/2-beta r/alpha)f/Omega],
    normal_bar(phi)=Omega Pi_bar+(beta r/alpha)barphi.

Thus zero normal momentum of phi and zero normal momentum of barphi are
not equivalent. If a rescaled Gaussian has a nonzero scri value, choosing
normal_bar(phi)=0 would generally make P singular. With Pi_bar=0 the numerator
K/3-Omega Ar/2-beta r/alpha has a compatible zero at scri. Its nonzero physical
scalar momentum then also sources the momentum constraint, so a Hamiltonian-only
TT initial-data solve is insufficient. The pinned primary text [2506.15311v4](https://arxiv.org/html/2506.15311v4)
has now been checked: its Eq.(21) variable c_Pi is the *coordinate-time*
derivative of barphi, not Pi_bar above. Equation(22), used for its sigma=0
QNM case, gives zero physical normal momentum. Equation(27) with sigma=-1
instead requires the additional momentum solve. Its neutral Fig.3 pulse is
r^2 A exp[-(r^2-mu^2)^2/width^4], A=.0005,mu=.4,width=.05, not the broader
.001,.5,.1 tail-family profile. The inventory is corrected accordingly. This
narrow pulse and its endpoint regularity need to be resolved explicitly; no
published black-hole scalar perturbation has been implemented or admitted.

## Preferred next source test: physical collapsed-lapse asymptotics

The sign of a restoring term alone is not the whole inner gauge: the lapse
also contains the reference offset Khat+3. Write nu_inner for the coefficient
F/alpha as alpha tends to zero at r=0 (nu for linear/blended slicing, or
nu(.5+scale)/scale for the rational family). The existing multiplicative AF
coefficient equation is

    A0_t=(2b1-6nu_inner+2xi)A0+2.

Thus its total local coefficient can already be damped even though its restoring
piece has a positive derivative. The CMC source-control antidamping diagnosis
must not be transferred uncritically to the main AF experiment.

A more directly motivated candidate than the bistable source is a smooth
restoring profile

    xi_eff(r)=xi+(3nu_inner-xi)(2Omega)^4,

combined with the *already implemented* collapsed-inner background correction.
The profile equals3nu_inner at the inner end and xi at scri; its first three
normal corrections vanish there. With multiplicative restoring, the inner
reference drive -6nu_inner alpha is canceled by +2xi_eff(0)alpha. The collapsed
background correction cancels the separate additive2r^2 term. Therefore

    A0_t=2b1 A0

for a leading AF end with Khat=O(r^2), matching physical1+log transport at
Khat(0)=0. More generally at a collapsed trumpet the leading physical lapse
source becomes -2nu_inner alpha_physical Khat, retaining the intended live
reference-reduced Khat choice off constraint. Minkowski equilibrium remains
exact; the interior linear source changes, while the first scri limits remain
unchanged. This is an unimplemented source-profile proposal, not a successful
wormhole evolution. Test its native/AF limits, regular pulse behavior, separated
source-only/profile-only controls and genuine wormhole/collapse evolutions
before considering adoption. Keep kappa4,-.99 and a separately labeled
reference-lapse damping control so source and damping effects remain visible.
