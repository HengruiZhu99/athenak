# Corrected joint inner principal candidate: independent pencil review

PASS for source/pencil consistency only. The authoritative note is
ASSESSMENT-v2.md SHA1b3f607a3c18d30a594f7b759e26495469e3577c488f1bdbedffdcd843e6b854.
The original ce8aad displayed A_t beta transcription remains preserved as
incorrect; no scientific/source execution or numerical/CAS calculation occurred
in this review. The v2 note, actual C++ and scalar_matrix source were read.
The result supports preparing a separate actual full20 source gate, not an
implemented/principal-gated or adopted inner option.

## Scalar wave transformation

The eight displayed rows match scalar_matrix(f,mu,ea,ec), including
A_t=-2ell/3+cchi/3-h/2+2Lambda/3 and
Lambda_t=-4pi/3-2vartheta/3+4beta/3. The latter's Theta coefficient follows
actual ConformalRHS pole −(2/3)alpha*(2gradP+gradTheta)/Omega. This is a
P row: physical_k=P+2Theta, not an independent Kphysical row. kernel_symbol
sets trace and theta input amplitudes to Omega and divides those output rows
by Omega*alpha, giving pi=P/Omega,vartheta=Theta/Omega in the frozen symbol.

For H=h+2cchi, direct row addition gives
H_t=-2A+4pi/3+8vartheta/3; another substitution cancels the ell and Lambda
terms and yields H_tt=h+2cchi=H. For V=Lambda+2cchi, direct addition gives
V_t=2vartheta and V_tt=Lambda+2cchi=V. The remaining relation is
cchi_tt=(4mu-2ec)cchi/3+2(ea-1)ell/3+2(1-mu)V/3.
Also ell_tt=f*ell. All these are frozen principal equations with common
normal-derivative factors suppressed; they are not nonlinear time equations
with derivatives of f,mu,C omitted.

At ea1,ec=2mu²/(1+mu)², q>0 for mu>0 follows from
mu/(1+mu)²<=1/4. Factoring q−1 gives exactly
(mu−1)(4mu²+5mu+3)/(3(1+mu)²), hence the displayed rational C satisfies
C(q−1)=2(mu−1)/3. In X=cchi−CV the two light V terms cancel. No division
by q−1,q−f or f−1 appears, so collision values have no hidden singular
transformation. The explicit inverse is correct: pi=−ell_t/f,
vartheta=V_t/2,cchi=X+CV,beta=pi+2vartheta−3(X_t+CV_t)/2,
h=H−2cchi,Lambda=V−2cchi,A=2pi/3+4vartheta/3−H_t/2.
It proves finite-positive-f/mu completeness of this proposed frozen scalar
family, conditional on the actual kernel having the asserted modified rows.

The actual transverse source block is h_t=−2A+beta,
A_t=(Lambda−h)/2,Lambda_t=beta,beta_t=mu*Lambda. Defining
Z=(Lambda−h)/2 gives Z_t=A,A_t=Z, while Lambda_t=beta,beta_t=mu*Lambda
is an independent wave pair. This verifies semisimplicity at mu1 without a
resonant denominator. The unchanged tensor pair h_t=−2A,A_t=−h/2 is light.
This supplies20 frozen directions; it provides no uniform diagonalizer as
alpha,chi tend to zero, or an actual oblique/SPD compiled check.

## Physical rows, deviation sources and exact reference

The preserved rwm::Gauge C++ regular beta principal terms are
A0*Lambda−alpha*chi*gInv*gradalpha+.5alpha²*gInv*gradchi plus advection,
with analytic reference-deviation contractions. Its assembled lapse contains
−alpha²P/Omega, not −alpha²Kphysical/Omega. Complete reference connection/Z
terms are retained in the note's proposed base; dropping them would change
both source identity and reference stationarity.

B=(1−W)G0+W*A0 is positive when alpha,chi,G0 are positive.
mu=B/A0,kappa=B/(A0+B)=mu/(1+mu). Adding
(B−A0)(Lambda−Lambdahat) changes precisely the frozen Lambda coefficient
to B. The chi deviation gradient has principal derivative coefficient1,
so the second addition changes .5 to2kappa²=2mu²/(1+mu)². The lapse-gradient
coefficient stays epsilon_alpha1. The lapse correction
−2(1−W)alpha(P−Phat)/Omega changes alpha² toalpha²+2(1−W)alpha, hence
f=1+2(1−W)/alpha. At the reference all three deviation brackets vanish,
including chi_j−chi*chihat_j/chihat. At W1 every addition vanishes and the
full outer RWM source is recovered. Offconstraint P must remain P; only
Theta0 permits replacing it by physicalK for the Einstein-sector core BM
interpretation. The note makes that restriction explicitly.

The direct physical coefficients B and alpha²*2kappa² are bounded at the
collapsed core without evaluating mu, and the exact W1 branch must bypass
unneeded inner arithmetic. This is a pencil identity, not a robust tiny-value
implementation or proof that live finite fields remain admissible.

## Puncture and outer limitations

In the exact Minkowski/Cauchy core the asserted equation is correctly
beta_t=beta.gradbeta+G0Lambda−alpha*chi*gInv gradalpha
+2alpha²kappa²*gInv gradchi−eta_I beta. With differentiated alpha~r^p,
chi~r²,p>0 and bounded nondegenerate conformal metric, both displayed
gradient sources are O(r^(2p+1))=o(r). A nonzero G0 requires Lambda=O(r)
or genuine cancellation of stronger connection residues; bounded metric
values alone do not imply it. Lambda=lambda1*x gives precisely
G0lambda1+v²−eta_I*v=0. On Einstein data lambda1 is supplied by the
metric connection, not freely assigned independently of Z. Isotropic
Lambda0 still yields only the necessary eta_I=v condition.

The limiting coordinate gauge speeds sqrt(G0),sqrt(4G0/3) follow from
A0*mu=B and A0*q as A0 tends to zero; light/lapse speeds collapse. This
requires a genuine puncture characteristic/regularity treatment, not an
outflow assumption or well-posedness at alpha=chi0. Nonzero logarithmic
inner lapse restoration would alter the collapsed power balance; the
existing stationary class needs it zero. No mathematical finite-time
puncture-exponent argument is equated to finite-grid transition failure.

The outer physical-reference mass-log issue is unchanged, and no BH RHS
subtraction or hidden BH reference is proposed. Minkowski/native gates,
full actual symbol, finite contrast, all lower-order source/pole checks,
resolved-core and later wormhole-to-trumpet tests remain required.
