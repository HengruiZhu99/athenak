# A complete coupled inner gauge family to test

Source/pencil only,2026-10-09. No CAS, numerical mathematics, executable helper,
kernel query or evolution is run. This is a candidate derivation, not adoption.
The later BH goal remains a wormhole-to-trumpet transition with the hyperboloidal
Minkowski reference retained. Existing native Minkowski failures are not
explained or repaired by this inner calculation.

The fixed reference and physical-reference wave-map gauge are retained outside
the inner-gauge cutoff. The old exact-core weighted connection driver becomes
weak as alpha²chi tends to zero; increasing only that coefficient crosses
known defective longitudinal/light or longitudinal/lapse collisions. The
following *joint* shift-gradient choice has a direct finite-positive-lapse
principal completeness proof and a bounded chi-gradient coefficient.

## Actual scalar block and conventions

Use the frozen orthonormal constrained20 normalization from kernel_symbol.cpp.
The scalar eight fields are

    (ell,cchi,h,pi,vartheta,A,Lambda,beta),

where ell is the principal normal derivative of delta alpha/alpha, cchi that of
delta chi/chi, h the tracefree metric nn derivative, pi=P/Omega,
vartheta=Theta_phys/Omega, A=Atilde_nn in the same orthonormal normalization,
Lambda the normalized contracted connection and beta the normal derivative
of delta beta/alpha. A common normal derivative and normalized time factor
are suppressed in the following matrix equations. The production physical
trace is P=K_phys-2Theta_phys; pi is not K_phys/Omega. Reference P_hat=K_phys_hat
because reference Theta=0. In the exact Omega1 core pi=P and vartheta=Theta.

The actual source-inspected scalar principal block is

    ell_t = -f pi
    cchi_t = 2pi/3+4vartheta/3-2beta/3
    h_t = -2A+4beta/3
    pi_t = -ell
    vartheta_t = cchi+Lambda/2
    A_t = -2ell/3+cchi/3-h/2+2beta/3
    Lambda_t = -4pi/3-2vartheta/3+4beta/3
    beta_t = -epsilon_alpha ell+epsilon_chi cchi+mu Lambda.

These are pseudodifferential derivative-order statements, not an undifferentiated
local constraint boundary condition. The two transverse blocks and two tensor
blocks are exactly those of the implemented20 symbol.

Define H=h+2cchi and V=Lambda+2cchi=H+2Z_n, where
Z_n=(Lambda-h)/2. Direct substitution gives

    H_t=-2[A-2(pi+2vartheta)/3], H_tt=H,
    V_t=2vartheta, V_tt=V,
    ell_tt=f ell,
    cchi_tt=q cchi+2(epsilon_alpha-1)ell/3+2(1-mu)V/3,
    q=(4mu-2epsilon_chi)/3.

The second-time equations suppress the common second normal derivative as
well. H,V and their momenta encode the four scalar principal constraints.
They do not impose any stronger nonlinear or scri falloffs.

## Collision-safe coefficient choice and explicit inverse

Choose, for finite f>0 and mu>0,

    epsilon_alpha=1,
    epsilon_chi=2mu²/(1+mu)²,
    q=(4mu-4mu²/(1+mu)²)/3,
    C=2(1+mu)²/(4mu²+5mu+3),
    X=cchi-CV.

Then q>0 because mu/(1+mu)²<=1/4, and

    q-1=(mu-1)(4mu²+5mu+3)/(3(1+mu)²),
    C(q-1)=2(mu-1)/3,
    X_tt=qX.

The term epsilon_alpha=1 removes lapse-to-shift forcing at q=f. The rational
C cancels the light-to-shift forcing at q=1 without dividing by q-1.
At mu1 the finite limit is C=2/3,epsilon_chi=1/2,q1. No pole or missing
eigenfield is hidden there.

The map from the scalar eight fields to

    (ell,ell_t,X,X_t,H,H_t,V,V_t)

has the explicit inverse

    pi=-ell_t/f, vartheta=V_t/2,
    cchi=X+CV, cchi_t=X_t+CV_t,
    beta=pi+2vartheta-3cchi_t/2,
    h=H-2cchi, Lambda=V-2cchi,
    A=2pi/3+4vartheta/3-H_t/2.

It is finite and invertible for every f>0,mu>0. It conjugates the scalar block
to four independent wave pairs with speed squares f,q,1,1. Consequently all
q1,q=f,f1 coincidences are semisimple. Each transverse block already splits
into a constraint pair with speeds±1 and a gauge pair with speeds±sqrt(mu),
including mu1; each tensor block has speeds±1. This supplies the complete20
basis algebraically. It does not establish a uniform diagonalizer as lapse or
chi tends to zero or mu/f tends to infinity.

## A direct reference-preserving row definition

Let W=W_gauge and cW=1-W, distinct from the geometric compactification weight.
For positive alpha,chi define A0=alpha²chi and a dimensionless fixed G0>0:

    B=cW G0+W A0,
    mu=B/A0,
    f=1+2cW/alpha,
    kappa=B/(A0+B).

Starting from the full physical-reference wave-map rows G_R, define only

    Delta alpha_t = -2cW alpha(P-P_hat)/Omega,

    Delta beta_t^i = (B-A0)(Lambda^i-Lambda_hat^i)
      +alpha²(2kappa²-1/2) gtildeInv^{ij}
                         [chi_j-chi chi_hat_j/chi_hat]
      -cW eta_I(beta^i-beta_hat^i).

This leaves the harmonic lapse-gradient coefficient epsilon_alpha=1 unchanged
and changes the chi-gradient coefficient from1/2 to2kappa². It must start from
the **complete** coupled wave-map rows, retaining all connection/Z terms and
the analytic reference contractions. It is not a lapse-only or Gamma-only
replacement. Additions vanish identically at the stationary Minkowski reference.
At W1,B=A0,kappa1/2,cW0, every addition is exactly zero and the original outer
physical-reference wave-map condition is retained.

The displayed direct expressions never need to form B/(alpha²chi) in an actual
gauge implementation. kappa has positive denominator, lies strictly between0
and1, and epsilon_chi is bounded between0 and2. The physical chi-gradient
coefficient is alpha²*epsilon_chi, not a constant times grad(log chi). This
avoids the1/r force that the simpler unbounded choice epsilon_chi=mu/2 would
produce in an isotropic trumpet with constant B. Exact branch W1 must short
circuit the inner definition at scri; cW support has positive Omega for the
current gauge radii. Robust finite arithmetic still needs future tiny-value
and high-contrast tests; these pencil identities are not such a test.

All coefficients are dimensionless except eta_I,P,Lambda,chi_j, which have
inverse-length units. The lapse correction has inverse-length units, as do
the shift rates. G0 can initially be compared at the old3q0/4=3/8 value, but
the actual q is now the rational expression above, not the old q0. Parameter
names must distinguish these. No legacy validator may silently approve the
new coupled family.

## Puncture/trumpet restrictions remain genuine gates

In the exact geometric core and on Einstein data, the lapse becomes modified
BM alpha_t=beta.grad(alpha)-alpha(alpha+2)K_phys. Off that sector it retains
P=K_phys-2Theta_phys. Set the logarithmic lapse restoration to zero for the
existing regular stationary power-law calibration. The core shift is

    beta_t^i=beta.grad(beta^i)+G0 Lambda^i
       -alpha chi gtildeInv^{ij}alpha_j
       +2alpha² kappa² gtildeInv^{ij}chi_j-eta_I beta^i.

For a controlled radial expansion alpha~a0 r^p,p>0,chi~c0 r²,beta~v x,
bounded nondegenerate conformal metric and differentiated remainders, the two
gradient terms are O(r^(2p+1))=o(r), kappa tends to1. The constant connection
response needs Lambda=O(r), or cancellation of any stronger geometric
connection residues, if beta is to remain O(r). Bounded metric components
alone do not prove this. An isotropic trumpet Lambda0 is admissible at leading
order and still needs eta_I=v>0; necessity is not sufficiency. If instead
Lambda=lambda1 x+o(r), the leading balance is

    G0 lambda1+v²-eta_I v=0.

lambda1 must come from the actual geometric contracted connection on Einstein
data, not an independently assigned Z-violating Lambda. Full stationary radial
coordinate/metric equations and all subleading terms remain unsolved. This
source choice therefore removes a known principal obstruction and supplies a
nonvanishing conformal connection response; it does not prove formation or a
stationary trumpet.

Relative to advection, transverse and longitudinal gauge coordinate speeds
approach sqrt(G0) and sqrt(4G0/3), respectively. Lapse and light speeds still
collapse. Nonzero gauge speeds near the removed puncture require an actual
origin/inner characteristic and numerical regularity treatment. They cannot
be called automatically outflow. Finite-positive-lapse completeness does not
certify well-posedness at alpha=chi0.

An inner-only change leaves the conditional stationary mass-log obstruction
of the unchanged outer physical-reference wave-map equation intact. Later BH
work still needs a derived live-field/asymptotic outer extension, independent
mass-corrected wormhole data, resolved core scales and the actual authorized
wormhole-to-trumpet evolution with the Minkowski reference. No BH RHS subtraction
or hidden stationary BH reference is proposed.

## Curvature-radius attribution is a separate control

For fixed S and r in the CMC collar,
Omega=(S²-r²)/(2aS),alpha_hat=(S²+r²)/(2aS),beta_hat=-r/a and P_hat=-3/a.
Increasing a reduces these coordinate curvature/speed coefficients, but it
also changes physical R=r/Omega, pulse proper scales, minimum Omega and the
dimensionless damping K=kap_input*a²/S. It is not a clean stability diagnosis.
The exact Cauchy core remains Omega1, so the layer is not globally self-similar
under a simple time/length scaling.

A useful source-only preliminary comparison would keep the same r-grid/span,
geometry/gauge cutoffs and fractional native perturbation definition, and bind
both C0 and wave-map controls at each a. Separately distinguish fixed kap_input
from K-matched kap_input(a)=kap_input(a_ref)*(a_ref/a)². Record closest-cell
Omega, actualdt, all reference derivatives and normalized local/source/operator
rates before selecting any private native run. Compare equal characteristic
crossing fractions, not only a fixed coordinate endtime; include the same
dt-cap and half-cap control at each a. A changed cutoff or compactification
needs its own exact reference, complete principal, pole and constraint-source
gates. Reduced outer-shell rates followed by spatial/time refinement would
make an a-change informative; a later abort at fixed coordinates alone would
only show parameter sensitivity. No such calculation/run is admitted here.
