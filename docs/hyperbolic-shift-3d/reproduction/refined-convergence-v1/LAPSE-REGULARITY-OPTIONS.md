# Lapse normalization and candidate remedies

The plotted evolved lapse is baralpha=Omega*alpha_phys. In the current core
Omega=.42 for r<=.4, so baralpha=.35 means physical lapse .8333. The deliberate
precollapsed puncture lapse and its gauge evolution are separate from this
normalization. Plot both lapses in the finite interior when diagnosing collapse.

For the stationary outer reference, f=1-2M/R, a_s=sqrt(f+R²S²),
baralpha_star=Omega*a_s and beta_star^r=-Sr. Before the hyperboloidal tilt,
alpha_phys=a_s is close to one away from the hole, while declining Omega makes
baralpha small. As the slice tilts, a_s grows and baralpha rises to its scri
value one. The areal20–40M switch is compressed into a compact-coordinate
feature: peak abs(dbaralpha/dr)=29.48, halfmax width .02137. This feature
already exists in exact stationary data with vanishing continuum constraints;
it does not establish dynamically singular slicing. Finite nonzero baralpha
at scri requires alpha_phys~1/Omega, so physical lapse near one all the way to
scri is not the appropriate target.

Large derivatives can amplify discretization/cancellation error in the lapse
Hessian, curvature and conformal-factor terms. Those couplings are required
parts of Einstein evolution; deleting them or clamping lapse to one is not a
consistent remedy. New short boosted errors concentrate near the mesh transition
.95, beyond the original gradient peak. Interface attribution must accompany
the slicing investigation.

## Proposed controls, not implemented

1. Smooth the exact slicing/map and compatible reference gauge together.
   COMPACT-SLICING-STUDY.md gives fixed-central-scale candidates with gradient
   peaks18.30/13.53. Physical slicing endpoints move outward. Native boosted
   height derivatives, sources, conformal frame and scri must all be qualified.
2. Evolve a normalized lapse a=baralpha/baralpha_star or a perturbation, with
   analytic reference derivatives. For time-independent reference and
   L=partial_t-beta.grad, the exact transformed equation is
   L a=(L baralpha)/baralpha_star+a*beta.grad(log baralpha_star).
   This is a variable change, not a stability proof. A regular puncture chart
   is needed wherever the reference vanishes. Consistent chi/metric/shift
   perturbations may reduce cancellation if all product-rule terms remain.
3. Consider a well-balanced spatial operator
   F_h(q)-F_h(q_star)+F_cont(q_star), subtracting partial_t q_star as well when
   evolving a perturbation. Set F_cont(q_star)=0 only for a genuine stationary
   solution of the complete selected gauge. The inner precollapsed wormhole
   with1+log gauge is not automatically stationary, and a stationary spherical
   reference is not the exact evolving boosted solution. Blind subtraction
   would change the problem. Retain full independent constraint diagnostics
   and perturbation dynamics in any future control.

The current prescribed gauge source cancels continuum stationary outer gauge
imbalance; it does not subtract the full discrete geometric RHS. Setting core
Omega to one only renormalizes conformal variables unless the complete joining
construction changes consistently; it does not itself cure physical gradients.
No production equations have changed in this study.

## Meaning of D_x

D_x is the Cartesian x component of auxiliary shift field D^i, not a derivative
operator or constraint. Its equation is L D^i=L Lambda^i-D^i. It feeds the outer
shift through mu*D^i, mu=1.5*chi*(baralpha²+4*Omega²*baralpha). The inner plateau
instead uses direct L beta^i=k² Lambda^i-6 beta^i. A large D alone does not prove
physical pathology: examine mu*D, other shift terms and their cancellation,
shift gradients, and independent constraints.
