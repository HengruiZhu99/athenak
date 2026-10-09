# Held exact-flat control for the actual native angular gauge pulse

This is a source-only proposal. No new numerical, CAS, compiler, kernel, matrix,
scalar propagation, native evolution or payload readback has been performed.
All execution and implementation remain held for root review. Existing accepted
and failed experiments are read-only dependencies.

The first decisive control should use a differentiated retarded Kirchhoff
integral on the complete initial layered Minkowski hyperboloid, reduced by
coarea to compact initial radius and source azimuth. The old native
Cartesian scalar operator is useful as a separate discrete comparison. Its
failure alone could not establish a continuum coordinate caustic because it
shares the spherical ghost continuation under investigation.

## Question and exact initial data

Does the physical-reference harmonic coordinate map for the *same finite native
angular pulse* stay locally invertible, globally usable on the requested target
region, and compatible with spacelike native time slices through the native
failure events and t_native=2? This separates a possible exact-flat coordinate
limitation from the off-constraint Z4c/discretization failure. It does not prove
lower-order Z4c stability or choose a later black-hole inner gauge.

Use exactly S=1, a=.5 and the retained geometry cutoff (.05,.95). The reference
conformal lapse is h, boost b, physical radius R=r/Omega, and
L=Omega-r*Omega_r. The complete reference height is H(R), with H=0 in the core,
H_R=b/h and an outer branch sqrt(R^2+a^2)+constant. Do not replace the layered
height by a pure-CMC height through the transition. Native initial computational
coordinates coincide with the reference coordinates on the initial slice.

The authoritative initializer is src/z4c/z4c_hyperboloidal.cpp, function
InitializeHyperboloidal. Its large pulse is

```
f=(1-r^2)^4 exp(-r^2/.35^2),
deltaalpha=.2*f*(1+.2*x+.3*y*z),
deltabeta=.1*f*(1+.3*y*z, .2*x, .1*x*y),
alpha=h+deltaalpha, beta=betahat+deltabeta.
```

This has no compact sub-scri cutoff. Only in the exact outer a=.5 branch is
f=Omega^4 exp(-r^2/.35^2). The proposed continuum control uses this analytic
profile on the whole initial slice, not a grid interpolant, freely chosen wave
or old manufactured dipole. The input pin is the actual N24 large t2 input;
the pulse formula itself is independent of N. The reference geometry, A/P and
connection initially remain exactly the Minkowski reference, Theta=Z=0.

Follow the independently reviewed IVP note in the inventory. For four fixed
inertial scalar components Y^A=X^A+u^A, impose

```
Box_eta u^A=0, u^A|Sigma=0,
s^A=n^B partial_B u^A
   =(h/alpha-1)n^A-(Omega/alpha)deltabeta^i E_i^A,
E_i^A=partial_i Xhat^A,
J^A_B|Sigma=delta^A_B-s^A*n_B, detJ=h/alpha>0.
```

For robust initial-data evaluation use the factored components in the IVP
note, rather than subtracting nearly equal reciprocal lapses. The initial
conformal scalar fields phi=u/Omega have phi=0 and
Pi=nbar(phi)=s/Omega^2. Equivalently phi_tau=h*s/Omega^2. Generic radial/time
components vanish as O(Omega) initially. Do not impose zero boundary values on
later outgoing radiation. The rational denominator alpha produces infinitely
many angular harmonics at finite amplitude.

## What can actually be reused

| Artifact | Reusable evidence/component | New work still required |
|---|---|---|
| coupled-wave-isolate, frozen exporter/source snapshot | Actual LayerReference coefficient conversion, regular Rbar expression, native Dx/Dxx/Dxy/Lx/KO, exact ray and MLS donor substitution, scalar operator and point/gradient ordering; existing N16/N24 span2.2 matrices | Actual .2/.1 initial normal data, four inertial columns, inverse-map derivatives and target-time reconstruction; old dipole oracle is inapplicable |
| common-rho dense-mass scalar model | Regular-origin solid harmonics, dense mass, Jacobi congruence, exact polynomial integration-by-parts test pattern | Full hyperboloidal variable coefficients, scalar potential, characteristic/SAT interfaces and exact-scri treatment; it is not an admitted hyperboloidal solver |
| radial_sbp.hpp | A checked outgoing shell transport/SAT identity | A second-order coupled scalar equation, origin, angular sectors and incoming field; it cannot serve as this IVP oracle |
| held wave-map J0 radial sources | Physical/gauge source conventions only | They have no admitted full-angular scalar solver and finite-rb incoming SAT is not transparent exterior data |

No existing artifact read here supplies an independently validated complete
boundary-fitted hyperboloidal conformal scalar evolution. Building one is a
possible later cross-check, not a prerequisite that should be hidden inside the
first experiment. A scalar energy estimate alone does not control the inverse
coordinate Jacobian.

## Primary oracle: compact radial/azimuth integral, no finite angular truncation

KIRCHHOFF-PENCIL.md independently checks the root's coarea proposal as well as
the null-ray representation. For p=(T,X), Re=|X|>0, put
u_ret=T-Re and D_h(q)=H(q)-q. The allowed source radii have endpoints

```
lower: D_h(q)=u_ret if u_ret<0; H(q)+q=u_ret if u_ret>=0,
upper: H(q)+q=2*Re+u_ret.
```

Define w=u_ret-D_h(q) and
mu=1-w/q+w/Re-w^2/(2*Re*q), the source-direction cosine with the event axis.
The retarded kernel's distance denominator cancels in coarea. With compact
source radius r, q=r/Omega(r), and compact event radius r_e=Omega_e*Re,

```
u^A(p)=(1/(4pi*Re)) integral q*sqrt(1-H_q^2)
                        [integral_0^(2pi) s^A(q,nu(mu,az)) d az] dq,
phi^A(p)=u^A(p)/Omega_e
        =(1/(4pi*r_e)) integral [r*L/(h*Omega^2)]
                        [integral_0^(2pi) s^A(r,nu(mu,az)) d az] dr.
```

For reference events u_ret=D_h(Re)+tau, so it is evaluated without subtracting
large H and Re. Compute D_h from its bounded derivative
dD_h/dr=-L/[h*(h+b)], with exact core D_h=-r and exact outer
D_h=C+a^2/(sqrt(q^2+a^2)+q). Evaluate initial native pulse data with factored
outer (1-r^2)^4/Omega^3=(2a)^4*Omega at S=1. This removes large-height and
normal-data cancellation without changing the IVP. The native initial
integrand is O(Omega), not a prescribed later falloff.

At Re=0 use the separate exact center reduction: q_star solves H(q)+q=T and
u=q_star*sqrt(1-H_q^2)/(1+H_q)*sphere_average(s). An arbitrary switch near the
center needs a derived analytic limit and overlap comparison, not division by
small Re. For higher derivatives there, use regular Cartesian symmetry/Taylor
moments or the ray formula. Split integration at the known reference layer
breakpoints and at any required source geometry charts.

The null-ray integral is an independent parameterization control:
q=p-lambda*(1,omega), with weight
lambda*s(q)*sqrt(1-H_q^2)/[1-H_q*nu_q.dot(omega)]. A *fixed* boosted frame can
resolve the lab-frame angular cap; the weight becomes ell*s(q)/[-n(q).k] for
k=B(1,omega). Hold B constant during local jet differentiation. Changing the
integration coordinates does not rotate the fixed target scalar components.

All four fields and their first/second inertial derivatives share the same
geometric endpoints and quadrature nodes. Differentiate endpoint roots and
the transformed integral analytically/with ordinary Taylor jets, retaining
every Leibniz endpoint term and complete initial-data/reference derivative.
The azimuthal average is smooth at mu=+/-1, while a naive differentiated
sqrt(1-mu^2) parameterization is not: pair azimuths/derive its even moments or
use a regular endpoint chart, and gate the limit independently. Do not clip
mu to [-1,1] to conceal an endpoint/root error. No Cartesian finite differences
of a noisy integral may silently replace exact derivative construction.
First derivatives suffice for J and target-time
causality. Second derivatives are a useful independent wave residual and later
metric/K reconstruction gate, not a reason to invent missing reference jets.

The bounded height defect D_h needs only a compact transition quadrature:
core D_h=-r, integrate its bounded derivative, then use the exact outer
primitive and matched constant. The old 8192-panel/Hermite height-I table is not an error
certificate for this purpose. Derive analytic height and data derivatives from
the cutoff/reference formulas, and bound value quadrature separately.

This avoids a numerical radial outer boundary, l cutoff and time integration.
It does not automatically provide a uniform exact-scri error bound: D can be
small in the lab frame, source intersections can be far away, and the r->1
limit requires separate weighted/tail estimates. Restrict first conclusions to
finite physical query events and stated compact target subdomains. The
hyperboloidal slice is not a global Cauchy surface for all of Minkowski; the
retarded construction is used only in its future domain of dependence. Check
the domain explicitly and retain endpoint/flux terms in the derivation.

## Proposed staged implementation and gates, not released

1. **Source/math gates.** Independently review the Green-function sign, null-ray
   coarea weight/bounds, fixed-boost transformation, unique-root/domain proof, complete
   height derivative capsule and native initial-data round trip. Save exact
   source/recipe/runtime/dependency pins before any first query. Zero amplitude,
   zero shift, isolated lapse/shift and the exact original combined pulse are
   controls; the combined .2/.1 case is the scientific target.
2. **Integral oracle gates.** Use flat T=0 constant/affine velocity controls and
   the pure-CMC zero-Dirichlet l=0,1,2 exact waves displayed in the pencil. Test
   first and second inertial derivatives, initial s and detJ, wave residual,
   coarea versus lab/fixed-boost rays, independent radial/azimuth quadratures and
   root/height precision. No native target inversion is admitted until these
   pass. Preserve every failed attempt and do not relax tolerances.
3. **Efficient local target screen.** Evaluate the full inverse map at the
   actual native failure events listed below, their saved-time predecessors,
   and a fixed sample set of core/transition/collar and nonsymmetric angular
   directions. This is point/branch evidence only. A large strict margin can
   justify expanding to coverage; it cannot prove global injectivity.
4. **Coverage/derivative control.** Cover the requested native target region
   from t=0 through t=2 by branch continuation from the identity initial map,
   adaptive spacetime boxes and both radial/angle refinement. Bound the field
   and derivative quadrature errors on those boxes. Report min singular value
   of J, detJ, native time-gradient causal margin, inverse residual, future
   orientation, and coverage extent. Stop at a certified obstruction or an
   unresolved numerical/domain limit; no extrapolation beyond the queried
   reference time or physical radius is allowed.
5. **Native scalar discrete comparison.** Only after independent-oracle
   progress, reuse the frozen ray N16/N24 span2.2 scalar matrices and gradient
   ordering, evolving four initial (phi,Pi) columns with a fresh driver. First
   bind the exact data at the actual scalar/native coordinates. Use RK4 with
   the frozen nominal .1*h/max_outgoing and half step, matched reference-time
   outputs, and compare to the integral at the *same physical events*. Optional
   frozen MLS comparison changes only continuation. Do not rerun old drivers:
   they export/overwrite artifacts and initialize the unrelated dipole. No new
   spectrum is needed. The scalar principal system has no gauge pole, so this
   RK4 step is not the native Z4c pole step.

The initial proposed integral implementation uses 80/110-digit independent
arithmetic, radial rules32/64/128 on each smooth source segment and periodic
azimuth rules32/64/128, plus adaptive subdivision and a distinct quadrature
family. The ray control uses nested polar/azimuth rules32x64,64x128,128x256
with cap subdivision or a fixed boost. These are proposal
parameters to be pinned before implementation review, not completed controls.
Use scaled targets 1e-10 for u/J/second-derivative oracle comparisons,
1e-12 for the rescaled inverse residual, and 1e-10 for initial determinant/data
identities. Any observed small determinant/causal margin must exceed the full
propagated error bound by at least a factor 10 before a positive regularity
claim. Failure to meet a target is inconclusive, not a caustic. A posteriori
rule differences alone are convergence evidence; rigorous global exclusion
requires validated quadrature/derivative/box bounds or an independently proved
analytic estimate. Record these two assurance levels separately.

## Target events and time/inversion conventions

Prioritize the *original failed physical-reference wave-map* events:

| Event | native t | native x,y,z | status |
|---|---:|---|---|
| N24 wave-map large | .79277343749968521 | .1375,-.9625,-.2291666666666667 | Original native failure, not a valid saved field |
| N16 wave-map large | 1.3671453700579306 | .20625,-.89375,-.34375 | Original native failure, not a valid saved field |
| N16 C0 large | 1.9994906249988413 | .48125,-.61875,.61875 | C0 comparison event only; its gauge is not the exact-flat RWM IVP |

The last line is a geometric location/time comparison and cannot validate the
C0 gauge. Also retain exact native output times .775 and 1.350028... by reading
the already frozen observation receipts in any later released implementation,
without replacing their full precision by the displayed rounded labels. The
initial proposal targets t_native=0,.02,.1,.25,.5,.75,1,1.25,1.5,1.75,2 plus the
two exact RWM failure times. Final target sampling and spatial coverage must be
pinned in an execution recipe; the failure-event input hashes must be rebound
from the authoritative stopped launch receipts rather than this prose.

For each native target solve the full four-dimensional equation

```
Y^A(X)=Yhat^A(t_native,x_native),
Yhat=(t_native+H(|x_native|/Omega_target),x_native/Omega_target).
```

The reference evolution label tau=X0-H(|XI|) is not t_native. Direct integral
queries avoid a reference-time interpolation cap, but each solved event still
needs tau>=0 and a controlled source-intersection domain. Monitor

```
t_native(X)=Y0-H(|YI|),
c=-eta^AB*(partial_A t_native)*(partial_B t_native)>0,
physical_lapse=c^(-1/2), conformal_lapse=Omega_target*c^(-1/2).
```

Local detJ>0 does not establish global injectivity. A sufficient global route is
a proved small-Lipschitz deformation bound on an explicitly covered convex
physical region; if that fails despite nonzero J, use a proper-map/degree and
boundary/branch argument with validated coverage. Do not assume a Euclidean
norm threshold in compact coordinates supplies either proof. Native spatial
SPD, exact time orientation and target-level-set causality are separate gates.

On a valid branch, the exact physical metric is
E_target^T J^(-T) eta J^(-1) E_target. Its analytically exact Einstein constraints
are zero. A future numerical metric reconstruction still needs second-jet
chain rules and its own residual gate. It does not identify a numerical Z4
constraint error by construction.

## Angular, energy and finite-domain caveats

The primary integral uses the full angular-rational initial data directly, so
there is no finite-l assumption. If a later spectral radial solver is added,
project all four *fixed inertial scalar* components, with lmax4,8,12,16 and
independent doubled angular quadrature as proposed controls. Coefficient and
differentiated-tail errors must converge; four radial scalars or a fixed lmax
are not the actual pulse. No vector-harmonic component-frame approximation is
needed for these four scalar equations.

For zero initial field the physical stationary Killing energy is
(1/2) integral s^2 R^2 dR dOmega: the normal boost h/Omega cancels the slice
volume factor Omega/h. The native s=O(R^-3) tail gives finite initial energy.
This is a useful global flux/tail consistency identity, but energy alone does
not bound first derivatives pointwise or global inverse-map coverage. Derive
commuted energies and Sobolev/trace constants on a stated region before using
an energy as a Jacobian bound. Do not transfer the old sampled conformal
Killing energy's degenerating scri norm to a uniform inertial Jacobian estimate.

If instead using a finite rb<1 radial solver, the initial data outside rb are
nonzero and the ingoing characteristic is nonzero there. A homogeneous
Sommerfeld/SAT boundary is a different IVP. Supply exact exterior incoming
data from the integral, or derive a causal domain where they cannot enter;
finite-radius damping is not transparent by assertion. A whole-scri radial
solver still needs an exact characteristic endpoint and origin regularity
gate. No arbitrary boundary Dirichlet values are part of this proposal.

## Outcome labels

* Converged/validated invertible map and spacelike target coverage through the
  corresponding native failed event exclude an unavoidable exact-flat RWM
  coordinate failure on the stated covered branch/region. They do not prove
  Z4c lower-order stability or attribute the native failure uniquely.
* Converged loss of J invertibility, target-time causality or coverage is an
  exact-flat coordinate obstruction. Report which one fails; similarity in
  native failure time is not a complete causal attribution.
* Scalar native-comparator growth with a regular independent map identifies a
  discrepancy of that discrete control, not a continuum caustic.
* Failed quadrature, unresolved high-frequency angular content, missing global
  injectivity or uncontrolled scri/domain endpoints are explicitly inconclusive.

No outcome admits a black-hole gauge. The later objective remains a single
hole surviving wormhole-to-trumpet evolution with the Minkowski hyperboloidal
reference retained; this global harmonic exact-flat control supplies neither
that inner gauge nor mass-consistent asymptotic closure.
