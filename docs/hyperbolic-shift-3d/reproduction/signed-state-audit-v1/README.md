# Telegraph PR and sign-stop audit

The user identified the open telegrapher-gauge PR and requested that negative
lapse/chi be recorded rather than treated as a stopping criterion on the way
to20M. Reference: IAS-Astrophysics/athenak PR790, open head
9db7169c9ea0ec943caf286b2199676e138ae9ee on HengruiZhu99/athenak
project/z4c_overhaul (fetched2026-10-30).

## Verified reference behavior

src/z4c/z4c_update.cpp uses direct linear RK updates for all stored fields.
In z4c_calcrhs.cpp lines704-717, the opt-in telegraph gauge adds
 gamma^{ij} partial_i B_j to the lapse RHS and evolves
 B_i,t = lapse_advect*beta^j partial_j B_i + (kappa partial_i alpha-B_i)/tau.
The contraction uses the physical inverse metric. It is a partial divergence,
not a covariant divergence. z4c_newdt.cpp lines53-72 adds physical-metric
characteristic and tau relaxation timestep limits.

The default chi_psi_power is-4, chi_div_floor=-1000 and floor_chi=false.
Thus the default division guard is not a positive chi floor; the optional
post-RK chi floor is separate. There is no direct lapse floor in the inspected
Z4c source. This PR is closer to the present experiment than main's gauge,
but it is not the same transformed hyperboloidal PDE: the present gauge also
contains conformal/reference terms, radial attenuation and lapse-dependent
telegraph coupling. PR validation claims do not establish20M stability here.

## Present run and mathematical checks

Job8990333 uses direct lapse RK/KO, log-chi RK/KO and floors. A finite negative
lapse candidate is floored before the geometry check; the chi variable cannot
represent signed values. The preceding CF1 run failed on positive-infinite
lapse at9.156M, not a negative-value stop.

Simply deleting the end-step sign predicate does not admit signed evolution:
- BuildJet and HyperbolicGauge separately reject nonpositive lapse/chi.
- ConformalRHS requires positive chi and lapse, and forms g_ij/chi before
  checking that metric. Negative chi violates that positive metric check.
- The KO chart and RK use log(chi), undefined over the reals for negative chi.
- Stationary conformal-normal reconstruction divides by lapse, including at
  the puncture; zero lapse is a true arithmetic singularity of that expression.
- Scri null reconstruction and characteristic diagnostics have their own
  geometry/real-speed requirements.

There are two distinct experiments: retain floors and count negative candidates
before clipping, or store signed lapse/chi with an explicitly defined extension
of the RHS/diagnostics through unphysical states. The latter requires changing
variable representation and deciding how to evaluate singular/invalid quantities.
The user clarified: keep floors and log negative candidates. No signed-state
PDE extension was made. See ../candidate-signs-v1 for the implemented diagnostics
and the subsequent instrumented20M replay.
