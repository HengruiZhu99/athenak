# Circumbinary diagnostics, schema 2

Base: `origin/circumbinary-cyclic-zoom`, f48f3126c1096d4df0d3f72909f9075969c4efa6
(verified against the remote on 2026-11-11). The implementation is isolated from
prescribed binary parameters and the evolution EOS. Diagnostic stress reconstruction
supports ideal gamma-law Valencia MHD with ADM, in three spatial dimensions. Disable
`problem/flux_diagnostics` for other configurations. Torque spatial output supports
one, two, and three dimensions with the evolution's nghost=2/3/4 derivative dispatch.

## Definitions and compatibility

Primitive velocity is W v^i; bcc is sqrt(gamma) B^i. Stored momentum is sqrt(gamma) S_i;
stored IEN is sqrt(gamma)(E-D), so conserved Eulerian energy is IEN+IDN. No additional
metric determinant belongs in a conserved volume integral. The generic evolution EOS
and physical configuration restrictions are retained. ValenciaComovingB's b0 means
alpha b^t, and b_u is the Eulerian spatial projection, not coordinate b^i.

The first 17 columns per configured transfer surface retain their ordering and signs:
mdot and edot are inward; pdot and ldot are outward. All are instantaneous fixed-surface
terms, including H1/H2. Legacy ldot is about the global coordinate origin. Compare with
manuscript inward F_P/F_J by NEGATING legacy pdot/ldot. H1/H2 retain their configured
Euclidean radii and identifiers; they are transfer spheres.

Appended `*_in_moving_*` currents already include the Reynolds correction:
F_in = - integral (f^i - q w^i) dA_i, where q,f are undensitized and
`dA_i = sqrt(gamma) dSigma_i`. Do not correct these columns again. Eeu is Eulerian
energy; Et is coordinate energy alpha E-beta.S, with flux
alpha(alpha S^i-beta^i E)-beta^j(alpha W^i_j-beta^i S_j).
Et is a flux diagnostic; no coordinate-energy volume closure is implemented. Fluid/EM
channels sum to total transport; separate closure would also need exchange terms.
Jx/Jy/Jz use the barycentric generators. Intrinsic flux about c follows from
F_J_intrinsic=F_J_global-c cross F_P; it is a finite-radius coordinate diagnostic,
not a physical black-hole spin-growth measurement.

Phi_B stays half-unsigned Heaviside-Lorentz flux, without a Reynolds correction.
The manuscript's sqrt(4 pi) conversion and distinct per-hole/fixed-total rate
normalizations remain in analysis. Modeled-horizon rates do not replace moving-5M
transfer-sphere denominators. Manuscript equations and consumers were inspected:
`cbd.tex`, `figures/source/magnetic_flux.py`, `current_budget.py`, and the continuation
extraction/analysis chain. Historical statistical screens are not a validity policy.

## Surfaces and validity

`problem/flux_modeled_horizons=true` adds `BH1_Kerr` and, when m2>0, `BH2_Kerr`.
These are modeled component Kerr level sets, not located apparent/event horizons.
Q=(I+a a^T/r_plus^2)/(r_plus^2+a^2), H=L^T Q L, a=m chi,
r_plus=m(1+sqrt(1-chi^2)); L is the metric's equal-time inverse boost. Radius along
n is (n.H.n)^(-1/2). Translation occurs before the boost. For the supported stationary
aligned corotating metric the trajectory is frozen at its metric epoch, H and c are
stationary and coordinate w=0 despite nonzero physical boost. In inertial coordinates
H_dot and c_dot follow the metric's trajectory derivatives, including mass, spin,
orientation, and boost changes. At fixed n, w=c_dot-r^3(n.H_dot.n)n/2.
The current diagnostic derivative requires subextremal spins; this does not restrict
the evolution without these optional surfaces. Production area uses the sampled full
simulation metric; isolated Kerr's 4 pi(r_plus^2+a^2) is only an isolated-metric check.

All complete-surface currents and area are NaN if any sample is unavailable. No
renormalization or NaN summation fills missing flux. `partial_M0_net_in_moving` and
`partial_covered_area` are explicitly partial. Ownership and valid-sample fractions
are sample fractions, not fractions of proper area. The unmasked quadrature is kept
separate from local ownership. MPI verifies exactly one owner per point. Validity
reason bits: 1 missing/duplicate ownership or bad stencil; 2 invalid lapse/metric
(Sylvester criterion); 4 invalid fluid/field/current; 8 underresolution; 16 a reset or
excision stencil; 32 frozen zoom support. Local spacing is measured at the owner's
stencil, conservatively doubled at multilevel block boundaries. The default required
minimum radial extent/spacing is 4 for modeled horizons; both threshold and largest
local stencil spacing are reported. Ownership alone is not resolution evidence.
Interpolation buffers and lifecycle coverage are described below as implemented.

## History files

Full labels replace ten-character truncation. Noninteger radii use distinct names.
The complete count is checked before surface allocation against existing capacity.
Native `athena_read.hst_segments` exposes every schema/restart segment and metadata;
`hst` retains earlier rows, unions columns with explicit NaN for unavailable older
fields, and retains the existing restart-branch pruning convention. Ambiguous
historically truncated duplicate names gain positional `__colN` suffixes and warn.
No new-width rows are appended without a fresh native header.

## Accepted-step mass, state time, and restart

Only selected modeled Kerr surfaces are sampled at every pre-stage and at final
qualification. Other transfer surfaces stay at history cadence. The scalar trial
integral follows the **actual** fixed-u1 MHD recurrence `a=gam0*a+beta*dt*rate`, starting
at zero each step. For RK2 this gives dt*(f1+f2)/2, not 1.5 dt. Distinct stages with
identical metric times are distinct samples. This implementation supports rk1/rk2/rk3;
the branch's MHD CopyCons does not implement the delta recurrence required by its rk4
coefficient table. Qualified integration rejects other integrators explicitly.

Only accepted intervals with all required stage and endpoint samples valid contribute
to `M0_qualified_cumulative_BH*_Kerr`. Coverage/unavailable durations count dt once per
accepted interval. A missing interval contributes no measured mass and remains in the
unavailable duration; no gap is interpolated. Histories and pre-checkpoint export are
read-only with respect to mass. A committed cycle/time prevents duplicate commits.
This is rest mass crossing a modeled component worldtube; it is neither gravitational
mass growth nor removal by an excision/reset operator.

The existing prescribed-ADM stage timing is unchanged. `diagnostic_state_time` is the
accepted state/output time; `diagnostic_metric_geometry_time` is the last actual ADM
setter time. Geometry and all related currents use that metric epoch, including the
legacy fixed-surface terms at that instantaneous diagnostic location. For stationary
ADM the metric epoch can remain zero. Pre-stage fluid states follow the integrator,
but inertial time-dependent geometry is phase-frozen over each evolution step. This
is only first-order temporal sampling for a geometry-only time-dependent current.
The RK weights alone do not confer high-order accuracy on this mixed-time evolution.
The manufactured recurrence tests separately verify RK2/RK3 quadrature of synchronized
analytic state rates and first-order sampling of phase-frozen geometry. No evolution
arrays or metric timing are modified to improve diagnostics.

The existing restart ParameterInput header stores schema, surface/configuration identity,
epoch start, committed mass/time/cycle, durations and last interval flags. No raw zoom
layout changes are made. State restores only for a genuine restart. Old checkpoints
start an explicit new epoch at restart time; they cannot recover previous accretion.
Changed geometry, thresholds or frame configuration requires `problem/flux_new_epoch=true`
to deliberately discard the incompatible epoch; that request is consumed once. No
quadrature endpoint is carried between visits. Geometry/ownership rebuild from the
restored mesh before each measurement. Pending checkpoint-cycle zoom events continue
to use the existing driver restoration path.
