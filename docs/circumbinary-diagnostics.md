# Circumbinary diagnostics, schema 2

Base: `origin/circumbinary-cyclic-zoom`, f48f3126c1096d4df0d3f72909f9075969c4efa6
(verified against the remote on 2026-11-11). The implementation is isolated from
prescribed binary parameters and the evolution EOS. Diagnostic stress reconstruction
supports ideal gamma-law Valencia MHD (mhd/eos=ideal and dyn_eos=ideal) with ADM,
in three spatial dimensions. Other evolution EOS configurations remain available
with diagnostics disabled; their generic evolution thermodynamics are unchanged. Disable
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
Stationary extremal geometry is supported. An undefined extremal spin-change normal
speed remains unavailable (nonfinite motion), rather than being silently regularized.
Modeled horizons require prescribed ADM, dynamic for inertial binaries. Production area uses the sampled full
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

The appended surface families are `M0_net_in_moving`, `M0_in_moving`,
`M0_out_moving`, `Eeu_{f,em}_in_moving`,
`{Px,Py,Pz}_{f,em}_in_moving`, `{Jx,Jy,Jz}_{f,em}_in_moving`, and
`Et_{f,em}_in_moving`, each suffixed by its surface identifier. Qualification fields
are `valid`, `reason`, `ownership_fraction`, `valid_sample_fraction`,
`partial_covered_area`, `stencil_dx_max`, `required_cells_per_radius`, and
`partial_M0_net_in_moving`, with the same suffix. Only explicitly partial transport/area fields retain partial-integral values when
a whole surface is invalid; qualification fields describe that invalidity.

## Accepted-step mass, state time, and restart

Only selected modeled Kerr surfaces are sampled at every pre-stage and at final
qualification. Other transfer surfaces stay at history cadence. The scalar trial
integral follows the **actual** fixed-u1 MHD recurrence `a=gam0*a+beta*dt*rate`, starting
at zero each step. For RK2 this gives dt*(f1+f2)/2, not 1.5 dt. Distinct stages with
identical metric times are distinct samples. This implementation supports rk1/rk2/rk3;
the branch's MHD CopyCons does not implement the delta recurrence required by its rk4
coefficient table. Qualified integration rejects other integrators explicitly.

Only accepted intervals with all required stage and endpoint samples valid contribute
to `M0_qualified_cumulative_BH*_Kerr`. Coverage/unavailable durations count the committed mesh-clock increment once per
accepted interval (avoiding independent dt-sum drift at a large absolute epoch). A missing interval contributes no measured mass and remains in the
unavailable duration; no gap is interpolated. Histories and pre-checkpoint export are
read-only with respect to mass. A committed cycle/time prevents duplicate commits.
This is rest mass crossing a modeled component worldtube; it is neither gravitational
mass growth nor removal by an excision/reset operator.

The existing prescribed-ADM stage timing is unchanged. `diagnostic_state_time` is the
accepted state/output time; `diagnostic_metric_time` is the last actual ADM setter time. Histories retain the
legacy output-time locations: `diagnostic_geometry_time` equals the accepted output
time, and both fixed and moving channels use the same instantaneous geometry. This
preserves existing origin/lever-arm analysis. The field/metric timing mismatch is
explicit; no evolution array is refreshed to conceal it. At integration cadence the
auxiliary mass and budget quadrature follows the actual stored ADM epoch at each
sample. For stationary ADM that epoch can remain zero.

Pre-stage fluid states follow the integrator, while the first stage can still see
the previous step's metric epoch. The evolution stage then refreshes prescribed ADM
at mesh time, so subsequent stages see the current step's mesh epoch, rather than
their RK state abscissae. A uniform-mesh manufactured geometry-only rate f(t)=t gives
RK2 error 1/N-1/(2N^2) over unit time: first order. The helper test reproduces this
actual setter ordering separately from synchronized analytic-state RK2/RK3 quadrature.
An actual inertial executable case checks state, geometry and stored-metric clocks.
No evolution arrays or metric timing are modified to improve diagnostics. Existing
restart initialization can refresh an inertial background at the current mesh time;
restart trajectory equivalence for that mixed-time evolution is not established here.
The matched restart/zoom regressions use the supported stationary background.

The existing restart ParameterInput header stores schema, surface/configuration identity,
epoch start, committed mass/time/cycle, durations and last interval flags. A stable
FNV-1a identity covers geometry/configuration and all trajectory-table values. Region
state is a versioned, comma-separated list of 20 full-precision scalars; unavailable
entries are explicit strings. This keeps the existing 40-KiB parameter-header reader
limit intact. State format is 2, history schema is 2. No raw zoom
layout changes are made. State restores only for a genuine restart. Old checkpoints
start an explicit new epoch at restart time; they cannot recover previous accretion.
Changed geometry, thresholds or frame configuration requires `problem/flux_new_epoch=true`
to deliberately discard the incompatible epoch; that request is consumed once. No
quadrature endpoint rate is carried between visits. Geometry/ownership rebuild from the
restored mesh before each measurement. Pending checkpoint-cycle zoom events continue
to use the existing driver restoration path.


## Regional budgets and actual mutations (unit C)

`problem/flux_budget=true` selects `budget_r_inner`, `budget_r_outer` among the
existing R surfaces, and the existing H1/H2 transfer spheres. Defaults are 20 and 400;
set them to the actual configured boundaries. H spheres must be disjoint and wholly
inside Rinner, the whole outer sphere must fit in the domain, and each actual puncture
excision ellipsoid must lie within its H sphere. This implementation deliberately
supports **stationary prescribed corotating geometry without radiation/Z4c**, with
fixed puncture boundaries. It does not enable new physical configurations. Extending
regional closure to inertial/deforming regions, evolving spacetime, or a coupled
radiation sector is remaining scope; surface transport and qualified rest mass have
the broader prescribed-ADM coverage described above.

The disjoint partition is near1 = H1 minus actual excision X1, near2 = H2 minus X2,
cavity = Rinner minus H1/H2, annulus = Router minus Rinner. Single-hole configurations
omit near2. There are no shared volumes; cavity+annulus is the manuscript exterior
volume, and all regions together are the gas volume inside Router excluding excision.
Excision boundaries are modeled using the exact coordinate puncture Kerr-radius mask
geometry, not the optional modeled Kerr horizons. Frozen zoom support is unavailable.
Fractional Cartesian cells use `budget_subcells` equal subdivisions per direction
(default 2, allowed 1..8); the same classifier supplies region fractions and Cartesian
first moments for storage and every mutation measurement. Torque uses those moments
and the same full Valencia helper as spatial torque output.

Jz = sum (first_moment_x u[IM2] - first_moment_y u[IM1]) dV_coordinate,
without a second sqrt(gamma). Complete budget boundary inflows are

* near1: F_J_in(H1) - F_J_in(X1), and similarly for near2;
* cavity: F_J_in(Rinner) - F_J_in(H1) - F_J_in(H2);
* annulus: F_J_in(Router) - F_J_in(Rinner).

All fluxes use barycentric z angular momentum and all actual boundaries. The same
qualified surface sampler checks the excision boundaries, including local resolution.
If their reset/underresolution stencils are invalid, near-hole residuals are unavailable;
they are never closed by omitting the inner boundary. Instantaneous storage, full
coordinate torque including F^x_y-F^y_x, and elapsed-step changes are recorded.

The reported residual is delta J minus the integrated boundary inflow, continuum
coordinate torque and other measured conserved changes. It is a continuum geometric
budget, not exact finite-volume cut-cell closure. Interpolated surfaces, fractional
volumes and the evolution numerical flux are different discretizations. A residual
is not called a physical source. Only total fluid-plus-EM closure is provided.
Coordinate-energy outputs remain flux-only, without a time-metric/Q_t closure source.

Nonoverlapping before/after region **scalar reductions**, rather than a new full-mesh
snapshot, measure the actual mutations at these operations:

| Channel | Actual operation |
|---|---|
| recovery_reset | committed primitive recovery, floors/ceilings, excision/reset and its boundary conversion; an honest aggregate |
| physical | enabled MHD source operator and user cooling/coupling operator |
| coordinate_rG | actual coordinate momentum update, recorded separately; already represented in continuum torque, so never added twice |
| nonfinite_repair | active-cell conserved repair; an unquantifiable event if pre-change momentum is nonfinite |
| remap_jump | ordinary remapping with its immediate recovery, including its angular first-moment change |
| zoom_jump | storage correction/restoration/masking and subsequent recovery; nested recovery is suppressed |
| initial_recovery_jump | zero-duration startup/restart boundary/recovery/preconditioning, separate from accepted-step sources |

Only active evolved cells contribute. Ghost repairs and FOFC trials do not. Flux
restriction, FOFC and flux repair remain part of the numerical flux and are not also
counted as volume sources. Magnetic damping changes the field through CT; any resulting
conserved recovery change belongs to the recovery aggregate. Direct attribution of
separate suboperations within that aggregate is not claimed. Stage mutations are
propagated by subsequent gam0 coefficients, rather than raw-summed. Source MPI sums
occur once in driver order, outside retryable tasks. Already global histories and
cumulative states contribute on rank zero. Cumulative source/jump columns are explicitly
**quantified partial** sums, with unavailable-event counts; no finite correction is
invented for a nonfinite pre-state or unavailable region. A restart while a region is
frozen adds a separate unavailable startup-recovery measurement event; event counts
therefore need not be restart-invariant. This never changes a quantified jump or the
qualified mass ledger.

One necessary expansion of restart scope was found by the zoom regression. A final
checkpoint just after zoom had already selected the next dt; restart recovery and
recomputation could enlarge it (0.24132089 to 0.24156908 in the small test), shifting
which intervals were unavailable. Startup now preserves the stored post-zoom dt cap
while retaining any smaller fresh stability/end-time limit. Scheduled pre-event
checkpoints still complete the existing pending-AMR path. The evolution update and
prescribed metric stage timing are unchanged. Both sides now reproduce uninterrupted
qualified mass and coverage within the regression tolerance.

## Python consumers and practical use

`disk_diagnostics.py` and `post_process_cbd.py` honor supported corotating, single-hole,
trajectory-table and spin-ramp definitions. Eulerian Wv is rotated as a spatial vector;
it never has grid rotation subtracted from it. Files lacking frame metadata require
`--historical-frame inertial` (or the explicitly justified corotating interpretation).
Per-dump caches and stitched products carry frame, rotation epoch and a geometry /
sampling identity. Incompatible or ambiguous caches require explicit re-extraction;
old manuscript numerical results and archived vendor kernels are not rewritten.

The shared `scripts/cbd_history.py` adapter is also included in the manuscript worktree
under `analysis/native_diagnostics`. Select `mode='moving'` explicitly for corrected
columns; supplying another Reynolds correction raises an error. Legacy fixed or
legacy-plus-explicit-Reynolds modes remain separate. Missing samples stay NaN. Keep
restart segments/epochs distinct and do not use a temporal outlier screen to fill
unavailable coverage. The older snapshot disk-structure statistics are not a replacement
for the new native whole-surface validity or accepted-step mass records.

Interpolation result buffers and proper covectors are reused. Geometry, ownership,
metric sampling and support qualification rebuild at each requested evaluation, so
motion, mass/spin changes, refinement, redistribution, restart and zoom restoration
cannot leave an ownership/metric cache stale. No evolved values are cached across
stages. Use a sufficiently precise history format, e.g. `%24.16e`, for budget work.
Angular sampling is bounded at 16,777,216 points per surface. The exact requested
history size is checked against the existing 2000-slot capacity before allocation:
44 per surface, 8 per present horizon accumulator, 3 time columns, and 22 per region.


Current reset provenance also survives restart. The pre-checkpoint hook saves a
compact whole-surface reset/excision-support flag for each configured surface and
budget boundary. It only qualifies support; it does not integrate the other surfaces'
currents or advance mass. A restored flag conservatively invalidates the whole surface
until the first evolved stage replaces that state. This avoids pretending that loss
of a transient per-cell repair marker made a restart interval clean. Repeated
checkpoints before evolution retain the inherited flag. Evolved variables and the raw
zoom-state layout remain untouched. A manufactured checkpoint-provenance injection
test covers this path separately from tests of actual evolved fields.
