# Native composite integration audit

This is a source audit of the isolated implementation at `f37942825`, not a
native evolution result. No native mode was enabled, no production guard was
removed, and no evolution or MPI job was launched for this audit. The existing
field-layout adapter has component evidence; it is not a stage/task interface.
The findings below identify implementable work, not a mathematical obstruction.

## Existing hooks and what they actually do

| Source location | Current action | Required composite-mode action |
| --- | --- | --- |
| `z4c_tasks.cpp:42–66`, `z4c_calcrhs.cpp:33–37` | `CopyU` precedes ordinary `CalcRHS`; the legacy Cartesian hyperboloidal pointer has an early RHS return | Dispatch one composite bulk+shell RHS owner after the appropriate stage state is ready; do not run either ordinary Cauchy RHS or the legacy cut-sphere RHS |
| `z4c_tasks.cpp:156–186` | `CopyU` first calls the six-shell `Extension::BeginStage`, then updates native `u1` | Split stage bookkeeping from global RHS evaluation; composite shell state must use the same stage recurrence as the bulk and scri atlas |
| `extension.cpp:314–328` | Existing `BeginStage` fills ghosts, copies stage arrays, calls shell `CopyU`, shell Cauchy `CalcRHS<3>`, then finite-radius Sommerfeld RHS | Do not reuse this path for the hyperboloidal solver; call the global original-vector solve and full conformal tensor/lapse RHS once for all patches |
| `z4c_update.cpp:22–43` | All25 native fields get the low-storage RK mixture, then the extension separately updates its shell | Evolve all authoritative volume and surface state with the chosen common RK method; beta is algebraic stage data and needs special handling |
| `z4c_Sbc.cpp:127–128`, `extension.cpp:291–313` | Ordinary boundary RHS is skipped for existing extensions; the extension supplies finite-radius Sommerfeld | Composite mode supplies the tensor scri limit and independent normal/fixed-angle vector data; no existing Sommerfeld routine applies |
| `z4c_tasks.cpp:276–319` | Native Cartesian restriction, exchange and prolongation precede `ApplyPhysicalBCs`; extension fills coarse/bulk/shell donors | Retain native interior AMR communication where consistent, then fill all composite seams/corners and actual scri ghosts from the common composite state |
| `z4c_tasks.cpp:212–228`, `z4c.cpp:336–381`, `z4c_adm.cpp:218` | Algebraic projection, shell projection and ordinary/legacy physical ADM conversion | Explicitly select projection policy for this formulation and supply mapped Penrose-to-physical ADM conversion for every active bulk/shell cell |
| `z4c_newdt.cpp:34–44,97`, `extension.cpp:342` | Legacy cut-sphere pole CFL or ordinary/finite-radius speed estimates | Use the actual mapped transport, physical-log lapse, reference-vector flux relaxation and scri SAT bounds across both packs |

The current legacy modes cannot be combined by setting two input booleans:
`z4c_hyperboloidal.cpp:31–48` admits only one uniform Serial block, no AMR,
ng=3 and RK3; `extension.cpp:36–59` rejects that pointer, nontrivial MPI,
telegraph and restart, and admits only the regular cubed-sphere test pgen.
These are deliberate implementation guards. Keep them for the existing modes;
introduce a separate composite owner/dispatch instead of broadly deleting them.

## Field conventions and two concrete integration hazards

`Z4c::nz4c` is25 (`z4c.hpp:61–71`). The new adapter uses the actual enums;
the full physical trace is native `I_Z4C_KHAT=K_phys-2 Theta_phys` with the
conformal curvature/chi/lapse conventions already documented by
`athenak_bridge.hpp`. Its22 coefficient fields omit the three algebraic beta
components. Lambda stays in `I_Z4C_GAMX..GAMZ`; it is never assigned the
metric-derived connection V.

**Flux semantics.** Native `I_Z4C_BX..BZ` are already lapse-flux storage, not a
Gamma-driver auxiliary. They are aliased as `vB_d` in `z4c.cpp:116,126`.
The Cauchy telegraph RHS (`z4c_calcrhs.cpp:710–724`) treats them as a covector,
adds `chi g_tilde^{ij} partial_i B_j` to the Cauchy lapse, and drives them with
the gradient of the total lapse. The selected hyperboloidal extension instead
uses reference-Cartesian vector components, the Euclidean reference divergence
`chi partial_i(w B^i)` and the gradient of `alpha-A_ref`. Equality of component
numbers under the *flat reference* is not lowering an index with the evolved
metric. Reusing the slots is feasible, but running the Cauchy gauge kernel in
addition would change the requested equations. The adapter comment has been
corrected to make this explicit; no functional adapter code changed.

**Algebraic beta and RK registers.** `NativeCompositeFields::Write(...,rate)`
sets the beta RHS to zero. That does not keep beta equal to its elliptic solution
under native RK: `ExpRKUpdate` mixes `u0` with `u1` for all25 slots, and `CopyU`
also accumulates old beta values in `u1` for RK4. The composite mode must exclude
beta from that dynamical recurrence or overwrite it from an accepted-state
solve after the recurrence and solve it again before every tensor RHS. Stale
beta in an unused RK buffer must never become a new gauge evolution. Restart
and diagnostics must clearly identify which accepted geometry its beta solves.

## Native RK4 is not the pilot RK4

The native driver (`driver/driver.cpp:143–174`) implements Ketcheson's four-stage
RK4()4[2S], whereas `Pilot::Step` uses classical four-stage RK4. Native stages use

    stage1: u1 = u_start
    later stages: u1 += delta[s] u0
    u0 = gam0[s] u0 + gam1[s] u1 + beta[s] dt F(u0).

For a constant unit RHS, the exact stage abscissae derived from these actual
coefficient arrays are

| RHS stage | c |
| --- | ---: |
|1|0|
|2|1.193743905974738|
|3|0.4314013217808047|
|4|0.9999999999999964|

The final c is0.9999999999999996 at the stored precision. Existing extension
times `{0,1,.5}` are SSPRK3-specific and must not be copied into a new RK4 path.
Even for stationary Omega and clock, any evolving surface/AF coefficients must
use the same recurrence as the corresponding volume stage.

There is a second distinct choice. The adapter converts a chart rate into an
unfactored physical rate correctly, but applying native RK to physical fields
is not the same discrete method as applying RK to logarithmic/chart fields.
Both can approximate the same PDE; they need not be bitwise equal. To reproduce
the existing pilot's discrete chart evolution, give the composite owner its own
authoritative coefficient RK registers and publish physical `u0` mirrors to
native packs. Alternatively retain physical native RK, document that numerical
choice, and establish independent time convergence/positivity controls. Neither
choice changes the Minkowski reference gauge. Do not claim equivalence of the
finite-step methods from the field-layout rate test alone.

## Physical ADM and diagnostics need a separate branch

Regular `Z4cToADM` interprets native chi/alpha as ordinary Cauchy fields. Penrose
chi/alpha cannot be passed through that conversion. The existing legacy
`HyperboloidalADM` (`z4c_hyperboloidal.cpp:169–205`) illustrates the correct
`ToPhysicalADM(u,Omega)` call and enforces separate ADM gauge storage, but it
iterates one Cartesian cut-sphere block. A composite branch must use mapped x
and Omega for all active cells and the actual Cartesian-component tensors.
No physical ADM conversion is defined for the separate Omega=0 surface atlas;
active cell diagnostics remain included, while surface diagnostics are the
conformal/null/clock/pole quantities. Exterior ghost points also cannot be
treated as physical Omega>0 cells.

Regular ADM constraints, Weyl/extraction and tracker tasks queued at stage end
are likewise not automatically the correct conformal diagnostics. Route the
new mode to the tested conformal constraint kernel on mapped jets; retain
existing extraction/trackers disabled until patch-aware versions are supplied.
Native `AlgConstr` rescales the metric and removes A's trace. It does not reset
Lambda, but it is a numerical projection absent from the present pilot. Either
bypass it in the free-evolution comparison mode or validate/document the
selected determinant-convention enforcement; do not silently introduce it.

## Shell allocation, restart and regrid ownership

The legacy extension allocates exactly six blocks through a temporary Cartesian
tree (`extension.cpp:98–139`) and then replaces their geometric interpretation.
The new descriptor already supports arbitrary shell layers/face subdivisions.
A storage-only shell pack can still be allocated independently of the central
AMR tree, but every ordinary routine that would consume its fake Cartesian
neighbors, coordinates or Cauchy physics must be bypassed. Generate maps,
patch identities, block count and physical ownership from `CompositeMesh`, not
from loop bounds `0..5`. The shell's internal Z4c construction must not
recursively create another global composite owner.

Native restart currently counts/writes only `pm->pmb_pack->pz4c->u0`
(`outputs/restart.cpp:56–126`) and reads only that primary pack
(`pgen/pgen.cpp:245–265,550–585`). A shell pack and evolved scri atlas are not
covered merely because they also contain25 fields. Add a versioned composite
section with descriptor/block IDs, all shell fields, all surface fields, chart
and gauge controls, source binding and shift warm start. Reuse the research
checkpoint's explicit validation concepts, but it is not itself native restart.
The read path must allocate the described shell before restoring it, must not
reinitialize restored surface data from analytic initial data, and must rebuild
halos/hierarchies from the restored descriptors. The current pgen restart guards
at `pgen/tests/z4c_cubed_sphere.cpp:39` and `extension.cpp:25` stay until both
writer and reader exist. Preserve old native restart layouts for modes without
the new section. A cycle-boundary restart needs accepted state and controls,
not stale stage scratch; a claimed mid-stage restart would need all RK buffers.

Native AMR does transfer all25 primary-pack fields: counts at
`mesh_refinement.cpp:116`, same-rank copy/restrict/prolongate at541–623.
It also destroys/recreates primary `MeshBlock` and `Coordinates` objects around
line660. Thus raw block indices, cached host mirrors and pack geometry
references cannot survive regrid. The sequence counter is incremented at160;
`Mesh::GetAMRLoadBalanceUpdateSeq()` is the existing invalidation key used by
the legacy donor plans. On a sequence change, rebuild the central leaf
descriptor from live bounds and refinement levels, regenerate the union
bulk/shell map, global interpolation rows, coefficient coarsening, MG hierarchy
and field adapter. Do not preserve an adapter holding a reference to a deleted
grid. Reconstruct coefficients from the transferred physical fields, or use an
explicit chart-consistent transfer if chart buffers are authoritative. Surface
state remains independent when only the interior refines. Re-solve beta on the
accepted new mesh before consuming it. Native primary AMR is useful plumbing;
it does not currently notify or transfer the standalone shell/surface state.

## Minimal implementable sequence and falsifiable gates

1. Introduce one explicit composite owner on the primary Z4c object. Factor the
   existing stage physics out of the test `Pilot` into a reusable stage operator;
   it must consume geometry/state and return original-equation residuals and
   rates without owning an entire time loop. Keep legacy pointers distinct.
2. Allocate shell packs from `CompositeMesh`; establish a checked union mapping
   covering every active composite block exactly once. The present adapter
   checks duplicate entries within a call, not completeness across all packs.
   Fill ghosts through the common active-only plans and scri atlas; its current
   `Write` intentionally writes only active cells and does not provide halos.
3. Insert a single global stage task after stage-state preparation and before
   any physical update. Gather all packs, fill globally, solve the coupled vector
   equation, publish stage beta, evaluate full tensor/gauge/flux and scri rates,
   then update volume/surface with a coherent chosen RK recurrence. Disable
   Cauchy RHS/boundary/ADM paths only for this explicit mode.
4. Add mapped timestep/diagnostics, then native cycle-boundary restart and
   sequence-based regrid rebuilding before claiming those input combinations.
   Serial host staging is a concrete first native integration milestone; it
   must be described as such, not as a distributed/device multigrid port.
5. Tests before nonlinear duration extension: a constant-ODE stage trace checks
   the actual native recurrence and abscissae; a nonlinear chart ODE checks the
   chosen authoritative variables; off-constraint Lambda and nonzero beta/flux
   catch unwanted projection/driver work; one/two-shell equilibrium and pulse
   stage comparisons include actual face state; a nonzero surface checkpoint
   restarts exactly; a refine/coarsen/regrid leaf touching the shell preserves
   all active fields, invalidates plans and re-solves beta. Keep all physical
   cells in acceptance norms and report interfaces/scri/core additionally.

The targeted transition/interface/scri numerical diagnostics being developed
in parallel remain the scientific gate. None of this audit justifies relaxing
their tolerances or advancing failing wormhole pilots to late time.

## Prepared coefficient stage-state component

`native_composite_stage.hpp` now provides `NativeCompositeStageState`; its
standalone audit is `tst/elliptic_shift_3d/test_native_stage.cpp`. These files
prepare the missing stage ownership without wiring or enabling a native mode.
They were prepared without running a CPU build alongside the root pilot.

`NativeCompositeRK::FromDriver` reads the live native Driver coefficient arrays.
The test independently parses the actual assignments in `driver.cpp`, rather
than maintaining another hidden RK table, and checks RK3/RK4 phase abscissae,
a coupled nontrivial linear ODE across volume/scri/extra endpoints, the
counterfactual native beta-register contamination, and bitwise mid-stage
register continuation. The test source argument may explicitly name
`src/driver/driver.cpp`; otherwise it locates that file relative to `__FILE__`.
No claim that these prepared tests pass is made until the root builds/runs them.

The component has no hard-coded field count, puncture exponent, origin or
beta-at-puncture condition. The authoritative coefficient groups are volume,
scri and arbitrary extra endpoints. Opaque chart metadata records a name,
parameters and mesh/chart/puncture revisions. Changing a chart's representation
requires explicitly transformed coefficients through `ReplaceAcceptedState`;
metadata cannot silently reinterpret live RK registers. Position/geometry
revision changes invalidate beta, and AMR replacement is restricted to accepted
cycle boundaries. All beta values live in a separate warm-start vector that is
never part of an RK sum.

The future task dispatch is explicit:

1. At the accepted cycle boundary, pull native physical pack state into the
   selected coefficients, or retain already-authoritative coefficient state;
   on a genuine regrid use `ReplaceAcceptedState` with transferred fields and
   descriptor metadata. Call `BeginCycle(FromDriver(driver),dt,time)` once.
2. At native `CopyU` scheduling time call `BeginStage(stage)` on the composite
   coefficient owner, instead of using the physical native `u1` recurrence for
   this mode. Publish the current coefficients as physical pack mirrors.
3. Build current mapped geometry, signal any geometry/position revision, then
   fetch `CurrentEpoch`. The global elliptic solve installs beta with that exact
   epoch. `TensorRHSState(epoch)` rejects an old geometry/stage beta.
4. Evaluate all full tensor/lapse/flux, scri and AF endpoint rates, then call
   `AdvanceStage(epoch,rates)` once. It advances every coefficient group with
   the actual native recurrence, invalidates beta, and publishes physical
   mirrors. The ordinary native25-field physical RK update must not also run.
5. At any required accepted-state diagnostic, solve again for the accepted
   epoch before consuming beta. A saved component snapshot contains all
   coefficient and RK registers plus time, phase, metadata and warm start.
   Restore checks method binding and retains register bits but deliberately
   invalidates the old solved-shift epoch after hierarchy reconstruction.

This is not native restart I/O: the eventual native checkpoint section must
serialize/validate the component snapshot and geometry descriptor. No sidecar
payload, task hook or shell allocation was added by this component preparation.

## Actual puncture tracker interface and later motion

`CompactObjectTracker` exposes `GetPos()`/`GetPos(a)` and `SetPos`, but its
velocity array is private and there is no velocity getter/setter. The black-hole
branch of `InterpolateVelocity` (`compact_object_tracker.cpp:100–114`) samples
native beta with `LagrangeInterpolator` and sets velocity to **minus beta**.
This agrees with the coordinate puncture sign; it does not pin beta to zero.
The existing interpolator derives a Cartesian cell stencil from the native
pack, not the composite map.

`TrackCompactObjects` (`z4c_tasks.cpp:333–344`) executes only at the final RK
stage and calls `EvolveTracker`. Its ODE mode uses one forward-Euler
`pos += dt*vel`; its walk mode searches a Cartesian cell-centered lapse minimum.
Neither is a stage-accurate position law for a moving singular coefficient chart.
The native restart records tracker positions (`restart.cpp:267–269`) but not
the private velocity or any position RK registers. The MPI tracker reduces
position/velocity over owning ranks, so ownership must be recomputed after
movement and regrid.

A standalone `PuncturePosition` abstraction should own global Cartesian
position, stage velocity and revision, with a stage law `Xdot=-beta(X)` from the
converged composite Cartesian-component shift. Include its stage coordinates
in the generic endpoint state or an equally synchronized RK register bundle;
its explicit chart time derivatives then use the same stage velocity. In the
first fixed-puncture radial comparison, the position can remain explicitly
fixed while nonzero diagnostic beta is still measurable. For actual motion,
publish accepted positions to `CompactObjectTracker::SetPos` for existing
output/refinement consumers, and bypass the legacy Euler/walk update to avoid
moving twice. Velocity output needs a small explicit accessor or a dedicated
composite position report; it cannot be installed through the present public
tracker API. A translated fractional manufactured test can verify the position
abstraction and changing chart powers without claiming a physical translating
black-hole evolution. No tracker or original experiment was altered here.
