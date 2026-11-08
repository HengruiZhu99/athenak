# Planned native hyperboloidal diagnostics integration

Status: **implemented, validation in progress** on 2026-11-08. This document
preserves the preimplementation contract and its acceptance gates. The native
coordinator, history integration, sparse sampler, and FastFlow extension now
exist. Current evidence is recorded separately in `history-output.md`,
`fastflow-external.md`, and the corresponding evidence files; listing a gate
below does not claim that it has passed. No Aurora diagnostic or performance
result is established. This work does not change the adopted
[N12 comparison protocol](comparison-protocol.md).

The implementation must use the original native `Driver`, output cadence,
MeshBlock ownership, `MappedExchange`, and FastFlow solver. There must be no
alternate evolution driver, globally replicated evolution field, or whole-state
gather to rank zero. Global leaf metadata and small scalar/harmonic/surface
reductions are compatible with this design.

## Reference observable and source locations

The target and provenance limitations are recorded in
[N12-reference.md](N12-reference.md). N12 uses a fixed horizon center at the
origin and **disabled puncture tracking**. Do not import N24's tracker settings.
Coordinate time, horizon lengths, and horizon masses are multiplied by three
when reported in units of M. The target is step 3551, coordinate time
1.6669886746517166, with the unchanged nominal timestep
0.00046944203735617445.

Native source references, relative to the repository root:

| File | Relevant existing entry point or limitation |
| --- | --- |
| `src/driver/driver.cpp` | `Driver::Initialize`, `Execute`, and `Finalize` call existing output `LoadOutputData`/`WriteOutputFile`; regular accepted-step output currently precedes AMR. |
| `src/outputs/history.cpp` | `HistoryOutput` registration and dispatch currently recognize ordinary `pz4c`; `WriteOutputFile` reduces rank-local scalars with `MPI_Reduce`. |
| `src/outputs/outputs.hpp` | `HistoryData`, the history loader declarations, and `NHISTORY_VARIABLES=20`. |
| `src/athena.hpp` | `PhysicsModule` output dispatch identifiers. |
| `src/z4c_hyperboloidal/z4c_hyperboloidal_lifecycle.cpp` | `InitializeBoundaryValues` already exchanges residuals and samples/solves scri, but also enforces algebraic constraints and projects owned surface fields. It is not a read-only diagnostic preparation call. |
| `src/z4c_hyperboloidal/z4c_hyperboloidal.cpp` | `FormResidual`, accepted field/reference storage, and `EnforceAlgebraicConstraints`. |
| `src/z4c_hyperboloidal/tasks.hpp` | The native RK task dependencies; the final update/project occurs after the last RHS derivative evaluation. |
| `src/z4c_hyperboloidal/point_state.hpp` | `BuildPointState` and `RestorePointState`, including reference-subtracted residual derivatives and chi/r-squared restoration. |
| `src/z4c_hyperboloidal/z4c_hyperboloidal_rhs.cpp` | Anonymous `EndpointState` currently assembles endpoint jets; its unchanged numerical assembly should be shared with diagnostics. |
| `src/z4c_hyperboloidal/surface_exchange.hpp` and `.cpp` | `Sample`, `Solve`, `Kernel::ResidualJet`, and `Kernel::NormalCorrection`; these provide endpoint reconstruction without requiring `Evaluate` or a full bulk RHS. |
| `src/z4c_hyperboloidal/spherical_policy.hpp` | `SphericalPolicy::Physical` converts stored field conventions into the twice-conformal `Z4cJet`; this alone is not physical ADM conversion. |
| `src/z4c_hyperboloidal/conformal_rhs.hpp` | Existing `EvolvedConstraints` evaluates the physical constraints directly in evolved variables. |
| `src/z4c_hyperboloidal/conformal_constraints.hpp` | Metric geometry, stationary Omega normal completion, and physical/conformal Z4 constraint contractions. |
| `src/z4c_hyperboloidal/mapped_halo.hpp` and `.cpp` | `NativeDonorSelector`, sparse native donor metadata, and the evolution halo's intentionally restricted stencil set. |
| `src/bvals/mapped_exchange.hpp` and `.cpp` | Persistent peer transport with posted receives, packed local values, incomplete receive polling, and retirement. |
| `src/z4c_hyperboloidal/dissipation_plan.cpp` | Existing Jacobian volume calculation and physical-inner-start half weight for the outer nodal layer. |
| `src/z4c/fastflow.hpp` and `.cpp` | Existing native FastFlow flow, angular quadrature, integrals, and output; this checkout does not yet contain the external geometry-sampler interface below. |
| `src/z4c/z4c_tasks.cpp` | Ordinary Z4c queues `FindHorizon` after ADM conversion/tracking and only runs it on the final RK stage. Hyperboloidal diagnostics must supply their own accepted-state geometry instead of calling the Cartesian derivative/interpolation path. |
| `src/coordinates/adm.hpp` and `.cpp` | Ordinary ADM storage uses nominal Cartesian block layout. It should not be populated with a fictitious Cartesian representation of the mapped atlas. |

The predecessor checkout is read-only at
`/Users/hz0693/.codex/worktrees/scri-ko01/hyperboloidal`.
Exact source paths under that root are:

| Predecessor file | Reference behavior |
| --- | --- |
| `src/mesh/cubed_sphere/composite_difference.hpp` | `CompositeDifference::DiagnosticVolume` defines unique-coverage norm weights; constructor defines mapped cell volumes and the outer nodal half weight. |
| `src/z4c/hyperboloidal/hyperbolic_composite_device.hpp` | `Diagnose::operator()` and `Diagnostics` assemble jets, call `EvolvedConstraints`, and accumulate full/exterior squared constraints. |
| `tst/hyperbolic_shift_3d/endpoint_policy/fastflow_history.hpp` | `EndpointHorizonSampler::Node`/`Sample` define six-point residual interpolation, derivative weights, restoration, and physical geometry sampling. `EndpointFastFlow` gives N12 settings. Its root-only complete-state wrapper must not be ported. |
| `tst/hyperbolic_shift_3d/endpoint_policy/fastflow_diagnostics.cpp` | Saved-state replay, full harmonic exterior mask, scalar reductions, RMS normalization, and replay gate. Its legacy evolution/complete-state scaffolding must not be ported. |
| `src/z4c/horizon_geometry.hpp` | `HorizonGeometry` and `ValidHorizonGeometry`: physical Cartesian metric, extrinsic curvature, and metric first derivatives. |
| `src/z4c/hyperboloidal/horizon_geometry.hpp` | `PhysicalHorizonGeometry`: conversion from twice-conformal evolved jets to physical ADM jets. |
| `src/z4c/fastflow.hpp` and `.cpp` | Already implemented `GeometrySampler`, `BatchGeometrySampler`, `SampleGeometry`, `Result`, and unique/complete sample ownership validation. |
| `src/z4c/fastflow_surface.hpp` | Full harmonic `FastFlowSurface`, `Radius`, `Outside`, and conservative bounds for avoiding unnecessary harmonic evaluations. |

The production checkout's current sources are not automatically the exact
historical N12 diagnostic binary. Preserve the source/build provenance caveats
in `N12-reference.md` and its evidence files when constructing an oracle.

## Constraint and volume semantics

For each region, report

```
RMS(C) = sqrt(sum(weight * C_squared) / sum(weight))
```

The four squared quantities are H squared, the **physical-metric contraction**
of the momentum covector, physical Theta squared, and the physical-metric
contraction of Z. Use `EvolvedConstraints` and its `z4` result; do not replace
physical M/Z contractions with their Penrose-conformal counterparts. The
regularized formulas avoid dividing by Omega in the constraint evaluation.

The reference quadrature weight is the compact-coordinate map Jacobian volume,
not proper ADM volume. In native map conventions its basic value is
`1/(abs(det(J))*N^3)`, where `J` maps Cartesian derivatives into logical ones.
The outer nodal layer retains its half weight only at the physical inner
endpoint; newly introduced internal radial starts have full weight.

Independently evolved extra cells have zero diagnostic quadrature weight.
On nominal cells, the spherical core/shell blend uses

```
t = clamp((r - inner_radius)/(half_width - inner_radius), 0, 1)
shell_weight = t^3 * (10 - 15*t + 6*t^2)
core_weight  = 1 - shell_weight
```

Multiply the map volume by the appropriate blend weight. Thus both nominal
charts cover the blend without counting the full volume twice. Do not reuse
ordinary `LoadZ4cHistoryData`'s `sqrt(det(physical_metric))` factor or its
chi-excision criterion: either would change the N12 observable. Invalid values
and geometry must still be counted on every evolved cell, including zero-weight
extra cells.

The exterior region is defined by the **complete harmonic FastFlow surface**
and `FastFlowSurface::Outside`, including its center. It is not a mean-radius
cut or a spherical approximation. A failed/incomplete horizon must be reported
explicitly; a missing surface must not silently become an all-exterior mask.
The fixed outer-radius region used by the old replay is a separate validation
observable and must not be mislabeled as horizon-exterior.

## Minimal proposed interfaces and data flow

Names below are proposals, not existing APIs.

1. Add a native diagnostics implementation, for example
   `src/z4c_hyperboloidal/diagnostics.hpp/.cpp`, owned by the module. A
   `RefreshDiagnosticWorkspace()` method recomputes residuals and runs existing
   bulk/surface exchange, `Sample`, and `Solve` through native task polling.
   It does not call `EnforceAlgebraicConstraints`, `Project`, surface `Evaluate`,
   bulk RHS, KO, or RK update. Stage four already establishes the accepted
   projected state; diagnostics must not change it to conceal differences.
   Writes to difference ghosts and diagnostic workspace are allowed; owned
   bulk/scri values, accumulators, and ChiRecords remain unchanged.
2. Cache readiness by accepted cycle **and** native mesh-update sequence, with
   explicit invalidation on initialization, restart, remesh, and accepted
   state changes. The last RHS workspace is not automatically current after
   the final RK update. Repeated output consumers at the same accepted state
   should reuse the refreshed workspace. Diagnostic solves must not overwrite
   or masquerade as the 32 true stage-diagnostic observations in an eight-step
   oracle.
3. Share the current point/endpoint jet assembly without changing its numerical
   order. A local constraint reduction returns two records of the form
   `ConstraintSums { volume, h2, m2_physical, theta2, z2_physical }`, plus
   validity/worst-location diagnostics. This requires jets, not the complete
   right-hand side. Keep the original scri reconstruction and normal lift.
4. Add a hyperboloidal history registration/loader and an unambiguous output
   physics identifier. Let the existing `HistoryOutput::WriteOutputFile`
   reduce the ten raw sums. Only then convert the eight squared sums to RMS;
   never sum rank-local RMS values. Keep the two volumes in the output. The
   existing 20-variable capacity is sufficient for these ten entries. Existing
   FastFlow summary/shape output can carry horizon results separately, keyed by
   accepted time/cycle, rather than placing replicated horizon scalars in a
   sum reduction.
5. Invoke preparation/finding/reduction at the actual native diagnostic/output
   cadence, including initial and final output. The normal output loader is a
   suitable synchronous entry point; any NR task adapter must use the same
   accepted-state preparation and cadence. Do not run expensive horizon searches
   at every RK stage. Preserve original Driver scheduling rather than adding an
   evolution loop. Regular output currently occurs before AMR; a later consumer
   after remeshing must invalidate the topology-dependent cache.
6. Extend the existing FastFlow class with the predecessor's small external
   sampler/result/surface interface, retaining the ordinary Cartesian path.
   Provide a native batched callback with the existing signature shape
   `SampleHorizonGeometry(count, positions, geometry, owner_flags)`.
   Each trial point has exactly one native owner. Missing owners, multiple
   owners, or an invalid owned jet fail the complete search. Reuse FastFlow's
   flow, quadrature, integrals, convergence, and file writing.

## Sparse horizon sampling and physical ADM geometry

Preserve the reference sampler's six-point tensor interpolation of
reference-subtracted residuals, including chi/r-squared normalization. At each
trial point, choose the same geometric containing leaf, build its 6-by-6-by-6
local sample lattice and first-derivative weights, interpolate the residual
jets, restore the analytic reference at the query point, and reconstruct the
dependent algebraic components. Do not interpolate physical ADM tensors
directly; that is a different nonlinear operation.

The evolution `MappedHalo` fills the axis/mixed derivative and KO stencil
union. It intentionally does **not** fill every triple-corner ghost needed by
a general 6-cubed tensor sampler. Reading those padding cells would be a hidden
dependency, even when a smooth initial condition happens to leave plausible
values there.

Construct an explicit sparse sampling plan for the required lattice nodes.
Use an evolved local node directly; resolve a ghost node through the existing
`NativeDonorSelector` and request only its evolved donors with `MappedExchange`.
Preserve the old two-level interpolation/restoration and anchored accumulation
conventions. A dedicated tensor-halo staging plan is also possible if it uses
the same rows and is shown equivalent. Cache immutable donor/node data for an
accepted state and invalidate it on state/topology changes. Trial-surface
changes can reuse unchanged stencil support, but weights must follow the
current query. No global field download is required.

The predecessor physical geometry helper uses, at finite interior points,

```
gamma_ij = gtilde_ij / (chi * Omega^2)
K_ij     = Atilde_ij / (chi * Omega)
           + gamma_ij * (P + 2*Theta) / 3
d_a gamma_ij = [d_a gtilde_ij
                - gtilde_ij * (d_a chi / chi + 2*d_a Omega / Omega)]
               / (chi * Omega^2)
```

Here `P` is the full physical trace-minus-twice-Theta after the stored reference
offset is restored. Include both symmetric tensor copies and Cartesian
derivative indices. Require positive finite chi/Omega and valid physical metric
geometry. Scri is excluded from physical ADM horizon sampling; do not introduce
an Omega floor. This finite-interior conversion is distinct from regularized
constraint evaluation near scri.

## N12 configuration and comparison targets

Use fixed origin, tracking disabled, `lmax=12`, `ntheta=24`, at most 300 flow
iterations, alpha-beta constant `.1`, initial radius `.02`, mass tolerance
`1e-10`, and expansion RMS tolerance `1e-3`. Preserve the recorded search time
range and accepted-state accounting. Only surface samples, harmonic
coefficients, ownership counts, and scalar integrals may be collectively
communicated by the finder.

The preserved step-3551 values are:

| Observable | Reference |
| --- | ---: |
| Full H RMS | 0.0073257648025702423 |
| Full M RMS | 0.0039666959397409406 |
| Full Theta RMS | 0.00063679744011741466 |
| Full Z RMS | 0.00074930798279592369 |
| Horizon-exterior H RMS | 0.0066650464364187327 |
| Horizon-exterior M RMS | 0.003785788884853934 |
| Horizon-exterior Theta RMS | 0.00062887813299441033 |
| Horizon-exterior Z RMS | 0.0007488764374473728 |
| Compact mean horizon radius | 0.030549496850187822 |
| Areal horizon radius/M | 1.9999860425879357 |
| Irreducible horizon mass/M | 0.99999302129396783 |
| Expansion RMS | 1.4887607237254066e-5 |

The preserved history is
`docs/z4c_hyperboloidal/evidence/n12-fastflow.csv`. These final values do not
replace comparison of all available common times through the target.

## Acceptance gates to execute before admission

These are planned checks. No result is claimed by listing them.

1. Compare native and independent legacy pointwise constraint values and
   physical ADM jets on exactly matched initial, nonspherical, and saved N12
   data. Cover core/shell interfaces, mixed levels, final radial reconstruction
   rows, and query stencils crossing faces/owners. Require bitwise equality
   when arithmetic order is preserved; otherwise use the adopted short
   arithmetic gate `1e-12 + 5e-10*abs(reference)`, reporting maxima, per-component
   errors, weighted RMS, worst coordinates, and complete coverage. Keep the
   separate chi guard wherever chi itself is compared.
2. Poison unused padding, tensor corners, and workspace before preparation and
   sample construction. Verify all required data is populated and unused data
   is not read. Check every owned bulk/scri field, RK accumulator, and ChiRecord
   bytewise before/after diagnostics; diagnostics must create no floor events.
   Exercise repeated calls and cache invalidation after restart and remesh.
3. Compare every nominal quadrature weight and overlap mask to legacy, including
   the smooth blend and physical outer-start half weights. Compare full and
   exterior scalar sums/RMS on 2 and 8 ranks, including ranks with no horizon
   samples. Validate pointwise finite/geometry checks even on zero-weight
   evolved extras. Add analytic map-volume and constant-integrand controls.
4. Replay exactly the same saved checkpoints using the unchanged replay gate
   `5e-11*(1+abs(expected))` for the original full/fixed-outer RMS observables.
   Separately compare the complete harmonic exterior mask/results; do not
   confuse a fixed-radius reference with a horizon-exterior reference.
5. At every common time through step 3551, apply the adopted native-history gate
   `abs(delta) <= 5e-10 + 1e-6*abs(reference)` to all eight full/exterior RMS
   histories. Report roundoff-sized initial diagnostics separately. Preserve
   successful and complete horizon status, and use the adopted horizon
   radius/mass relative tolerance `1e-7` with absolute floor `1e-10` in M units.
   Record expansion, iterations, full coefficients/center, and ownership;
   these do not replace complete-state agreement.
6. Force missing/duplicate horizon ownership and invalid owned geometry, and
   require collective failure of the complete search without deadlock. Check
   fixed-origin N12 settings explicitly. Regress the existing ordinary
   Cartesian FastFlow path after introducing the optional sampler interface.
7. Preserve the existing normal-compatibility bound `1e-7` and actual stage
   history. Compare clock diagnostics to the reference; do not impose zero
   clock. Diagnostic evaluations must not be counted as additional RK stages.
8. Record source/helper/object/executable hashes, independent reference
   provenance, decomposition, complete coverage, and unchanged sources through
   each run. Test 2 and 8 ranks independently. Report diagnostic startup,
   sampling, search, reduction, and I/O cost separately from evolution timing;
   correctness runs do not establish a speedup.

Fixed-N12 replay establishes preservation of that discretization. Accuracy and
stability of adaptive meshes remain subject to the separate AMR transfer,
interpolation, and coupled-regrid contracts; a successful horizon or small
global norm alone does not establish them.
