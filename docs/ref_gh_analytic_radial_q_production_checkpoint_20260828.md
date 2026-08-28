# Ref-GH analytic radial-q production checkpoint (2026-08-28)

## Scope and provenance

This checkpoint starts from
`2d99137ca41de7df12ef1e3234f076b0ef2d8835` on
`codex/ref-gh-analytic-device-view-performance-20260828`.  It retains the
accepted 12-Real static and 8-Real stage radial-q representation and does not
change the radial-q ansatz or the Ref-GH equations.

The generic 1171-Real cache remains an independently dispatched oracle.  The
analytic backend allocates none of `reference_provider`, `reference_workspace`,
`reference_evolution`, or `reference_diagnostic`.

## Completed local gates

- Original deterministic coefficient oracle: 216 samples, maximum error
  `8.88178e-15`.
- Expanded coefficient oracle: 2160 samples over ten radii from `0.03M` to
  `5M`, four fixed off-axis directions, and the unchanged q/qdot/qddot matrix;
  maximum operation-conditioned error `1.48837e-13` against the unchanged
  `2e-13` threshold.
- Generated full-geometry oracle: 2376 samples; maximum conditioned error
  `2.33147e-15` against `256 epsilon`.
- Generated moving-gauge oracle: 2160 samples.  Maximum conditioned errors are
  `Hhat=1.16535e-15`, `dHhat=1.07001e-15`, `theta=7.08259e-16`,
  `dtTheta=3.54866e-17`, and frame motion `1.24829e-14`, all below
  `256 epsilon`.
- Complete 61-component RHS oracle: 4320 comparisons covering compatible and
  standard Phi ordering, gamma0, gamma2, gauge driving, and reference gauge
  subtraction; maximum conditioned error `2.84217e-14` below `256 epsilon`.
- Fresh GCC 13.3 / Kokkos Serial build and the existing q-controlled source
  unit gate pass without weakened tolerances.

The smallest-radius generic gauge contractions are cancellation-conditioned.
The gates therefore propagate the independent generic contraction operation
condition, rather than enlarging any tolerance.  Both raw differences and the
condition scale remain available in the failure telemetry.

## Production integration status

`reference_backend=analytic_radial_q` now allocates exactly 12 static and 8
stage Reals per ghosted cell.  In the 16-cubed one-block local gate this is
`1,327,104 + 884,736` bytes, with `generic_bytes=0`.

The production RHS, ADM reconstruction, constraints, and timestep paths build
compact analytic points from those views.  The covariant scalar source and
moving gauge baseline call generated contracted expressions and never invoke
the monolithic `PopulateGeneratedAnalyticRadialQGeometry`, recursive
`ReferenceSpin`, `ReferenceSpinDerivative`, or `ReferenceRiemann` interfaces.
The monolithic generated geometry remains oracle-only.

A fixed-q stationary-trumpet RK4 cycle completed locally in both dispatches.
At the printed precision both gave:

```
time=2.090302e-02
field Linf=1.548652e-03
physical metric Linf=3.693307e-05
lapse Linf=1.393833e-08
shift Linf=2.797825e-08
constraint Linf=4.984912e-03
```

This is dispatch smoke evidence, not a convergence or performance result.

Disabled fixed-q runs queue no q-measurement task.  Prescribed q evaluates its
trajectory without physical-state sampling.  Closed-loop q owns a compact
precomputed eligible-cell/weight list and uses one device reduction and one MPI
collective.  The former 410-Real workspace estimator is retained only as a
named oracle method and is not called by production dispatch.

## Claim boundary and remaining work

- Analytic coefficient/geometry qualified locally: **yes**.
- Mixed-third-derivative moving-gauge path qualified locally: **yes**.
- All-61-RHS equivalence qualified locally: **yes**.
- Analytic production allocation integrated locally: **yes**.
- Aurora PVC eight-rank/eight-tile evolved dynamic-q cycle: **qualified** by
  job 8789426 at commit `58db23dcb6055f9fc17c10accbe0dde7746f108e`.
  Eight ranks mapped to distinct PVC tiles `0.0` through `3.1`, both the
  one-rank and eight-rank full-output dynamic-q RK cycles completed, and their
  finite Ref-GH histories agreed to conditioned Linf `3.88981e-14` against the
  unchanged `5e-12` gate.  The compact pass record is
  `artifacts/ref_gh_analytic_radial_q_20260828/aurora_pvc_8789426_pass.txt`.

  The preceding compiler-portability sequence is retained for provenance.
  Job 8789242 stopped during PVC code generation before producing an
  executable: IGC segfaulted on the pre-existing monolithic FO-GH RHS unit
  kernel.  The compact failure record and hashes are in
  `artifacts/ref_gh_analytic_radial_q_20260828/aurora_pvc_8789242_failure.txt`.
  That unrelated unit-test kernel has subsequently been split into four
  equation-identical device reductions and both the FO-GH RHS unit and complete
  Ref-GH q-controlled source unit pass locally.  The first corrected rerun,
  job 8789291, compiled all four split FO-GH kernels but exposed a second IGC
  crash in the monolithic Ref-GH source-unit device image.  Its compact record
  is `artifacts/ref_gh_analytic_radial_q_20260828/aurora_pvc_8789291_failure.txt`.
  The oracle pgen is now a default-on validation component that is explicitly
  excluded from the production PVC executable; a default-on oracle build and
  an oracle-disabled analytic production cycle both pass locally.  Job 8789324
  then built that production image and proved eight distinct PVC tile mappings,
  but its one-rank full-output initialization failed at the runtime-dispatched
  RefGhToADM kernel with `UR_RESULT_ERROR_INVALID_KERNEL_NAME`; see
  `artifacts/ref_gh_analytic_radial_q_20260828/aurora_pvc_8789324_failure.txt`.
  RefGhToADM and CalcConstraints now use the same host-side compile-time
  analytic/generic dispatch as CalcRHS.  Job 8789358 showed that the typed
  analytic RefGhToADM kernel itself exceeded IGC's permitted per-thread scratch
  space: the build reported success after dropping SIMD, but the kernel was
  absent at runtime and failed with `UR_RESULT_ERROR_INVALID_KERNEL_NAME`.
  Its compact record is
  `artifacts/ref_gh_analytic_radial_q_20260828/aurora_pvc_8789358_failure.txt`.
  RefGhToADM now evaluates only the individual metric derivatives needed for
  `Gamma^0_ij`, without simultaneously materializing the full coordinate GH
  geometry.  A local full-output RK cycle is bit-for-bit identical to the saved
  pre-refactor histories for all 447 finite entries, at the unchanged `256
  epsilon` gate.  That correction produced the passing job 8789426 above.
- Warmed-up 64-cubed performance: **measured, unacceptable overall**.  Aurora
  job 8789453 measured `1.031481e5` active zone-cycles/s for dynamic-q Ref-GH,
  `1.034160e5` for static Ref-GH, and `7.135298e6` for the matched-grid Z4c
  hardware control.  Initialization-cancelled complete-stage times give
  dynamic/static `1.00165` and Ref-GH/Z4c `65.19`.

  The q reduction is `0.0874%` of a dynamic stage and the compact reference
  update is `0.00672%`, passing their `2%` and `10%` targets.  The directly
  measured Ref-GH/Z4c RHS ratio is `10.06` without dissipation (`9.62` with
  dissipation), failing the stretch target.  The projected physical boundary
  is the larger complete-stage bottleneck at `87.52%`; it still reconstructs
  two full reference geometries per boundary cell.  See
  `artifacts/ref_gh_analytic_radial_q_20260828/aurora_performance_8789453.txt`.
- Production ready: **no**.

The measured RHS ratio is above twice Z4c, so compiler-guided source splitting
or team-per-cell tests are now authorized.  Production readiness still requires
repair and remeasurement of both the heavily spilled scalar-source/Pi kernel
and the generic-geometry projected boundary path.

The first authorized compiler discriminator changed only CSE grouping: the ten
generated curvature and moving-frame source components were independently
CSE'd and called sequentially.  Local determinism, both builds, all source
oracles, and a full-output cycle passed at unchanged tolerances.  Aurora job
8789565 also passed the bounded evolved PVC cycle, but the primary scalar-source
kernel still spilled about 1300 Reals and slowed from `0.0659391` to `0.159179`
seconds per call (`2.414x`).  Raw throughput fell `26.03%`, from `1.031481e5` to
`7.629483e4` active zone-cycles/s.  This optimization is therefore rejected and
the production source has been restored to the previously qualified joint-CSE
form.  The discriminator script and compact failure evidence are retained in
`artifacts/ref_gh_analytic_radial_q_20260828/component_cse_8789565_regression.txt`.
The matched benchmark input now explicitly selects `trumpet_q_controlled` and
`analytic_radial_q`.  Its closed-loop and static control modes both pass a
one-cycle local dispatch smoke; this is input validation only, not performance
evidence.

The next equation-preserving optimization replaces the measured dominant
q-controlled physical-boundary path with compact radial projection.  The
analytic boundary kernel no longer constructs either full reference geometry;
it reconstructs only the eight post-controller-RK stage scalars from the
resident 12-scalar static coefficients.  A new 2160-sample projection oracle
passes at unchanged `256 epsilon` with conditioned errors `4.56474e-14`
(metric), `9.21858e-16` (gauge), and `7.11991e-16` (subtracted gauge).  The
corrected full-output analytic/generic cycles agree to `4.32393e-14` in the
primary Ref-GH history and give identical error summaries at printed
precision.  See
`artifacts/ref_gh_analytic_radial_q_20260828/compact_boundary_local_checkpoint.txt`.

The current source subsequently passed Aurora job 8789663: one node, eight
ranks, eight distinct PVC tiles, and an evolved dynamic-q RK cycle.  Its
one-rank/eight-rank conditioned Ref-GH history difference was `3.88981e-14`
against the unchanged `5e-12` gate.  The executable SHA-256 was
`76231605157bb22518411720b3d4af93dbd56a65cc5a66194646e23f9b0a1287`.

Aurora job 8789684 then completed the matched warmed-up `64^3` benchmark on
one PVC tile.  Dynamic Ref-GH reached `7.368637e5` active zone-cycles/s,
static Ref-GH `7.470968e5`, and Z4c `7.230060e6`.  The compact boundary costs
`0.00665974` seconds per stage (`7.95%`), a `79.22x` reduction from the
historical job-8789453 boundary timing.  Dynamic Ref-GH throughput improved
`7.14x` over that historical baseline.

The three required overhead targets pass: q control is `0.628%` of a warmed
stage, reference preparation is `0.0495%`, and dynamic/static complete time is
`1.0181`.  The central performance gate remains failed: Ref-GH is `9.185x`
Z4c in complete warmed-stage time and `10.525x` in main-RHS time without
dissipation.  Thus the compact boundary optimization is retained, but the
backend remains **not production ready**.  Full compact evidence and the
remote artifact locations are recorded in
`artifacts/ref_gh_analytic_radial_q_20260828/compact_boundary_pvc_performance_20260828.txt`.

## Team-per-cell RHS discriminator

Because the remeasured main RHS remains `10.525x` Z4c, the next authorized
compiler-guided discriminator assigns one Kokkos team to each active cell.
One team lane prepares the compact physical geometry and common source
workspace in per-team scratch; the ten symmetric component iterations then
evaluate generated curvature/frame contractions and Pi/Phi/gamma2 updates.
The gauge target, gauge driver, reference subtraction, ordinary gauge source,
scalar source, Pi, compatible or standard Phi, and gamma0/gamma2 paths share
that one geometry.  The generic backend and the established matrix-valued
generated source remain independent oracle implementations.

The generator now emits additional independently CSE'd compile-time component
expressions without replacing the accepted joint-CSE matrix functions.  A
second full SymPy 1.14 regeneration was byte-identical.  The strengthened
all-61 oracle exercises the prepared workspace and component expressions over
all 4,320 cases and passes at `2.84217e-14` against unchanged `256 epsilon`.
Actual compatible- and standard-ordering evolved cycles agree with the generic
backend to `2.24305e-15` and `3.77065e-14`, respectively, in evolved common
histories.  This is local correctness evidence only; PVC and performance are
not yet qualified.  See
`artifacts/ref_gh_analytic_radial_q_20260828/team_per_cell_local_checkpoint.txt`.
