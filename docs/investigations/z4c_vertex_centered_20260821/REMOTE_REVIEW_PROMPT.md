# Read-only review prompt: native vertex-centered vacuum Z4c

Please perform a skeptical, read-only review of the native vertex-centered
vacuum Z4c implementation and bounded evidence.

Repository: <https://github.com/HengruiZhu99/athenak>

Branch: `codex/z4c-vertex-centered-cartoon-amr-20260821`

Implementation commit: `5d37b5e5c278ac4a1afd52f9553dee6ffed48d0e`

Start with:

- `docs/investigations/z4c_vertex_centered_20260821/REPORT.md`
- `docs/investigations/z4c_vertex_centered_20260821/EVIDENCE_MANIFEST.json`
- `docs/investigations/z4c_vertex_centered_20260821/implementation_inventory.md`
- `docs/investigations/z4c_vertex_centered_20260821/unsupported_feature_inventory.md`
- `docs/design/z4c_vertex_centered_design.md`
- `docs/design/z4c_vertex_cartoon_axis_regularization.md`

## Established evidence

- CC remains the default and its bounded history/timestep payload is
  byte-identical to the pre-VC baseline.
- Native VC uses N+1 active nodal ranges, deterministic shared-node
  synchronization, nodal AMR, native diagnostics, restart/output metadata, and
  explicit rejection of nonvacuum physics.
- Fixed-grid VC O4 N128/N256/N512 reaches `t=0.5 M`; nontrivial fields converge
  at effective order 3.31--4.08 and duplicated vertices match bitwise.
- Common-tree N128/N256/N512 executes the same six AMR events at zero-ULP time
  difference and the same hierarchy checksum.
- Those common-tree runs fail the post-RK metric-SPD gate at `t=3.4832`,
  `2.4372`, and `1.3893 M` respectively.  Median common-interval constraint
  orders are C -2.91, H -2.89, M -3.04, Z -2.12.
- The retained failure points are off-axis and not adjacent to a recorded
  coarse-fine interface.  The earlier bug that treated every MeshBlock lower
  radial vertex as `rho=0` is fixed and regression-tested.
- Aurora exposed a dynamic-refinement device failure.  A narrow repair changed
  the production lambda to capture device views, not `DualView` wrappers, but
  the exact-final-source retry still passes tests 1--16 and then faults in the
  first dynamic VC refinement.  Both authenticated failures are retained; the
  first diagnosis is explicitly not treated as conclusive.
- The selected numerical matrix covers many requested semantics under combined
  tests, but the governing prompt's complete stable-name matrix is absent; in
  particular there is no production VC gauge-wave test.  Do not infer this
  missing result from the linear-wave or derivative tests.

## Questions

1. Is the VC layout/topology model internally coherent across same-level,
   coincident coarse-fine, hanging, axis, physical-boundary, and ghost nodes?
2. Do AMR refinement, derefinement, load balance, restart, and accepted-state
   task ordering preserve one authoritative value per physical vertex?
3. Are the axis regularized formulas and parity assignments correct for all 25
   evolved components and O2/O4/O6 stencils?
4. Does any remaining code path classify a MeshBlock-local lower vertex as the
   physical axis, sample VC data with CC coordinates, or use stale topology
   after regridding?
5. Is `VC_AND_CC_SHARE_Z_MODE` a defensible discriminator label given the
   common resolution-worsening Z/constraint behavior, while explicitly not
   claiming the same source mechanism?
6. What is the smallest decisive stage-resolved diagnostic for the post-RK SPD
   loss?  Please identify exact source locations and observables that would
   separate bulk RK generation, duplicate synchronization, restriction,
   exchange/boundary reconstruction, and prolongation.
7. On the SYCL path, what is the smallest source-grounded diagnostic that can
   localize the invalid device access after the first accepted `RefineVC`
   transaction?  Review device-view lifetimes, topology-plan rebuilds,
   pack-capacity changes, and every kernel that first consumes the new tree.
8. Which missing stable-name tests represent genuinely new numerical coverage,
   rather than safe aliases of existing production-kernel tests?  Prioritize a
   real VC gauge-wave test and any AMR/MPI case not already covered at equal
   scope.

Keep observation, inference, and hypothesis separate.  Do not recommend a
metric/chi floor, clipping, weakened invariant gates, broad gauge/KO/CFL
sweeps, a long production run, or an unsupported convergence/Figure-3 claim.
The desired output is a source-grounded review and one narrow next diagnostic
or correction, not new execution.
