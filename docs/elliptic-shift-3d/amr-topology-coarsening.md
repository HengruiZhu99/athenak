# Global AMR hierarchy preparation

This records the original transfer-only gate. Subsequent optional solver wiring
and actual global scalar/vector cycle results are documented in
[global-cycle-qualification.md](global-cycle-qualification.md).

This component prepares the CPU composite-grid preconditioner for Cartesian
topology coarsening. It does not yet change `CompositePoisson`, the final vector
operator, the evolution mesh, or native AthenaK dispatch. Its tests establish
descriptor reduction and transfer algebra; they do not establish convergence of
a multigrid cycle or a coupled evolution.

## Existing limitation and implemented component

At source HEAD `9cced4c72698e9e2a37c4726d139bbb51e877e6d`, the
`CompositePoisson` constructor halves an even logical cell count and otherwise
calls `FactorCoarse()`. That function rejects more than 1,024 active cells before
allocating the dense matrix. For eight Cartesian roots and seven shell layers,
there are 50 blocks. N6 stops at N3, leaving 1,350 cells. Refinement leaves all
Cartesian blocks present even when N reaches two.

`src/multigrid/composite_topology_transfer.hpp` adds three isolated components:

* `CoarsenCompositeCartesian()` merges all complete sibling families at the
  globally finest Cartesian refinement level. It reconstructs dyadic indices
  from bounds anchored at `-half_width`, checks common refinement-zero widths
  and the eight-octant mask, and preserves every shell descriptor exactly.
  Refinement-zero Cartesian roots are retained. Incomplete families are never
  partially merged. Each changed descriptor has a new revision.
* `CompositeCoarseningPlan::Build()` reduces logical N by ceiling-halving down
  to two, then merges Cartesian families until refinement-zero leaves remain.
  This includes odd N3→2 and N5→3. It reports a bounded terminal-cell guard without
  allocating a dense matrix. An incomplete AMR tree and a shell/root topology
  whose N2 floor exceeds 1,024 return explicit unsupported reasons. Shell face
  coarsening and an iterative coarse solve remain future work.
* `CompositeLevelTransfer` maps point-major active fields between nested
  descriptors. It owns sparse restriction and prolongation rows and the
  represented physical measure. It has no patch-local fixed interface solve.

For a fine cell f overlapping coarse cell c, restriction uses

    w_cf = fine_physical_volume[f] * logical_overlap_fraction(c,f),
    coarse_volume[c] = sum_f w_cf,
    u_c = sum_f w_cf * u_f / coarse_volume[c].

All weights are positive, each fine cell's fractions sum to one, and all cells
participate. The default measure is the mapped physical cell volume. A caller
may supply positive finite, globally scaled metric-dependent physical volumes;
the next transfer must receive the preceding transfer's `coarse_volume`, rather
than resampling a new measure. Anchor-difference evaluation preserves a constant
bitwise. On a partial fine-cell overlap, the supplied fine volume/value is
piecewise constant. Thus N3→2 conserves the represented mass and preserves
constants; it is not an exact analytic integral over the subcell, nor an exact
linear restriction on mapped cells. This is a lower-order preconditioner choice.

Prolongation searches all coarse active physical centers with a KD tree. It
reconstructs physical Cartesian polynomial moments up to degree two from a
bounded donor ball (24, 48, 96, 192, 384, or 768 donors). Co-located active points
use identity transfer. Normalized local coordinates, a small moment solve,
moment-error and row-L1 checks reject nearly dependent nearest donors and grow
the ball. A documented linear fallback is counted, never hidden; the tested
meshes use no fallback. Negative prolongation weights are allowed, as in
polynomial interpolation. This is not a proof of multigrid smoothing stability
or an adjoint/Galerkin transfer pair. The largest tested row L1 is 15.97.

## Reproduction and evidence

From the isolated worktree, using its existing Serial Kokkos build flags:

```sh
python3 tst/elliptic_shift_3d/run_composite_topology.py \
  --build build-shift-3d \
  --output docs/elliptic-shift-3d/evidence/amr-topology-transfer-reproduction
```

The runner bounds both build and test to 180 seconds, freezes project source
headers/configuration before building, and records exact commands, source and
library hashes, process duration, binary hash, and source/library stability.
It does not edit CMake. The final completed record is
`evidence/amr-topology-transfer-final-20261027/manifest.json`; test output is
JSON lines in `test.stdout`. Its maximum mesh has 33,104 active cells.

| Test | Reduction/result |
|---|---|
| Mixed Cartesian levels, N4, two shells | 34→27→20 blocks; complete finest families only; unchanged shell descriptors |
| Uniform Cartesian refinement depth three, N2, seven shells | 4,096→512→64→8 Cartesian leaves; 4,138→554→106→50 total blocks; 33,104→4,432→848→400 cells |
| N6, seven shells | N6→3→2; 10,800→1,350→400 cells |
| Two shells, four blocks per shell face | 56 blocks; N4→2; terminal 448 cells |
| Seven shells, four blocks per shell face | Explicit terminal-floor rejection at 1,408 cells; no dense allocation |
| Incomplete sibling family | Explicit unsupported plan; no partial merge |

All tested transfers preserve constants bitwise. Mixed and deep Cartesian
topology transfers reproduce linear point values and Cartesian quadratic cell
averages to at most 3.34e-16. Global prolongation reproduces physical quadratic
point values to at most 2.77e-13 across all cells, including patch/face/layer
interfaces; no linear fallback occurs. Represented-mass relative errors are at
most 6.23e-15 for individual transfers. Additional chained transfers propagate a
normalized nonuniform metric-dependent physical measure and a nonpolynomial
field; their mass drift remains below 2e-14. N3→2 linear nodal restriction error
is 0.08343 on this deliberately coarse mapped hierarchy and is recorded, not
tested as exact. No cells or boundary regions are excluded.

The first nearest-96-donor control failed on thin shell layers: 320 rows fell
back to linear, quadratic error was 0.01012 and maximum row L1 was 345.71.
`evidence/amr-topology-transfer-20261027` and
`evidence/amr-topology-transfer-diagnostic-20261027` preserve those failed
attempts. Bounded donor growth plus moment/L1 rejection fixes that transfer
failure in `evidence/amr-topology-transfer-donor-growth-20261027` and the final
expanded test. This changes only candidate preconditioner transfer; no physical
equations, boundary data, mesh acceptance targets or elliptic tolerances change.

## Concrete integration path

1. In the `CompositePoisson` constructor, build/check the complete descriptor
   plan before dense allocation. Construct each level's `CompositeDifference`
   with the existing derivative order and radial-stencil choice. Every level
   must build its own global active donor/ghost plan, including faces, edges,
   corners, the Cartesian/shell interface and mixed Cartesian leaves. A topology
   descriptor alone does not establish those halos. Keep the 1,024-cell guard.
2. Replace `Cycle()`'s same-block `2*i+di` restriction and same-block trilinear
   correction loops with the corresponding `CompositeLevelTransfer::Restrict`
   and `Prolong` calls. Those loops currently assume identical block order/count
   and a 2:1 logical ratio, so using only the new descriptor would be incorrect.
   Carry represented physical volumes down the hierarchy. Coefficient
   coarsening, when added, must use the same transfer/measure contract.
3. Keep homogeneous correction boundary treatment in the coarse/global
   operator, smoothers, and residual calculation. Prolongated active values do
   not supply independent potential or interface Dirichlet data. Re-evaluate
   the global operator and original vector boundary residual after correction.
4. Validate actual preconditioner cycles first on flat global scalar and vector
   manufactured data: one/two/seven shells, subdivided faces, mixed leaves,
   N6/N8, coarse-fine interfaces, edge/corner residuals and complete all-cell
   physical norms. A good polynomial transfer alone does not establish that
   the existing weighted Jacobi smoother or rediscretized hierarchy converges.
   The nonsymmetric transfers are intended for outer FGMRES, not assumed SPD
   preconditioning. Record convergence and cost before using them in a coupled
   pilot.
5. After refinement/coarsening/restart or a changed chart/metric measure, rebuild
   the descriptor plan, transfer rows, global halos and coefficients together.
   Treat geometry and mesh revisions as cache keys. Rebuild the algebraic shift
   before tensor RHS, according to `NativeCompositeStageState`'s epoch contract.
   The evolution's accepted endpoint/scri/chart state must be transferred and
   restarted separately; this helper stores only solver hierarchy state.

No changes to native task graphs, MPI communication, Kokkos device kernels,
shell restart or `CompositePoisson` are included in this component. MPI/device
tests and a globally converged solve using this hierarchy remain unperformed.
