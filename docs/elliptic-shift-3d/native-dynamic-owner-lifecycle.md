# Native adaptive owner lifecycle and required transaction

Historical read-only audit after static-SMR qualification. The later explicit
regular-Minkowski transaction is implemented and qualified in
[native-adaptive-regular-integration.md](native-adaptive-regular-integration.md).
Wormhole adaptive and the other unsupported configurations remain guarded.
The parent owns the new host-engine regrid-transfer component separately.

## Actual ordering and object lifetime

`Driver::Execute` calls adaptive refinement at `driver.cpp:625`, after the cycle's
accepted-state outputs. `MeshRefinement::AdaptiveMeshRefinement` checks flags,
updates the tree and calls `RedistAndRefineMeshBlocks` if leaves change. Afterward
it marks the mesh updated, calls `Driver::InitBoundaryValuesAndPrimitives`, then
recomputes timestep (`mesh_refinement.cpp:145--187`). That boundary initialization
reaches `Z4c::ApplyPhysicalBCs` and the existing native `PrepareNative` hook.

Crucially, redistribution does **not** destroy Z4c, its native composite owner,
MeshBlockPack, ADM module or their existing task registrations. It mutates native
field Views in place during Steps5--9, then replaces only the pack's MeshBlock
and Coordinates objects in Step10 (`mesh_refinement.cpp:661--665`). The pack's
counts/GIDs and global logical-location arrays are updated immediately before
those replacements. `MeshBlockPack` and its Z4c/native owner persist at the same
addresses; references to the old MeshBlock/Coordinates must not persist.

The authoritative global22 coefficient arrays, scri state, shift warm start and
stage/register metadata in `NativeCompositeEvolution::Impl` survive untouched
unless explicitly transferred. The existing `NativeCompositeFields` adapter
holds a reference to the engine grid. Replacing/destroying the engine while
retaining that adapter would leave a dangling reference. The core map, block
mapping and expected native mirror buffers also become stale. The current
`ValidateCore` correctly rejects changed bounds/pack slots before using them.

## Why native u0 prolongation cannot define the new canonical state

Current redistribution restricts/copies/prolongs Z4c physical `u0` and
`coarse_u0` (`mesh_refinement.cpp:550,569,588,623`). Those operations carry25
native physical fields, including beta, not the authoritative22 positive-chart
coefficient fields. They do not transfer shell/scri state, canonical RK
registers, chart metadata or an elliptic certificate.

The composite native task route deliberately bypasses `RestrictU`, so its native
`coarse_u0` is not a maintained authoritative restriction. Derefining from that
array without a new transaction would be especially unsafe. The existing
`RefineCC(...,true)` high-order interpolation is a Cauchy physical-field
operation; interpolating positive chi/lapse then silently treating those values
as logarithmic coefficients would change the representation and can introduce
nonpositive temporary values. Interpolating beta as if it were an evolved RK
field would also violate the stage contract.

A future native branch should avoid making those native-only operations
canonical. It must transfer all22 coefficient fields and a separate shift warm
start with the chosen host-engine transfer, then scatter the complete validated
physical mirror and ghosts. Existing generic native restrictions/prolongations
may remain for other physics modules; the composite owner needs its own route.

## Implementable accepted-boundary transaction

1. Before destructive native field reordering, verify the composite stage is
   inactive/accepted, validate the current mirror/topology and retain an owned
   snapshot of authoritative volume/scri/endpoint values, register arrays, chart
   metadata, actual RK binding, time/cycle and separate shift warm start. Preserve
   the original engine and its valid adapters until a replacement is certified.

2. Obtain the planned new leaf list after `CreateZOrderedLLList` and load balance,
   but before Step5 mutates Views. Preflight the50000-cell bound, logical ghost
   contract, complete dyadic partition and1024-cell terminal hierarchy cap. New
   physical bounds can be computed with the same root/level `LeftEdgeX` convention
   as `MeshBlock`; compare them to actual new `mb_size` after Step10. For Serial,
   one complete pack is sufficient. MPI remains a separate communication problem.

3. Prepare a new engine/cache set with identical physics/chart/tolerance controls.
   Transfer all22 volume coefficients and separate beta warm start. Copy unchanged
   shell and scri values **bitwise**, with no angular averaging or radial reduction.
   Preserve their register data too. Transfer volume register arrays or explicitly
   specify a mathematically justified accepted-boundary reset contract; do not
   accidentally reset them by calling a convenience method unnoticed.
   `NativeCompositeStageState::ReplaceAcceptedState` currently sets registers equal
   to values, while `Restore` preserves registers and invalidates geometry once.
   That distinction must be decided and tested in the future dispatch.

4. After native Step10 installs new MeshBlock/Coordinates but before the next
   `InitBoundaryValuesAndPrimitives`, validate actual new records against the
   planned complete partition. Rebuild `NativeCompositeFields`,
   `NativeCartesianCoreMap`, native/composite indices and all mirror buffers for
   the new engine. Rebind/rebuild stage state with the updated topology revision;
   invalidate old shift certification. Recompute the original vector shift and
   direct boundary residuals on the transferred geometry, then scatter every
   native physical active/ghost field and recompute separate physical ADM.
   Only then expose the new owner to the existing boundary/RHS/timestep tasks.

5. Destroy old adapters before their old engine, commit atomically and mark the
   native mesh sequence once. Failure must leave a coherent old transaction
   snapshot or abort before exposing a partly transferred owner. It is insufficient
   to remove the adaptive guard and let the next stale-map check fail after native
   state was already destroyed. Recompute global CFL after the accepted shift.

The exact candidate hook boundaries are the native-specific preparation before
Step5 and commit after `SetNeighbors` in Step10, or an explicit owner transaction
spanning `RedistAndRefineMeshBlocks`. The existing `PrepareNative` hook is a useful
post-commit verifier and output point; by itself it is too late to preflight or
recover old native field reordering. The persistent owner makes the transaction
implementable without rebuilding all Z4c task registrations.

Outputs currently occur before AMR and use only the completed-cycle number.
Post-regrid diagnostics at the same accepted time need explicit topology revision
or a separate regrid event record to avoid overwriting or suppressing pre-regrid
files. Restart should bind the complete post-commit topology and chart state,
rebuild all caches and requalify beta. No midstage, MPI/device or moving-tracker
support should be inferred from a Serial accepted-boundary transaction.
