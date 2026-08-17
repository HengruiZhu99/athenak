# Read-only review request: Brill cycle-1722 AMR coarse-cache seam

Repository: https://github.com/HengruiZhu99/athenak

Branch: `codex/brill-amr-frozen-hierarchy-20260816`

Please review the branch head read-only.  The production event was captured at
commit `b465629caf20be81c4f19f7c818fd3e0b9b2c242` (tree
`9f797582eb44b9dc079457529e50474a7dd129d5`).

Primary handoff:

- `docs/investigations/brill_amr_stitched_transfer_diagnosis_20260817/REPORT.md`
- `docs/investigations/brill_amr_stitched_transfer_diagnosis_20260817/verdict.json`
- `docs/investigations/brill_amr_stitched_transfer_diagnosis_20260817/evidence_manifest.json`
- analyzer:
  `tst/test_suite/z4c/cartoon_amr_stitched_transfer_diagnose.py`

## Numerical event

- N256 Brill wave, amplitude `A=-0.047`
- ADM mass `2.660301967997158 M`
- cycle 1722, `t=9.50625 M`, 74 to 86 MeshBlocks
- O6, RK4, CFL 0.15, KO 0.02
- `dchi_max=0.01`, derefine threshold `0.5*dchi_max`
- `kappa1=kappa2=0`, `floor_chi=false`
- strict finite/positive chi gates unchanged
- declared restart SHA256:
  `83e996d2d5069307888a69fff47a7524c2f63f11869fb628630bca54dd5943ea`

No new PDE evolution was run for this handoff.

## New decisive evidence

1. Offline `P5_BLOCK` reproduces all 16 production child active states with
   maximum absolute error `1.7763568394002505e-15`.
2. Every checked ordinary same-level fine ghost equals its sender active cell
   exactly after receive and through later boundary phases: 5,632 cells per
   post-receive phase, all 25 components, depths 1-4.
3. The same-level `coarse_u0` overlap is exact for all 2,688 checked cells at
   `T3_01_MPI_RECEIVE`.
4. `T3_03_SAME_LEVEL_COARSE_REFRESH` then changes 1,344 of those 2,688 values
   above a `128*eps` scaled gate.  Worst is `Khat`, discrepancy
   `0.28677378470336334`, near `(rho,z)=(5.109375,-0.046875) M`.
5. The writer is `Z4c::Prolongate -> FillCoarseInBndryCC -> ProlCCSame ->
   RestrictInterpolation<4>`.  Sender and receiver routes use different
   block-local high-order restriction stencils/orientations for the same
   physical coarse location.
6. The strict disposition is `concrete_ghost_or_cache_bug_isolated`, more
   precisely a coarse-cache coherence/seam-semantics defect.  It is not an
   ordinary fine-ghost or axis-parity mismatch.
7. P8_STITCHED reduces exploratory L5-to-L6 C/H/M integrals to roughly
   0.345/0.415/0.330 of P5_STITCHED, but not the L4-to-L5 balance-region values.
8. The independent NumPy constraint port did not meet its strict production
   reproduction gate.  Consequently those stitched constraint ratios are
   exploratory only and are not the basis of the disposition.

## Source paths to inspect

- `src/z4c/z4c_tasks.cpp` (`Z4c::Prolongate`)
- `src/bvals/prolongation.cpp` (`FillCoarseInBndryCC`, `ProlCCSame`)
- `src/mesh/restriction.hpp` (`RestrictInterpolation<2/3/4>`)
- `src/bvals/bvals_cc.cpp` or the relevant receive/unpack implementation
- `src/z4c/amr_jump_diagnostic.cpp` (capture/census semantics)
- `src/z4c/cartoon_derivatives.hpp` and `src/z4c/z4c_adm.cpp` (constraint-port audit)

## Questions

1. Is it semantically valid for `FillCoarseInBndryCC` to overwrite an exact
   sender-authoritative same-level coarse overlap with a receiver-local
   restriction result that differs on non-polynomial data?
2. Which exact coarse slots are genuinely required corners for a neighboring
   coarse-fine prolongation stencil, and is the current loop writing a broader
   same-level face region than necessary?
3. What is the smallest safe correction: preserve received overlap values,
   restrict the refresh inventory to true mixed-level corners, or define one
   canonical global restriction stencil/orientation?
4. What minimal unit test will distinguish these alternatives across NGHOST
   2/3/4, axis-adjacent/off-axis layouts, and local/MPI ownership routes?
5. Can you identify the discrepancy in the NumPy Cartoon ADM-constraint port
   from `production_reproduction_validation.json`, without treating the port as
   authoritative meanwhile?
6. After the cache invariant is repaired, is one repeated zero-PDE cycle-1722
   transaction sufficient before a narrowly matched evolution rerun?

Please separate observations, deductions, hypotheses, and unsupported
possibilities.  Do not recommend chi floors, weakened positivity gates, broad
parameter sweeps, or unsupported convergence/Figure-3 claims.  Do not assume
that P8 should be promoted merely because one subregion improves.
