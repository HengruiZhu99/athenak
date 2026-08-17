# Read-only review prompt: Brill AMR constraint-jump causal audit

Please perform a skeptical, source-grounded, **read-only** assessment of the
AthenaK code and the authenticated evidence below. You cannot run the code.
Your job is to look for concrete bugs or mathematical inconsistencies and to
recommend the smallest decisive next step, not to manufacture a production
fix from incomplete evidence.

Repository: <https://github.com/HengruiZhu99/athenak>

Branch: `codex/brill-amr-frozen-hierarchy-20260816`

Capture source commit: `e6b1428cbe1fafe941ac6a41cbabe14430ed8d14`

Analyzer repair commit: `a96c49e454674df8d1567a6d489ac08ce40a6f01`

Derivative-audit commit: `392522dda2737508697662982f688809a154d571`

Derivative-audit tree: `054d370479af2fc73d775b5e3ef8325bf288f90d`

Primary report:
`docs/brill_amr_constraint_jump_causal_transfer_20260816/REPORT.md`

Evidence index:
`docs/brill_amr_constraint_jump_causal_transfer_20260816/evidence_manifest.json`

Plots and cell-level reductions:

- `docs/brill_amr_constraint_jump_causal_transfer_20260816/v3_causal_gate/`
- `docs/brill_amr_constraint_jump_causal_transfer_20260816/v4_derivative_audit/`

Governing prospective plan:
`docs/goal_mode_z4c_amr_constraint_jump_causal_transfer_comparison.md`

Earlier A/B/C hierarchy-causality handoff:
`docs/brill_amr_hierarchy_causality_20260816/`

## Fixed numerical event

This is the N256, `A=-0.047` Brill case with imported ADM mass
`2.660301967997158 M`. The exact restart is cycle 1721 at
`t=9.5015625 M`, immediately before the cycle-1722 level-2-to-3 event at
`t=9.50625 M` (74 to 86 MeshBlocks). The matched arms use the same restart,
input, executable, rank, GPU, proposed topology, O6 bulk derivatives, RK4,
CFL 0.15, KO 0.02, `dchi_max=0.01`, derefine threshold `0.5*dchi_max`, and
zero Z4c constraint damping. They stop after the target T1--T5 transaction and
before another RHS evaluation.

The Cartoon history measure is already the proper axisymmetric ring measure,
`2*pi*rho*sqrt(gamma)*d(rho)*dz`; this is not a fictitious collapsed-y volume
normalization jump.

## Prospective causal gate result

The global `limited_o2` transfer arm did **not** reduce the frozen-event jump.
Its proper-ring post/pre ratios were much worse:

| Constraint | high-order | limited-O2 |
|---|---:|---:|
| C | 95.2237 | 1304.7564 |
| H | 42.6188 | 1270.2369 |
| M | 195.8316 | 2040.8443 |

The two arms reconstruct byte-identical T0 evolved/ADM/constraint states and
accept the same child lattice. Their writer and regional ledgers close. The
dominant excess is at MeshBlock edges/corners, but limited-O2 increases rather
than cures it. Therefore the planned edge-only low-order implementation and
evolved three-method comparison were correctly skipped.

Job `57137565` completed both zero-PDE arms. Its allocation parent failed only
because the first offline analyzer divided by the zero volume of a valid empty
disjoint region. Commit `a96c49e4...` fixes only that offline case and the
immutable captures were not rerun.

## Same-state derivative-order audit

Job `57140240` evaluated O2/O4/O6 constraints from one accepted T5 evolved
state. Independently recomputed O6 bytes equal the production T5 bytes.
Proper-ring global integrals were:

| Constraint | O2 | O4 | O6 | O2/O6 | O4/O6 |
|---|---:|---:|---:|---:|---:|
| C | 32.8146 | 60.3145 | 74.8573 | 0.43836 | 0.80573 |
| H | 6.03992 | 10.4888 | 12.7106 | 0.47519 | 0.82520 |
| M | 26.6931 | 49.7402 | 62.0602 | 0.43012 | 0.80148 |

Localization is broadly stable across order: MeshBlock-edge/corner cells
carry about 73% of C, 95% of H, and 68--69% of M; coarse-fine cells carry only
about 7%, 3%, and 8%. The largest O2--O6 differences lie near the Brill ring at
`rho about 5.1 M`, close to internal MeshBlock edges. The selected refined set
contains no axis/physical-boundary cells, so this event cannot test the
axis-ghost hypothesis.

Current disposition:
`inconclusive_parent_resolution_or_derivative_sensitivity`, with
`qualification_claim=false`.

## Source areas to scrutinize

Please inspect at least:

- `src/z4c/amr_jump_diagnostic.{hpp,cpp}`
- `src/z4c/z4c_adm.cpp`
- `src/mesh/mesh_refinement.cpp`
- `src/mesh/restriction.hpp`
- `src/mesh/prolongation.hpp`
- `src/bvals/prolongation.cpp`
- the Cartoon parity/axis boundary paths used by Z4c
- `tst/test_suite/z4c/cartoon_amr_jump_analyze.py`
- `tst/test_suite/z4c/cartoon_amr_transfer_compare.py`
- `tst/test_suite/z4c/cartoon_amr_derivative_order_audit.py`

Check the common-lattice mapping, ring weights, region precedence, stencil
selection/orientation, derivative-reachable ghost freshness, parent/child
coordinate mapping, block-edge one-sided closure, axis parity, and the meaning
of the O2/O4/O6 comparison. Look for a defect that could survive the writer
ledger while appearing only when derivatives are recomputed on the new child
lattice.

## Questions to answer

1. Is the causal gate method mathematically sound, including its T0 parent to
   common-child-lattice comparison, or can it exaggerate the jump by comparing
   inequivalent discrete representations?
2. Is global limited-O2 expected to be this much worse for these Z4c point
   values, or does the magnitude suggest an implementation, orientation, or
   parent/child indexing bug?
3. Do the derivative-order integrals genuinely diagnose order sensitivity, or
   could a normalization, derivative ghost width, component contraction, or
   stale-ghost error explain their monotone O2/O4/O6 spread?
4. Why does the error remain concentrated at internal MeshBlock edges across
   all three derivative orders? Identify any source path where active values
   are correct but derivative-reachable ghosts can be stale, from a different
   RK stage, or filled with an inconsistent operator.
5. Reconcile this result with the earlier matched hierarchy evidence: dynamic
   A runs away under repeated regridding, frozen B remains bounded through
   `12.5 M`, but larger buffered frozen C fails late. What mechanism explains
   all three without overclaiming?
6. What is the smallest decisive next diagnostic? Prefer a bounded no-PDE
   parent-resolution/self-shadow test, exact stage/writer provenance, or a
   minimal two-level interface operator test. Specify the observable and the
   pass/fail decision table.
7. Only if a concrete defect is isolated, propose the smallest source change
   and an exact regression/falsification test.

Separate **observations**, **deductions**, **hypotheses**, and **unsupported
possibilities**. Do not recommend chi floors, clipping, weakened positivity
gates, relaxed thresholds, broad gauge/dissipation/AMR sweeps, or unsupported
convergence/Figure-3 claims. Do not assume an axis ghost bug when this selected
event has no axis cells, and do not call the code correct merely because the
ledgers close.
