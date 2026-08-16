# Read-only review prompt: Brill-wave Cartoon Z4c AMR instability

Please perform a skeptical, source-grounded, **read-only** review of this
AthenaK branch and its compact evidence handoff.

Repository: <https://github.com/HengruiZhu99/athenak>  
Branch: `codex/brill-amr-frozen-hierarchy-20260816`  
Numerical source commit: `21a268e4735308a39ac4f040d3621ea114b4ef1d`  
Source tree: `394aa38e76951249de0f247c3a893e0af4a0f1d9`  
Report: `docs/brill_amr_hierarchy_causality_20260816/REPORT.md`  
Evidence manifest: `docs/brill_amr_hierarchy_causality_20260816/EVIDENCE_MANIFEST.json`  
Plots/data: `docs/brill_amr_hierarchy_causality_20260816/{campaign,target_event,parent_state}/`

## Controlled experiment

All three N256 continuations use the identical cycle-1721 restart immediately
before the level-2-to-3 refinement at `t=9.50625 M`, with O6/RK4/CFL 0.15,
KO 0.02, `dchi_max=0.01`, derefine ratio 0.5, the same gauge/damping/data,
high-order transfer, and unchanged strict chi gates.

- **A dynamic:** 67 topology changes; timed out at `t=11.9547843933 M`
  after runaway refinement to level 12 and 350 MeshBlocks, `dt=4.58e-6 M`,
  `C=3.43e7`.
- **B frozen:** permitted the identical target event, then froze at level 3
  and 86 MeshBlocks; reached `t=12.5 M`, `dt=1.56e-3 M`, `C=86.3`.
- **C buffered frozen:** target event plus one full same-level MeshBlock buffer,
  frozen at level 3 and 128 MeshBlocks; failed at `t=12.4828125 M` with 346
  invalid chi parent stencils and catastrophic constraints.
- B2 limited-O2 was skipped prospectively because B reached the target time.
- The requested `diss=0.5` job 57102293 was revoked before allocation; it used
  no GPU and produced no science data.

The Cartoon history cell measure is already
`2*pi*rho*dx1*dx2*sqrt(abs(det(gamma)))`; the target coordinate ring volume is
conserved to roundoff. The constraint jump is not a fictitious suppressed-y
normalization effect.

## Target-event and parent-state evidence

At cycle 1722, the T0-parent to T2-child coarse residual is exactly zero and
canonical O6 prolongation agrees to `1.78e-15`. Active evolved fields do not
change under MPI receive or boundary prolongation; stored ghosts do, and the
constraint recomputation on the new fine lattice jumps strongly. The worst C
change is at `(rho,z)=(5.1328125,-0.0078125)`, one fine cell from a MeshBlock
edge, far from the axis and 0.133 M from the nearest coarse-fine interface.

The pre-event parent audit finds edge-localized self-shadow residuals: K is
0.394 in the edge band versus 0.00579 in the interior; Atilde is 0.329 versus
0.0169. All maxima are at an internal block corner near the Brill-wave ring,
not the symmetry axis. O6-O4 disagreement and block-local Nyquist content are
also non-negligible.

## Questions

1. What is the smallest source-level mechanism consistent with A running away
   under repeated topology changes while the same-event frozen B remains
   bounded through 12.5 M?
2. Why can the larger buffered frozen C fail late even though B is stable?
   Please scrutinize block-edge one-sided restriction, repeated point-value
   transfer, ordinary same-level/coarse-fine boundary refresh, stage freshness,
   and whether C's larger fine representation exposes a bulk mode.
3. Does the phase ledger really rule out a missing ghost-fill write at the
   first event, or could a derivative-reachable ghost inconsistency remain
   invisible to the active-field writer norm?
4. Given the parent edge/interior self-shadow split, is the likely defect an
   under-resolved parent state, a non-monotone/ill-conditioned transfer pair,
   a block-local edge closure, or a non-normal AMR-interface feedback mode?
5. Propose the **smallest decisive next diagnostic**. Prefer a bounded
   no-PDE regrid/refine-derefine cycle, stage-resolved first-invalid provenance,
   or a minimal frozen two-level operator test over another long evolution.
6. Only if the evidence isolates a concrete defect, propose the smallest
   source correction and the exact falsification/acceptance test for it.

Clearly label observations, deductions, hypotheses, and unsupported
possibilities. Do **not** recommend chi floors, weakened positivity gates,
threshold relaxation, broad gauge/dissipation/AMR scans, or unsupported
convergence/Figure-3 claims. Do not treat the unexecuted `diss=0.5` attempt as
data.

