# Read-only external review prompt

Please perform a source-and-artifact-only numerical-relativity review of this
AthenaK Cartoon Z4c AMR constraint-jump diagnosis. Do not propose edits until
you have identified a source-grounded failure mechanism, and do not assume you
can compile or run the code.

- Repository: https://github.com/HengruiZhu99/athenak
- Branch: `codex/z4c-amr-jump-diagnosis-20260815`
- Diagnostic/report commit: `4e38cb5093ad405b89000310eeb0d502115c60ee`
- Report: https://github.com/HengruiZhu99/athenak/blob/codex/z4c-amr-jump-diagnosis-20260815/docs/investigations/brill_n256_amr_jump_diagnosis_20260815/README.md
- Verdict and phase ledger: https://github.com/HengruiZhu99/athenak/tree/codex/z4c-amr-jump-diagnosis-20260815/docs/investigations/brill_n256_amr_jump_diagnosis_20260815

The authenticated N256 event is the runtime cycle-1722 refinement transaction
(the corresponding history sample is labeled cycle 1724). Active evolved data
survive refinement to roundoff. MPI receive and coarse-to-fine prolongation make
no active-cell change but alter stored ghost/interface data; after a complete
valid boundary reconstruction, the fixed-lattice constraint field changes
strongly. Algebraic projection is a much smaller contribution. Ring-coordinate
volume is invariant to roundoff and proper volume changes by only about
`2.7e-8` relatively, so this is not a collapsed-dimension normalization error.
The largest change occurs at `rho=5.1328125`, `z=+/-0.0078125`, half a cell from
a MeshBlock edge and near a new coarse-fine interface—not near the axis
`rho=0`.

Please audit, with exact file/line references:

1. Whether the AMR task order guarantees stage-current Z4c ghost values and
   complete O6 derivative halos after refinement, MPI exchange, physical/axis
   boundary fills, same-level coarse refresh, and coarse-to-fine prolongation.
2. Whether any newly created fine block can retain stale, partially filled, or
   semantically inconsistent corner/edge ghosts even though the task calls all
   complete.
3. Whether collapsed-x3 Cartoon parity/axis handling can interact incorrectly
   with coarse-fine or MeshBlock boundaries. Reconcile any such hypothesis with
   the observed off-axis worst points rather than assuming an axis bug.
4. Whether restriction/prolongation, coarse-cache construction, ownership,
   index ranges, or writer ordering are inconsistent for point-valued Z4c data
   at the 2D refinement interface.
5. Whether the fixed-lattice comparison or constraint recomputation could be
   sampling invalid intermediate halos. The report explicitly says separate
   post-MPI and post-prolongation constraint effects cannot be evaluated because
   those intermediate halos are intentionally incomplete; do not treat that
   limitation itself as proof of a missing production fill.

Return: (a) a ranked list of concrete bug hypotheses, (b) source evidence for
and against each, (c) the single smallest diagnostic or invariant test that
would distinguish the top hypotheses, and (d) whether a matched
`O6 bulk + limited-O2 AMR transfer` ablation is justified. Do not recommend a
chi floor, threshold relaxation, gauge sweep, or vertex-centered rewrite as the
first response. Clearly distinguish confirmed facts, inferences, and unknowns.
