# Remote review prompt: reference-frame FO-GH foundation

Audit branch `codex/ref-frame-fo-gh-puncture-20260818` adversarially, beginning at
parent `68000b4a753056d5f18333a63175d9e003a32300`.  Review only the new vacuum
reference-frame FO-GH work; do not broaden into fluid coupling, Kerr-Schild data,
or horizon finding.

The branch deliberately stops after flat-reference linear-wave and robust-
Minkowski gates.  Do not treat it as stationary-trumpet, puncture, SMR, restart,
GPU, or long-time qualification.  Use committed source and compact artifacts as
authority, and independently rerun checks where practical.

Please perform both a mathematical formulation audit and a conventional code
review:

1. Verify that `RefGhState` contains exactly 50 independent fields with correct
   symmetric-pair indexing and no hidden evolved gauge variables.
2. Check frame/coframe duality, index placement, reference connection use, and
   all metric and derivative transforms in `src/ref_gh/`.
3. Derive the standard-GH source and first-order scalar-wave reduction from the
   declared conventions.  Pay special attention to the documented sign erratum:
   with `Pi=-n^a partial_a Psi` and `Box Psi=S`, determine independently whether
   the lower-order terms must be `-Phi_i D^i alpha + alpha S`.
4. Compare `ref_gh_calcrhs.cpp` term by term with a trusted first-order GH
   equation, including damping, advective, lapse-gradient, shift-gradient, and
   reduction-constraint terms.  Identify any terms justified only by the flat
   reference specialization.
5. Inspect task ordering, ghost exchange, boundary handling, timestep selection,
   ADM conversion, enrollment, restart assumptions, and ownership/lifetime.
6. Check Kokkos device safety, array bounds, symmetric packing, races, and CPU/GPU
   portability.  Flag code that happens to pass Serial but is unsafe on CUDA.
7. Recompute the algebra/source oracle results and the three-resolution linear-
   wave orders.  Inspect whether the tiny errors are credible or could reflect a
   self-comparison, cancellation, or insufficient evolution time.
8. Reproduce the robust-Minkowski test through one crossing and judge whether its
   perturbation construction and measured norm can detect the expected unstable
   modes.  Do not extrapolate it to puncture stability.
9. Audit tests for missing negative cases and assess whether they are actually
   registered/executed by the repository test harness.
10. Review the preservation boundary: do not incorporate or fix unrelated dirty
    `bvals`, `fo_gh`, submodule, or untracked-tree changes.

Start with these evidence entry points:

- `docs/ref_gh_reference_frame_validation.md`
- `docs/fo_gh_artifacts/reference_frame_20260818/results.tsv`
- `docs/fo_gh_artifacts/reference_frame_20260818/provenance.txt`
- `tst/test_suite/ref_gh/reference_frame_audit.py`
- `tst/test_suite/ref_gh/standard_gh_source_audit.py`

Return findings ordered by severity with file and line references.  Separate
confirmed defects, evidence gaps, and optional improvements.  End with a clear
gate decision: whether this branch is sound enough to begin the stationary 1+log
trumpet reference implementation, without implying any puncture qualification.
