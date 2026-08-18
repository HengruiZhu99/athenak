# Remote review prompt: reference-frame FO-GH stationary-trumpet pause point

Audit branch `codex/ref-frame-fo-gh-puncture-20260818` adversarially, beginning at
parent `68000b4a753056d5f18333a63175d9e003a32300`.  Review only the new vacuum
reference-frame FO-GH work; do not broaden into fluid coupling, Kerr-Schild data,
or horizon finding.

The branch stops after the production-resolution stationary hard gate fails.
The exact n=2 1+log trumpet reference and common ADM/history diagnostics are
present, but initial RHS worsens strongly with resolution. It
does not implement the wormhole-to-trumpet transition and is not puncture, SMR,
restart, GPU, or long-time qualification.  Use committed source and compact
artifacts as authority, and independently rerun checks where practical.

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
5. Audit the generated trumpet table and interpolation, including branch
   selection, limiting behavior at the trumpet surface and infinity, derivative
   accuracy, table endpoints, and device safety.
6. Derive the stationary trumpet coframe, inverse frame, connection, structure
   coefficients, and derivatives. Verify `Psi=eta`, `Pi=Phi=0`. Explain why
   `Pi00` RHS grows `4.786e-9 -> 8.442e-8 -> 1.357e-7` and reference Ricci grows
   `1.372e-7 -> 7.210e-7 -> 2.316e-6` on `dx=1/16,1/24,1/32`. Determine whether
   the defect is continuum transformation, generated reference jets, or
   near-puncture cancellation.
7. Inspect task ordering, ghost exchange, boundary handling, timestep selection,
   ADM conversion, enrollment, restart assumptions, and ownership/lifetime.
   Treat the current periodic stationary test as a bounded local test, not an
   acceptable production outer-boundary implementation.
8. Check Kokkos device safety, array bounds, symmetric packing, races, and CPU/GPU
   portability.  Flag code that happens to pass Serial but is unsafe on CUDA.
9. Recompute the algebra/source oracle results and the three-resolution linear-
   wave orders.  Inspect whether the tiny errors are credible or could reflect a
   self-comparison, cancellation, or insufficient evolution time.
10. Reproduce the robust-Minkowski test through one crossing and judge whether its
   perturbation construction and measured norm can detect the expected unstable
   modes.  Do not extrapolate it to puncture stability.
11. Reproduce the three stationary t=1 runs and the t=0 n=16,24,32 ADM
    diagnostic ladder.  Explain why native regular GH constraints stay near
    roundoff while whole-domain common ADM H/M norms worsen with resolution,
    yet the fixed 2<=r<4 shell converges rapidly.  Decide whether this is an
    expected singular-coordinate diagnostic effect, an ADM-adapter defect, or a
    formulation defect.  Do not repair it by masking the primary histories.
12. Audit tests for missing negative cases and assess whether they are actually
   registered/executed by the repository test harness.
13. Review the preservation boundary: do not incorporate or fix unrelated dirty
    `bvals`, `fo_gh`, submodule, or untracked-tree changes.

Start with these evidence entry points:

- `docs/ref_gh_reference_frame_validation.md`
- `docs/fo_gh_artifacts/reference_frame_20260818/results.tsv`
- `docs/fo_gh_artifacts/reference_frame_20260818/stationary_trumpet_results.tsv`
- `docs/fo_gh_artifacts/reference_frame_20260818/provenance.txt`
- `docs/fo_gh_artifacts/reference_frame_20260818/stationary_gpu_gate.tsv`
- `docs/fo_gh_artifacts/reference_frame_20260818/perlmutter_stationary_gate_provenance.txt`
- `tst/test_suite/ref_gh/reference_frame_audit.py`
- `tst/test_suite/ref_gh/standard_gh_source_audit.py`
- `tst/test_suite/ref_gh/trumpet_reference_audit.py`

Return findings ordered by severity with file and line references.  Separate
confirmed defects, evidence gaps, and optional improvements.  End with a clear
gate decision: whether the formulation and diagnostics are sound enough to
implement the wormhole-to-trumpet transition and then begin bounded puncture
tests, without implying any puncture qualification.
