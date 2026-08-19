# Remote review prompt: covariant Ref-GH repair pause point

Audit branch `codex/ref-gh-covariant-source-repair-20260818` starting from
commit `e162917c`.  Review only vacuum reference-frame FO-GH; do not work on
fluid coupling, Kerr-Schild data, horizon finding, AMR/SMR, or unrelated dirty
files.

The current state is a pause point, not a puncture or long-time qualification.
The covariant lower-order source, high-precision independent oracle, flat
regressions, short four-GPU stationary ladder, and a repaired Ref-GH restart
path are available.  `t=20` stationary and wormhole-to-trumpet transition
gates have not been completed.

Perform a mathematical formulation audit and conventional code review:

1. Independently derive the frame-covariant GH lower-order source in
   `src/ref_gh/covariant_gh_source.hpp` and check all signs, contractions,
   frame-connection terms, and Riemann conventions against the coordinate GH
   source.
2. Audit the direct implicit trumpet oracle and Hermite reference provider for
   shared assumptions, interpolation consistency, index placement, and
   near-puncture conditioning.  Determine whether the reported `Q=Delta=0`
   residual is a genuine independent check.
3. Check the 50-field state, symmetric packing, task ordering, boundary fills,
   ADM reconstruction, timestep calculation, and device safety.
4. Review `e162917c` specifically.  Verify restart write/read offsets and
   collective MPI paths for Ref-GH are symmetric with FO-GH, and confirm that
   the prior checkpoint could not possibly contain Ref-GH state.  Inspect the
   actual nonzero-time restart history for continuity.
5. Reproduce the exact Minkowski, robust Minkowski, linear-wave, t=0,
   t=0.1, and t=1 stationary checks where practical.  Distinguish evidence of
   stationary roundoff preservation from a convergence or puncture claim.
6. Identify any remaining formulation blocker before the required t=20
   stationary ladder and time-dependent wormhole-to-trumpet test are launched.

Use these artifacts as the primary evidence:

- `docs/ref_gh_covariant_source_repair.md`
- `docs/fo_gh_artifacts/reference_covariant_repair_20260818/`
- `docs/fo_gh_artifacts/reference_covariant_repair_20260818/perlmutter/README.md`
- `tst/test_suite/ref_gh/high_precision_trumpet_source_oracle.py`
- `tst/test_suite/ref_gh/covariant_gh_source_audit.py`

Return findings ordered by severity with file and line references.  Separate
confirmed defects from evidence gaps and end with a clear decision: whether it
is sound to proceed to the outstanding stationary and transition gates.  Do
not describe the present work as puncture, production, or long-time qualified.
