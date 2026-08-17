# Prompt for a remote-agent audit of the vacuum FO-GH work

Audit the latest pushed HEAD of branch
`codex/fo-gh-puncture-driver-20260817` in the AthenaK repository.  Begin by
recording the full commit hash, branch, dirty state, and Kokkos gitlink.  Treat
the Git tree and the committed evidence as authoritative; do not use unrelated
dirty submodules or adjacent checkouts.

Scope is vacuum first-order generalized harmonic (FO-GH) only.  Do not work on
fluid coupling, Kerr-Schild data, or an apparent-horizon finder.  Do not start
or alter a Perlmutter allocation or any long-running evolution as part of this
audit.  In particular, do not interfere with another agent's jobs or working
directories.

Perform a detailed, adversarial source-and-evidence audit of all FO-GH work on
this branch against base commit `24dd527514a3b031d151ca8d3f2679e998a91b3d`.
Prioritize correctness over feature enumeration.  Review at least:

1. The fixed regularized continuum equations in `src/fo_gh/`, including index
   placement, symmetric-tensor storage, trace-free projections, shift
   advection, gauge-driver terms, characteristic speeds, and every Lambda
   source.
2. The ADM-to-regular and regular-to-standard-GH maps.  Verify independently
   that `RegularToStandardGh` initializes the complete reconstructed spatial
   metric before any `d0gamma` contraction.  Check all `g_ab`, `Phi_iab`, and
   `Pi_ab` formulas for non-diagonal metric, shift, and shift-gradient data.
3. The non-diagonal algebra, geometry, and nonlinear RHS oracles.  Determine
   whether they are genuinely independent of production expressions and
   whether any shared mistake could make them pass incorrectly.
4. Compatible-gradient evolution and repair of `Q=Dgtilde`, `X=Dchi`,
   `a=Dalpha`, and `B=Dbeta` across ordinary updates, restart, static
   refinement, and dynamic regrid.
5. History/checkpoint diagnostics, especially the `alpha >= 0.25` mask copied
   from the Z4c lapse-excision convention and the physical inverse-metric
   momentum contraction `chi*gtilde^{ij} M_i M_j`.  Check global and fixed
   near-region normalization and history/checkpoint agreement.
6. Exact and robust Minkowski tests, fourth-order uniform linear-wave
   convergence, real-SMR wave convergence, identical-Z4c puncture data,
   restart equivalence, and bounded constraint convergence.  Confirm that the
   tests exercise the intended production paths and that thresholds are not
   weakened to hide failures.
7. AMR, boundary, output, checkpoint, and restart plumbing for ownership,
   ghost-zone, prolongation, load-balance, and schema hazards.
8. Documentation claims versus artifacts.  Explicitly distinguish current
   corrected-source local evidence from older Perlmutter evidence that predates
   the non-diagonal RHS, geometric momentum-norm, and standard-GH map fixes.

Use these committed evidence entry points:

- `docs/fo_gh_puncture_formulation.md`
- `docs/fo_gh_puncture_validation.md`
- `docs/fo_gh_remote_review_handoff.md`
- `docs/fo_gh_artifacts/local_corrected_rhs_audit/README.md`
- `docs/fo_gh_artifacts/local_standard_gh_map_audit/`
- `docs/fo_gh_artifacts/perlmutter_20260817/`
- `docs/fo_gh_artifacts/perlmutter_20260817_current/README.md`

The latest focused Serial evidence should show: algebra, tensor, compatible
gradient/robust advection, non-diagonal geometry, and nonlinear RHS unit tests
passing; a finite identical-data puncture through `t=0.2M`; a three-resolution
masked-constraint ladder through `t=0.01M`; and bitwise two-cycle restart
equivalence.  Independently reproduce cheap focused tests if your environment
supports them, but do not call compilation alone GPU qualification.

Return a findings-first report.  For each finding give severity, exact file and
line, the violated equation/invariant or unsupported claim, concrete evidence,
and a proposed regression.  Then provide a requirement-by-requirement evidence
matrix with `proved`, `contradicted`, `incomplete`, or `missing`.  End with a
strict scientific conclusion for uniform and SMR cases separately.  Do not
claim long puncture stability: current-source GPU qualification, corrected
uniform/SMR long ladders, and evolution through `20M`, `50M`, and `100M` remain
unperformed.  Also state explicitly that the observed real-SMR wave order is
about 1.5, not fourth order.
