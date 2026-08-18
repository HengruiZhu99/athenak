# Prompt for the remote review agent

Audit the latest pushed HEAD of branch
`codex/fo-gh-puncture-driver-20260817` in AthenaK. Record full commit, branch,
dirty state, and Kokkos gitlink. Review code and paused evidence only: do not
launch a campaign, alter another agent's jobs/directories, or work on fluids,
Kerr-Schild data, or horizon finding.

Read first:

- `docs/fo_gh_artifacts/perlmutter_20m_20260817_partial/README.md`
- `docs/fo_gh_artifacts/perlmutter_20m_20260817_partial/FORMULATION_CODE_REVIEW.md`
- `docs/fo_gh_artifacts/perlmutter_20m_20260817_partial/analysis/`
- `docs/fo_gh_puncture_formulation.md` and `docs/fo_gh_puncture_validation.md`
- `src/fo_gh/`, `src/pgen/fo_gh/`, `src/coordinates/adm_constraints.cpp`, and
  relevant history, AMR, boundary, restart, and driver paths

The central observation is timestep collapse at 3.431611M (dx=1/16),
3.024995M (dx=1/24), and 2.658676M (dx=1/32), after valid 2M restarts. Treat a
formulation defect as a leading hypothesis, not fact. Boundary arrival cannot
explain onset on `[-32,32]^3`. No Z4c production case ran.

One independent code defect has since been confirmed and repaired:
`src/coordinates/adm_constraints.cpp` raised indices before all lower-index
tensor components were initialized, while history `fmax` masking could hide a
NaN M2 as zero. Review that fix and the non-diagonal/curvilinear flat-space
manufactured regressions in `src/pgen/fo_gh/geometry_unit.cpp`; the original
ordering fails 64 cells and the repair passes. Treat every common-ADM H/M
history in the paused artifact bundle as invalid; the operator is diagnostic
only and this defect does not explain the evolution collapse.

Perform two coupled reviews:

1. Independently derive the regularized FO-GH equations, constraint additions,
   gauge targets/driver, characteristic structure, and constraint propagation.
   Do not accept code-mirroring tests as independent validation. Identify any
   sign, coefficient, index, rescaling, or hyperbolicity defect and propose a
   manufactured or Fourier-mode regression.
2. Conduct a findings-first code review of compatible gradients, advection,
   AMR, determinant/trace handling, timestep, boundaries, restart, and
   diagnostics. Confirm that the common-ADM fill-then-raise repair eliminates
   all uninitialized reads and that the manufactured regression is independent
   and sensitive. Look for analogous initialization/order defects elsewhere.

Return severity-ordered findings with exact file/line, violated invariant,
evidence, impact, and regression; an observation/inference/hypothesis table; an
equation-to-code matrix; an evidence matrix marked proved/contradicted/
incomplete/missing; ranked root causes; and the smallest safe next experiment.
State strictly that the t=20M FO-GH/Z4c comparison remains incomplete.
