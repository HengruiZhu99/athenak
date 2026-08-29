# Read-only remote audit prompt

Please perform a detailed read-only scientific and code audit of
`HengruiZhu99/athenak`, branch
`codex/ref-gh-ordering-gauge-discriminator-20260829`.

Start from the pushed branch tip and verify its ancestry, clean artifact
manifests, and the claims in
`docs/ref_gh_ordering_gauge_discriminator_20260829.md`. Do not modify the
branch or rerun expensive jobs.

Audit these questions independently:

1. Does the A--E matrix support the bounded classification
   `GAUGE-DRIVER COUPLING DEFECT ISOLATED`? Confirm that A/B are fresh finite
   controls through 5M, E is finite through 3M, and C/D share the same failure
   time and exponential rate. Keep the failed A/B restart attempts separate.
2. Re-derive the gamma1=-1 compatible, standard, and standard-plus-driver
   principal matrices. Check the script
   `scripts/ref_gh/analyze_ordering_principal_symbols.py`, especially the
   repeated-eigenspace-invariant condition number and the claimed loss of ten
   eigenvectors at `beta_s=alpha` for compatible ordering.
3. Review `gauge_driver.hpp`, `physical_gauge_target.hpp`,
   `reference_gauge_baseline.hpp`, `reference_projection.hpp`,
   `stationary_gauge_data.hpp`, `standard_gh_source.hpp`, the relevant paths
   in `ref_gh_calcrhs.cpp`, stationary initialization, and physical boundaries.
   Compare the driver signs and characteristic coupling directly with
   Lindblom--Szilagyi arXiv:0904.4873 Eqs. (9), (11), and Appendix B.
4. Verify the three-resolution cycle-zero residual extraction in
   `scripts/ref_gh/analyze_ordering_gauge_phase67.py`. Decide whether the
   measured `r^-5` driver and `r^-7.27` KO envelopes are correctly described
   as moving-`r/h` binary64 cancellation rather than fixed-coordinate
   convergence.
5. Challenge the proposed next equation-preserving repair: directly evaluate
   regular residuals such as `F-Fref` and make exactly matched q=1 deltas
   bitwise zero while retaining the generic 1171-Real path as an independent
   oracle. Identify any hidden equation change or missing all-61 gate.
6. Search for an alternative explanation for the inner exponential mode,
   including a lower-order instability of the 1+log/Gamma target, an incorrect
   same-stage `d_t Hhat`, `dtTheta`, frame-motion sign, or boundary projection.
   Treat hypotheses as hypotheses and cite exact source lines and artifacts.

Primary compact evidence is under
`artifacts/ref_gh_ordering_gauge_discriminator_20260829/`. In particular:

- `phase23_aurora_8790897/`: full fixed-point decomposition and A--D matrix;
- `phase6_aurora_8790932/`: Case E and preserved restart failures;
- `phase67_aurora_8790947/`: fresh A/B through 5M and the residual ladder;
- `analysis/`: growth fits, chi-beta regions, principal-symbol tables, and
  fixed-point scaling.

Return findings ordered by severity, each with exact file/line or artifact
evidence. Then give a claim audit, a formulation audit, a numerical-method
audit, and a minimal next-test recommendation. Do not call the gauge-enabled
trumpet stable or convergent and do not recommend performance work before the
formulation gate is repaired.
