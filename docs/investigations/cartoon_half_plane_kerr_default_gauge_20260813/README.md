# Half-plane Kerr default-gauge failure investigation

This directory is the Git-sized, self-contained review bundle for the
three-resolution Kerr experiment.  It contains the complete investigation
report, all five existing plots in PNG and PDF form, and the parsed constraint,
horizon, near-axis, and termwise-RHS tables.  No new numerical run was made to
create this bundle.

Source identities:

- qualified implementation base: `f95a05802621b76cff2894d562c38df4b0d09661`
- default-off RHS diagnostics: `7ee7ecf327ed755392c27eb4eea4190257438c83`
- branch: `codex/cartoon-half-plane-rhs-diagnostic-20260813`
- Kokkos: `6739bc623081648af9e752b616d9671527922cbf`

The experiment used a pre-collapsed initial lapse, AthenaK's default advective
1+log lapse and direct Gamma-driver shift, O6/RK4, no chi floor, and
`dchi_max=0.02`.  M/32 reached 5M; M/48 failed reproducibly at approximately
4.5604M; M/64 was not run after convergence had already failed.

Start with `investigation_report.pdf`, then inspect
`data/diagnostic_summary.json`.  `REMOTE_REVIEW_PROMPT.md` is ready to send to
an independent remote agent together with this branch or directory.

The full 1.1-GB native restart/binary evidence remains outside Git.  The
diagnostic summary binds the decisive raw inputs by SHA-256, while this bundle
contains the portable derived evidence needed for an independent diagnosis.

