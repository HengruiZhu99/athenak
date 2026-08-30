# Remote read-only review prompt

Please perform a detailed, read-only code and evidence audit of:

```text
Repository: HengruiZhu99/athenak
Branch: codex/ref-gh-pvc-regression-repair-20260830
Scientific evidence checkpoint: fb55f78c
```

Fetch the branch with plain Git, verify the checkpoint and its ancestry, and
do not modify the branch or launch jobs. Start with
`docs/ref_gh_pvc_regression_and_evolution_qualification_20260830.md`, then
inspect the associated source commits and compact artifacts. Treat all claims
as hypotheses until they are supported by committed code, logs, histories,
hashes, and provenance.

Audit these questions in order:

1. Does the same-build/same-node W/R evidence establish a source-dependent
   Aurora PVC regression, and does the bisect support `ab30fa96` as the first
   bad production source? Check W `68bc1ee3`, R `3c9a34c8`, the passing
   midpoint, first-bad run, and the RHS/task hybrid evidence.
2. Do the one-rank, send-completion, source-only, and link-only negative tests
   justify classifying the `NotPresent` fault as a device-image/code-layout
   portability issue rather than an MPI-completion or physical-boundary bug?
   Identify any alternative explanation not excluded by the evidence.
3. Review `RankPackedSameLevelVarDataSize()` and its oracle. Confirm that the
   corrected metadata exactly matches pack/unpack behavior for all four
   Z4c-like/multilevel cases, that true multilevel behavior is unchanged, and
   that the fix is real but not misrepresented as the PVC root cause.
4. Review the retained `src/CMakeLists.txt` target compile-and-link setting for
   `-fsycl-device-code-split=per_kernel`. Confirm it is restricted to the
   Athena executable under `Kokkos_ENABLE_SYCL`, excludes Kokkos support
   targets, and changes no Ref-GH arithmetic, PDE, task graph, or numerical
   parameter. Check the compile database/link evidence and all three frozen
   12-tile positive-time runs. Note that all three one-cycle jobs landed on
   the same physical node, while the later scientific job exercised three
   nodes.
5. Independently analyze Aurora job `8791961` from
   `artifacts/ref_gh_pvc_regression_repair_20260830/phase8_aurora_8791961_t3_failure`.
   Confirm whether it is free of Level Zero `NotPresent` errors, whether it
   ends at `t=1.280090959752213M` before the requested `3M`, and whether the
   Hhat/Phi/Pi/Psi/theta/Upsilon RHS fits genuinely reproduce the old fast
   approximately `0.038M` inner mode. Check normalization, fit window,
   R-squared, sample count, extrema, and the maximum-location evidence near
   `r=0.14878M`.
6. Check that the report properly distinguishes the last finite history row
   from the subsequent invalid-effective-timestep failure, and that it does
   not claim stationary-trumpet stability, convergence, 5M/20M/100M
   completion, long-time robustness, or production readiness.
7. Perform a formulation-centered code review of the lower-order
   gauge-coupling path that could generate the observed common exponential
   mode. Focus on the fully-subtracted moving-reference gauge terms, Hhat and
   theta driver coupling, Pi/Phi coupling, gamma0/gamma2 terms, signs,
   index/frame conversions, and reference/physical subtraction consistency.
   Compare against the standard gamma2-damped first-order GH equations and
   against the mature paths already present in the repository. Do not propose
   parameter tuning as a substitute for identifying an equation or
   implementation defect.
8. Report any source/artifact mismatch, stale hash, missing negative control,
   insufficient independent-node sampling, analysis-script defect, or claim
   that exceeds the evidence. Separate confirmed defects from hypotheses.

Return a concise audit with: verified facts; code-review findings ordered by
severity with exact file/line references; evidence gaps; an independent claim
boundary; and the smallest equation-consistent next discriminator. Do not
recommend resuming expensive resolution or long-time evolution until the fast
inner mode has an identified and independently tested correction.

The large restart files were intentionally not committed. If filesystem
access is separately granted, their Aurora location and hashes are recorded
in the Phase 8 README and `restart_sha256.txt`; their absence from Git is not
evidence of a completed continuation.
