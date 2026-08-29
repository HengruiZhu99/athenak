# Remote review prompt: Ref-GH fully subtracted gauge repair

Review `HengruiZhu99/athenak` in read-only mode on branch
`codex/ref-gh-fully-subtracted-gauge-repair-20260829`. Verify the branch tip
reported by the requester before beginning. The repair branch was forked from
frozen discriminator commit `223947486ac4498bab2e197feca56462c77e6d76`.

Please perform a detailed formulation, implementation, test, and evidence
audit of the work through the current branch tip. Do not modify the branch,
open a pull request, launch expensive jobs, or infer production readiness from
unit/oracle results.

Primary review targets:

1. Check the derivation and sign/index conventions of the fully subtracted
   residual gauge equations documented in
   `docs/ref_gh_fully_subtracted_gauge_repair_20260829.md`.
2. Review `src/ref_gh/physical_gauge_target.hpp`,
   `src/ref_gh/gauge_driver.hpp`, `src/ref_gh/residual_gauge_source.hpp`, and
   `src/ref_gh/analytic_radial_q_source.hpp`. Confirm that the new compact
   ordinary-GH Einstein gauge-source evaluator is algebraically equivalent to
   the generic residual oracle, remains equation preserving, and does not
   recreate recursive full reference Christoffel/spin/Riemann work.
3. Inspect the tests in `src/pgen/ref_gh/source_unit.cpp` and the independent
   arbitrary-precision oracle in
   `tst/test_suite/ref_gh/high_precision_trumpet_source_oracle.py`. Look for
   correlated implementations, tautological comparisons, missing terms,
   insufficiently independent truth paths, weak coverage, accidental tolerance
   weakening, and exact-match special cases that could hide an error.
4. Recompute or independently check the static stationary-trumpet identities
   used by `exact_matched_static`: `F_ref = H_ref`, conformal Gamma of the
   reference is zero, and pure reference forcing vanishes for static `q=1`.
5. Audit the cancellation-free construction of upper-index `Delta B`, its
   lowering and derivative product identities, the `J_a` and covariant-source
   assembly, and all frame projections. Pay special attention to time/spatial
   derivative index ordering and signs.
6. Explain the preserved all-radius compact-versus-generic discrepancy
   `3.05105e-12` at `r=0.03M`. Decide whether this is expected binary64
   conditioning or evidence of an algebraic defect. Do not recommend simply
   loosening the tolerance. Suggest an independent high-precision discriminator
   and the coefficient/asymptotics checks required before production dispatch.
7. Verify the claim boundary: none of the new residual paths is yet dispatched
   from production `CalcRHS`; exact matched-state production initialization,
   host/device all-61 equivalence, fixed-point ladders, evolved PVC gates,
   convergence, and long-time stability remain unqualified.
8. Check that standard Phi ordering remains the production candidate and that
   this lower-order rewrite does not silently change the standard first-order GH
   principal part. Treat compatible ordering only as an oracle/research mode.

Evidence to inspect:

- `artifacts/ref_gh_fully_subtracted_gauge_repair_20260829/phase0_local/`
- `artifacts/ref_gh_fully_subtracted_gauge_repair_20260829/phase2_local/`
- `artifacts/ref_gh_fully_subtracted_gauge_repair_20260829/phase3_local/`
  (the preserved red gate)
- `artifacts/ref_gh_fully_subtracted_gauge_repair_20260829/phase3b_local/`
- `artifacts/ref_gh_fully_subtracted_gauge_repair_20260829/phase3c_local/`

Validate each `SHA256SUMS` file and distinguish current passing evidence from
the intentionally preserved negative result. The key current observations are:

- conditioned residual source-unit error: `3.82012e-14`, with unchanged
  `1024 * epsilon_binary64 = 2.27374e-13` tolerance;
- exact matched `q=1` residual source: bitwise zero at all ten sampled radii;
- 80-decimal-digit independent static gauge identity:
  `max |F_ref-H_ref| = 7.595510244205564e-75` and
  `max |conformal Gamma_ref| = 4.761790561071987e-80`;
- all-radius compact/generic diagnostic: `3.05105e-12`, worst at `r=0.03M`;
- raw reconstructed full-driver diagnostic: `1.05256`, explicitly not treated
  as truth because it is cancellation corrupted.

Please return:

1. findings ordered by severity, with commit/file/line references;
2. a formulation verdict for target, driver, and Einstein gauge source
   separately;
3. an oracle-independence and test-coverage verdict;
4. a specific analysis of the `r=0.03M` discrepancy;
5. the minimal equation-preserving corrections, if any;
6. a go/no-go decision for Phase 4 coefficient asymptotics and, separately,
   for production dispatch;
7. an explicit list of claims supported and not supported by the repository.

Do not claim trumpet convergence, long-time stability, Aurora/PVC
qualification, acceptable performance, or production readiness from this
checkpoint.
