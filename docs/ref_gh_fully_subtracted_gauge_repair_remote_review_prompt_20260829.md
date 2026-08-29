# Remote review prompt: Ref-GH fully subtracted gauge repair

Review `HengruiZhu99/athenak` in read-only mode on branch
`codex/ref-gh-fully-subtracted-gauge-repair-20260829`. Verify the branch tip
reported by the requester before beginning. The repair branch was forked from
frozen discriminator commit `223947486ac4498bab2e197feca56462c77e6d76`.
The residual-production source checkpoint is
`ab30fa963f5d1d7ce54748ffb287c91c87705153`; the branch tip also contains
equation-preserving validation-kernel staging, a source-unit-only SYCL
compile-topology correction, a test-only normal-value predicate probe, and
compact Phase-6 evidence. Verify every intervening commit from the repository
history rather than assuming the source checkpoint is the tip.

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
7. Audit the stationary-trumpet coefficient analysis in
   `scripts/ref_gh/analyze_fully_subtracted_trumpet_asymptotics.py`. Independently
   check the lapse exponent, the 26 reported coefficient powers, the
   high-precision identities, and the epsilon-sensitivity test. Pay particular
   attention to whether the lower-order maps are genuine residual coefficients
   rather than canceled pure-reference terms, and whether the stated energy
   estimate limitations are neither too weak nor too strong.
8. Audit `src/ref_gh/exact_matched_state.hpp` and its use in
   `src/pgen/ref_gh/stationary_trumpet.cpp` and
   `src/ref_gh/ref_gh_tasks.cpp`. Confirm that exact zero fills are selected only
   for the precisely matched, static, uncontrolled, unprescribed `q=1` case;
   that every nearby or moving case takes the general path; and that initial and
   physical-boundary data remain consistent.
9. Verify the claim boundary: the new residual driver/Einstein-source paths
   are dispatched only under the strict static, uncontrolled, unprescribed
   `q=1` predicate. Audit that production predicate and both the generic and
   analytic dispatches in `src/ref_gh/ref_gh_calcrhs.cpp`. General moving and
   controlled production modes must remain on the legacy path at this
   checkpoint.
10. Check that standard Phi ordering remains the production candidate and that
   this lower-order rewrite does not silently change the standard first-order GH
   principal part. Treat compatible ordering only as an oracle/research mode.
11. Audit the portability corrections in `src/pgen/ref_gh/source_unit.cpp` and
    `src/CMakeLists.txt`. Confirm that staging preserves the original all-61
    arithmetic and conditioned comparison, `per_kernel` is confined to the
    validation translation unit under SYCL, and the `min()` predicate probes
    test exact rejection without altering the production predicate.

Evidence to inspect:

- `artifacts/ref_gh_fully_subtracted_gauge_repair_20260829/phase0_local/`
- `artifacts/ref_gh_fully_subtracted_gauge_repair_20260829/phase2_local/`
- `artifacts/ref_gh_fully_subtracted_gauge_repair_20260829/phase3_local/`
  (the preserved red gate)
- `artifacts/ref_gh_fully_subtracted_gauge_repair_20260829/phase3b_local/`
- `artifacts/ref_gh_fully_subtracted_gauge_repair_20260829/phase3c_local/`
- `artifacts/ref_gh_fully_subtracted_gauge_repair_20260829/phase4_local/`
- `artifacts/ref_gh_fully_subtracted_gauge_repair_20260829/phase5_local/`
- `artifacts/ref_gh_fully_subtracted_gauge_repair_20260829/phase6_local/`
- `artifacts/ref_gh_fully_subtracted_gauge_repair_20260829/phase6_aurora_8791211_failed/`
- `artifacts/ref_gh_fully_subtracted_gauge_repair_20260829/phase6_staged_all61_local/`
- `artifacts/ref_gh_fully_subtracted_gauge_repair_20260829/phase6_aurora_8791265_failed/`
- `artifacts/ref_gh_fully_subtracted_gauge_repair_20260829/phase6_aurora_8791292_predicate_failed/`

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
- stationary trumpet lapse exponent:
  `p = 1.091297104795417177142734699770404899977`;
- all 26 fitted Phase-4 powers agree with their derived powers to at most
  `1.323972251e-4`, against a `5e-3` gate;
- Phase-4 independent identity maximum: `2.3738919364e-65`, against `1e-45`;
- the `1e-30` and `1e-24` high-precision differencing runs have identical
  40-digit JSON/TSV results;
- Phase-5 exact initialization: stored-field, stored-Hhat, and stored-theta
  Linf errors are exactly zero; physical reconstruction errors are at most
  `3.33067e-16`;
- Phase-6 direct generated Kref conditioned error: `1.15471e-15`;
- Phase-6 all-61 legacy-generic versus fully-subtracted compact RHS error:
  `4.13003e-14` over 4320 samples, including moving-reference `dtTheta` and
  compatible plus STANDARD Phi ordering;
- Phase-6 exact STANDARD production gauge sectors: bitwise zero;
- Phase-6 remaining total Pi RHS: `5.681872526233013e-14`, entirely in the
  existing covariant-vacuum source at its maximum;
- non-tautological all-radius physical-target diagnostic: `6.02413e-10`.
- Aurora job `8791211`: compile gate failed before device execution because
  IGC segfaulted while lowering the monolithic all-61 kernel (`icpx` exit 245);
- staged generic/compact/comparison kernels: local Serial regression retains
  the identical `4.13003e-14` result;
- Aurora job `8791265`: the staged source still failed in IGC because Kokkos'
  `device-code-split=off` kept the full validation TU in one SPIR-V module;
- source-unit-only `per_kernel` device splitting: implemented as a compile
  topology correction;
- Aurora job `8791292`: the full image compiled and 12 ranks mapped to distinct
  PVC tiles, but the first host-side predicate test failed before any oracle
  kernel because Intel's active mode flushed its `denorm_min()` false-case
  probe to zero;
- the follow-up replaces only those two test probes with `min()`, the smallest
  positive normal value; the production predicate, equations, task graph, and
  tolerances are unchanged, and the local Serial all-61 result remains
  `4.13003e-14`;
- no Aurora all-61 device equivalence, fixed-point execution, or evolution has
  passed at this checkpoint.

Please return:

1. findings ordered by severity, with commit/file/line references;
2. a formulation verdict for target, driver, and Einstein gauge source
   separately;
3. an oracle-independence and test-coverage verdict;
4. a specific analysis of the `r=0.03M` discrepancy;
5. the minimal equation-preserving corrections, if any;
6. separate verdicts on the Phase-4 coefficient/asymptotics evidence, the
   Phase-5 exact-state implementation, and production residual dispatch;
7. an explicit list of claims supported and not supported by the repository.

Do not claim trumpet convergence, long-time stability, Aurora/PVC
qualification, acceptable performance, or production readiness from this
checkpoint.
