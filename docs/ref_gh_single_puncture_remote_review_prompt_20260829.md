# Remote review prompt: Ref-GH single-puncture robustness failure

Perform a detailed, read-only scientific and code audit of:

- Repository: `https://github.com/HengruiZhu99/athenak.git`
- Branch: `codex/ref-gh-single-puncture-robustness-20260829`
- Campaign evidence commit before this prompt: `8c959059283a14720670f05909ad5aaefa7e57a0`
- Exact campaign base: `53f70903c82f2f8670df7aad6aa14e7ef646ad82`
- Frozen production source reference: `a09caf707f88d9fb6ca71f9abf62c9302fde3bac`

The final branch tip supplied with this prompt should differ from `8c959059`
only by addition of this review prompt.  Record the full checked-out commit and
verify that it is the remote branch tip before reviewing.

Do not edit, commit, push, submit jobs, or rerun expensive simulations.  Do not
broaden scope to fluid coupling, Kerr-Schild data, horizon finding, p control,
wormhole-to-trumpet evolution, moving centers, SMR/AMR, new boundary
conditions, or binaries.  Treat all existing scientific claims as hypotheses
to verify against committed evidence.

## Primary question

Audit why the frozen analytic radial-q Ref-GH implementation fails the exact
matched q=1 stationary-trumpet evolution at cycle 330, `t=1.123484M`, on both
tested MPI decompositions.  Determine what is directly established, what is
only inferred, and what additional minimal discriminator would distinguish a
mathematical formulation instability from an equation/code implementation
error.

Do not call long-time robustness, convergence, closed-loop q control, restart
equivalence, or damping robustness established.  The campaign's controlling
classification is `NOT ESTABLISHED` unless you find a concrete evidence or
analysis defect that invalidates the failure result.

## Required audit

### 1. Provenance and source freeze

Run at least:

```bash
git status --short --branch
git rev-parse HEAD
git rev-parse origin/codex/ref-gh-single-puncture-robustness-20260829
git merge-base 53f70903c82f2f8670df7aad6aa14e7ef646ad82 HEAD
git diff --exit-code a09caf707f88d9fb6ca71f9abf62c9302fde3bac -- src
git submodule status
```

Confirm whether production `src/` is genuinely byte-identical to `a09caf70`.
Classify separately the added input, scripts, documentation, and artifacts.
Report any hidden source, submodule, generated-header, parameter, or build-cache
drift that would undermine the comparison.

### 2. Phase-0 qualification evidence

Review:

- `artifacts/ref_gh_single_puncture_robustness_20260829/phase0_local/`
- `artifacts/ref_gh_single_puncture_robustness_20260829/phase0_aurora_8790831/`
- `inputs/ref_gh/ref_gh_single_puncture_robustness.athinput`

Verify the deterministic regeneration evidence and the coefficient, expanded
radial, geometry, mixed-jet gauge, compact boundary, all-61 compatible/standard
RHS, production-cache, and exact-Minkowski gates.  Verify Aurora job 8790831's
exit status, eight distinct PVC mappings, analytic allocation, GPU-aware MPI,
and one-/eight-rank comparison.  Check that none of these bounded gates is being
misrepresented as long-time stability evidence.

### 3. Telemetry implementation

Audit:

- `scripts/ref_gh/analyze_single_puncture_health.py`
- `scripts/ref_gh/compare_single_puncture_decompositions.py`
- the live Ref-GH/user/common-ADM history definitions in `src/`

Check every history-column mapping, normalization, volume division, RMS/Linf
interpretation, relative-metric eigenvalue/lapse calculation, inverse-metric
construction, coordinate reconstruction, and q-estimator algebra.  Confirm
that the puncture mask discards every point whose FD4 plus dissipation stencil
can overlap the puncture, and identify any off-by-one or geometric ambiguity.

Pay special attention to the fact that each failed run produced only a t=0
binary64 health snapshot because it died before the 2M output cadence.  Ensure
that `all_pass=true` in the health JSON is not used as positive-time evidence.

### 4. Reproduce the compact comparison

Using only committed artifacts, rerun:

```bash
python3 scripts/ref_gh/compare_single_puncture_decompositions.py \
  --left artifacts/ref_gh_single_puncture_robustness_20260829/phase2_attempt_8790836 \
  --left-label phase2_matched_q100_h24 \
  --right artifacts/ref_gh_single_puncture_robustness_20260829/phase2_discriminator_8790840 \
  --right-label phase2_matched_q100_h24_r12 \
  --output /tmp/refgh_decomposition_review.json \
  --tolerance 5e-12
```

Independently verify:

- both logs last report cycle 330, `t=1.123484M`, then the invalid-effective-
  timestep fatal error;
- PBS exit status 143 in jobs 8790836 and 8790840;
- all eight history streams have matching shapes/times;
- global conditioned Linf agreement is approximately `1.573e-13`;
- the reported GH/reduction/curl/physical-error growth values and exponential
  e-folding fits are calculated correctly;
- the performance arithmetic for 12 versus 216 ranks is correct and is not
  overinterpreted as a profiler result.

Check whether comparison of histories alone is sufficient to call the failure
decomposition-independent, and state any limitation from not having positive-
time binary field snapshots.

### 5. Formulation and code review

Trace the exact q=1 stationary path through initialization, reference
construction, gauge baseline/subtraction, scalar-wave source, Pi/Phi updates,
gamma0/gamma2 terms, dissipation, physical boundary projection, and timestep
calculation.  Compare the implementation with the standard gamma2-damped
first-order GH equations and with the mature Z4c execution conventions where
relevant to portability or diagnostic ordering, without proposing a change to
the underlying physical problem.

Center the review on:

- whether q=1 should make the relative state an exact fixed point in binary64;
- the initial RHS Linf `5.84253e-11`, component 10, radius about 1.423M;
- signs/index placement in compatible Phi evolution and reduction/curl
  propagation;
- gamma0/gamma2 damping signs and characteristic fields;
- moving/reference-frame spin, connection, curvature, and source-frame terms
  even though qdot=0;
- gauge-driver target, reference subtraction, theta/dtTheta handling, and
  whether a nominally stationary q=1 state can seed a growing gauge mode;
- boundary projection and ghost-fill ordering across RK stages;
- the fail-closed timestep calculation and which state/speed becomes invalid;
- whether the rapid Q/Delta/source-frame growth is cause, consequence, or only
  a diagnostic correlate;
- whether the near-region versus regular-annulus histories justify an
  inner-region localization claim;
- whether failure before the simple face-to-r<1 estimate of about 1.63M
  materially weakens an outer-boundary explanation.

Do not label a suspected cost center or equation term causal without direct
evidence.  Provide file-and-line references for every concrete code finding.

### 6. Artifact/provenance integrity

Review:

- `docs/ref_gh_single_puncture_robustness_20260829.md`
- `artifacts/ref_gh_single_puncture_robustness_20260829/phase2_attempt_8790836/`
- `artifacts/ref_gh_single_puncture_robustness_20260829/phase2_discriminator_8790840/`
- `artifacts/ref_gh_single_puncture_robustness_20260829/phase2_decomposition_comparison.json`
- `artifacts/ref_gh_single_puncture_robustness_20260829/aurora_jobs_final.txt`

Check job IDs, queues, node/rank/tile mappings, compiler/Kokkos/SYCL/MPI
configuration, executable/input hashes, exact command lines, checkpoint status,
and the distinction between committed compact evidence and uncommitted large
Aurora payloads.  Note that line-ending/trailing-space normalization may make a
committed text artifact differ bytewise from the remote original; use the
provided remote and committed hash manifests according to their stated scope.

## Deliverable

Return a read-only audit report organized as:

1. **Executive verdict**: whether the `NOT ESTABLISHED` classification and
   Phase-2 stop are supported.
2. **Findings by severity**: correctness/formulation issues first, then evidence
   and tooling issues, each with file-and-line citations.
3. **Requirement matrix**: Phase 0 through Phase 8, marked passed, failed,
   conditionally not executed, or unsupported.
4. **Independent numerical checks**: exact recomputed values and any mismatch
   with the committed report.
5. **Causal assessment**: confirmed facts, strongest hypotheses, counterevidence,
   and unresolved ambiguity.
6. **Minimal next discriminator**: one narrowly scoped, equation-preserving test
   that maximally separates the leading hypotheses.  Do not recommend a broad
   parameter sweep or threshold weakening.
7. **Claim corrections**: exact replacement language for anything overstated.

Be adversarial and evidence-driven.  A clean build, local oracle, bounded PVC
cycle, or process exit alone is not scientific qualification.
