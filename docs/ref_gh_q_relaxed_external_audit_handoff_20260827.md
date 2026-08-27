# Ref-GH q-relaxed controller: external audit handoff

Date: 2026-08-27

## Repository state

- Repository: `git@github.com:HengruiZhu99/athenak.git`
- Branch: `codex/ref-gh-q-relaxed-controller-20260826`
- Baseline commit: `70e1579cf12a00f9d3a12fb2fe9874d3b344c3fc`
- Evidence commit before this handoff: `76bba8a86d410e528f7e77e70937782b1ec05e35`
- Scope: standalone vacuum Ref-GH only. Fluid coupling, Kerr-Schild data, and horizon finding are out of scope.

Start with the scientific and implementation report:

- `docs/ref_gh_q_relaxed_controller_20260826.md`

The compact evidence bundle is:

- `docs/fo_gh_artifacts/ref_gh_q_relaxed_controller_20260826/`
- machine-readable status: `docs/fo_gh_artifacts/ref_gh_q_relaxed_controller_20260826/STATUS.json`
- integrity manifest: `docs/fo_gh_artifacts/ref_gh_q_relaxed_controller_20260826/SHA256SUMS`

## Completed evidence

The branch implements the analytic initialization, static reference state, and scalar q-controller foundations, followed by several equation-preserving SYCL portability changes to the dynamic reference-cache execution path.

The following evidence is complete within its stated scope:

- Static controller-off q = 0.9, 1.0, and 1.1 runs at three resolutions reach t = 0.1M.
- In the regular annulus 0.25 <= r/M < 0.375, the metric errors converge at approximately 3.5--3.8 order and common constraint errors at approximately 3.3--3.6 order. The q = 1 case is at roundoff.
- The innermost 0.125 <= r/M < 0.25 metric Linf error is non-monotone. This negative result is retained and is not described as convergence.
- A local prescribed q pulse, 1 -> 0.9 -> 1, remains finite through t = 0.05M. This is only a short smoke test.
- The latest specialized-provider correction passes the local source unit/cache tests and a complete local RK4 evolved cycle. The physical trumpet summary is bitwise identical to the previous path, and the largest reported principal relative difference is 1.45e-15.
- One-tile Aurora source/cache gates pass before the evolved multi-rank failure discussed below.

The compact static convergence table is at:

- `docs/fo_gh_artifacts/ref_gh_q_relaxed_controller_20260826/static_t0p1_convergence.md`
- `docs/fo_gh_artifacts/ref_gh_q_relaxed_controller_20260826/static_t0p1_convergence.json`

The latest local equivalence evidence is at:

- `docs/fo_gh_artifacts/ref_gh_q_relaxed_controller_20260826/local/q_provider_split_equivalence/README.md`

## Unresolved Aurora blocker

The full-output evolved Ref-GH cycle has not passed on Aurora PVC. Five bounded eight-rank attempts terminate with a Level Zero `NotPresent` GPU write failure at the PDP level after the one-tile source/cache gate passes:

- 8785680: initial evolved failure.
- 8785718: still fails after splitting the boundary projection.
- 8785796: staged q reductions localize the dynamic cache rebuild to stage 2.
- 8785833: fence instrumentation localizes the first fault to the q-controlled provider profile launch.
- 8785882: one jet per work item still fails, disproving simultaneous three-jet construction/capture as a complete explanation.

The latest failure bundle is:

- `docs/fo_gh_artifacts/ref_gh_q_relaxed_controller_20260826/aurora/job_8785882_failed/README.md`

Large Aurora outputs and complete scheduler logs remain outside Git at:

- `/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_q_relaxed_20260826_e403baf_v1`
- `/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_q_relaxed_20260827_f184fcde_v2`
- `/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_q_relaxed_20260827_bd40d98b_v1`
- `/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_q_relaxed_20260827_5ac38180_v1`

No Aurora jobs from this campaign were queued or running when this handoff was prepared.

## Claim boundary

The evidence does not establish an evolved moving-reference PVC cycle, closed-loop q relaxation, restart continuity for the controller, long-time trumpet stability, a three-resolution evolved convergence result, or production readiness. The local and static results must not be promoted into any of those claims.

## Requested read-only audit

Please perform a detailed read-only formulation and performance-portability audit of this branch. In particular:

1. Trace the ownership, allocation, extents, initialization, and lifetime of every view read or written by the q-controlled reference-provider profile kernel.
2. Look for an earlier asynchronous out-of-bounds write whose first observed failure is the provider launch, rather than assuming the reported launch itself is the corrupting kernel.
3. Compare execution-space, view-capture, team-policy, scratch-memory, and fence patterns directly with mature Aurora-working Z4c code in the same repository.
4. Audit whether repeated provider launches expose a Kokkos/SYCL portability issue, including device-callable object size, private storage, nested tensor temporaries, and kernel argument capture.
5. Verify that the q-controller stage ordering and reference-cache rebuild preserve the documented mathematical algorithm and that the portability refactors did not change equations.
6. Review the static convergence calculation, puncture-adjacent exclusions, and the distinction between regular-annulus convergence and the non-monotone inner-shell result.
7. Recommend the smallest discriminating correction or reproducer. Do not weaken numerical thresholds or infer scientific qualification from a build, a one-tile source gate, or the local CPU/GPU cycle.

Suggested checkout and integrity verification:

```bash
git fetch origin codex/ref-gh-q-relaxed-controller-20260826
git switch --detach origin/codex/ref-gh-q-relaxed-controller-20260826
cd docs/fo_gh_artifacts/ref_gh_q_relaxed_controller_20260826
sha256sum -c SHA256SUMS
```

