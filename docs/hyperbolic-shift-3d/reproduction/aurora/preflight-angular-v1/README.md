# Cross-face V3 preflight passed

All twelve cases passed on one Aurora debug node with twelve SYCL MPI ranks
on twelve distinct GPU tiles. This establishes component correctness,
four-step evolutions and exact restart handling for the cross-face angular
candidate. It does not establish duration stability or the requested 100 M
puncture transition. The [duration protocol](../stability-angular-minkowski-v1/protocol.md)
is a separate test.

The [preserved preflight protocol](protocol.md) identifies frozen source-v10,
commit `ec0dd5afcf9d489e06c1fdf92e802a6fb8c14679`, and its exact source and
executable hashes. The fresh SYCL/MPI/PVC build used `make -j64` and passed
in 315.53 seconds. Build configuration, compiler/module versions, full source
manifest and logs are retained in [compile-evidence/](compile-evidence/). This binary includes final
algebraic projection, cached scri jets, centered cross-face angular derivatives
and the optional horizon angular-resolution arguments. Later diagnostic-only
reporting changes are not included in this frozen run. J is disabled and
the original gauge coefficients are unchanged.

| Check | Result |
| --- | --- |
| Bulk, exchange, original-scri and algebraic component comparisons | All twelve rank reports passed for each component, with actual SYCL execution. |
| Cross-face angular component | Native rate scaled error 7.8192e-14; dependent-D error 1.8920e-14; null residual 9.9920e-16. Smooth manufactured derivative tests passed, including mixed derivatives at patch corners. |
| Minkowski, n16, four steps | Coordinate t=.002; maximum stored-field deviation 1.0053e-14; H RMS 7.5297e-15; M RMS 2.8493e-15. |
| Pulse direct/restarted | Four direct steps match two steps plus restart and two steps. Complete 39,362,632-byte V3 files, including all four arrays and header, are byte-identical. |
| Isotropic puncture, four steps | t=.002=.006 M; accepted coordinate horizon radius .06294960072; areal radius .66667007086; mass .33333503543; expansion RMS .00032518. |
| Communication benchmark | Twelve ranks, n16/n24/n32, all three routes, both timing modes and mixed sequences passed full receive-array validation. |

The puncture finder uses lmax=12, ntheta=24, nphi=48 and the unchanged .001
expansion tolerance. Its exterior H RMS is .02483017035, measured outside
the sphere at the mean found horizon radius. Surface/ghost null residuals
are at most 8.8818e-16 / 1.2213e-15. These are early-time measurements; four
steps do not demonstrate late-time horizon tracking or spatial convergence.
Minkowski and pulse plots use coordinate time, without a black-hole mass
normalization.

The new cross-face operator changes derivative jets in the scri evolution.
The original owner-face surface-to-ghost value interpolation remains in
place. This is a controlled discretization candidate, not a claim that all
boundary defects are resolved.

Job 8983281 finished in PBS state F with exit status zero. All native cases
returned zero without timeout or storage error. The allocation was released.
The campaign footprint at worker completion was 1,279,667,098 bytes, below
the fixed 5 GB cap. [terminal.json](terminal.json) retains selected scheduler
facts without owner/email or allocation-balance fields.

[verification.json](verification.json) independently rechecks all 97 recorded
artifact hashes and lengths, 155 acceptance checks, exact case order, build
and input bindings, checkpoint headers, generated restart inputs, terminal
status and rank/tile placement. It independently compares the complete direct
and resumed checkpoint files. The source manifest's 1,116 non-Kokkos entries
also match the frozen Git commit in an independent read-only audit. The
original [admission](admission.json) binds the earlier
[verification-at-admission.json](verification-at-admission.json); its exact
verifier bytes are preserved in [verify-at-admission.py](verify-at-admission.py).
The later verifier adds stricter case/build-coverage and exit-status checks;
the same evidence passes all of them. No bound admission was rewritten.

The [communication analysis](communication-analysis/README.md) retains its complete
report once as verified gzip data. Direct device MPI is faster than explicit
staging in every volume/mixed median; ordinary host buffers reproduce the
sharp large-message cost increase. The existing transport is retained.

Five short-preflight checkpoints were subsequently purged from Aurora only
after complete local archival and hash verification, freeing 196,813,160
logical bytes. The exact [manifest](purge-angular-preflight-v1-manifest.json)
and [receipt](purge-angular-preflight-v1-receipt.json) identify those files.
No current duration output was removed.

The small [reports and histories](output/), rank-00 component logs, worker
record, plan and binding are retained. Full checkpoints and omitted rank
logs remain in the independently verified local raw campaign; hashes remain
in the worker record. No full 3D payload or executable is committed here.
To replay the checks against that raw campaign and its retained build files,
run from the repository root with a fresh output path:

```sh
python3 docs/hyperbolic-shift-3d/reproduction/aurora/preflight-angular-v1/verify.py \
  --output output/hyperbolic-core-shell/aurora/preflight-angular-reverification.json
```

This command reads evidence only. The [assessment](assessment.json) contains
compact scientific values; the original reports remain authoritative.
