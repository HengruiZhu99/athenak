# Aurora Ref-GH PVC performance wrap-up

Status wrapped on 2026-08-21.  The final tested source authority is commit
`fc2d61adf3559b5c09707a50cb6ddbc2b612bc26`.
No Aurora job owned by this campaign was queued or running at wrap-up. This is
an interim performance checkpoint, not completion of the controlling Goal Mode
campaign and not a new scientific qualification.

## Resumed diagnostic split checkpoint

The campaign resumed at commit
`338b18f0e075b1ddf842acf67aca8f12762958b4`.  Production cache rebuilds now
honor `include_diagnostics=false` by skipping reference Ricci; checked cache
validation still requests and verifies it.  Aurora jobs `8773745` and
`8774035` established the call-count distinction:

| Case | Provider builds | Ricci builds |
|---|---:|---:|
| Static profile with initialization diagnostics | 1 | 1 |
| Time-dependent checked/oracle cycle | 5 | 5 |
| Time-dependent validation-off production cycle | 5 | 2 |

The two production Ricci calls are initial and terminal history, not RK-stage
cache updates.  The checked full-output stationary result remained
`9.992007e-16` in field Linf and `2.461044e-14` in native-constraint Linf; the
time-dependent production cycle also ended at `9.992007e-16`.  No Level Zero
fault or nonfinite state occurred.  The measured bounds-off rate was
`3.550192e5` active zone-cycles/s, but the `1.1702x` difference from the prior
job is not assigned to this setup-only change because the timed scalar-source
loop is unchanged.

Both PBS jobs completed their numerical payloads but returned status 1 in
offline postprocessing: the first assertion ignored the initialization
diagnostic and the follow-up ignored terminal history.  The corrected compact
table and complete provenance are under
`docs/fo_gh_artifacts/ref_gh_pvc_performance_20260821/diagnostic_split_8773745_8774035/`.
No campaign job remains queued or running.

## Rejected symmetry-compression experiment

An exact-symmetry workspace compression was attempted and rejected by the
unchanged checked oracle before benchmarking.  Full metric/inverse-metric
compression at `056aafcd` produced conditioned spin-derivative disagreement
`5.94778e-13`; retaining only inverse-metric compression at `dfdf9fee` produced
`9.96623e-13`.  Both exceed the existing `5.68434e-14` threshold.  No tolerance
was weakened and no performance result from either build was accepted.

The likely mechanism is floating-point accumulation-order sensitivity between
separately evaluated analytic tensor orientations near the trumpet puncture;
this is an interpretation, not a formulation defect.  Commit `47d0ad99`
reverts both compressions and is source-identical to the last passing Ref-GH
state.  Compact negative evidence is under
`docs/fo_gh_artifacts/ref_gh_pvc_performance_20260821/symmetry_rejected_8774093_8774143/`.

## Final lean-source checkpoint

Commit `fc2d61ad` adds a production-only covariant scalar-source evaluator.  It
uses the same equations and loop order as the full diagnostic/oracle evaluator,
but computes only the ten evolved symmetric source components and does not
materialize five final diagnostic sector matrices.  The full implementation is
retained unchanged as the independent oracle.

Aurora job `8774182` passed on one PVC tile with PBS exit status zero.  The
source oracle compared 1000 flat and 128 nonflat manufactured states, with
maximum errors `5.55112e-17` and `3.33067e-16`.  The checked stationary cache,
full-output evolved cycle, native/common histories, restart, time-dependent
stage-time cache, and performance cases also passed.  The final bounds-off rate
was `9.592259e5` active zone-cycles/s: `2.7019x` the immediately preceding
diagnostic-split measurement and `8.3391x` the original Ref-GH baseline.  It is
still `7.4389x` slower than the matched mature-Z4c control.

## Outcome

The equation-preserving static-reference and lean-source changes raised the
matched one-PVC-tile Ref-GH benchmark from the original `1.150275e5` to
`9.592259e5` active zone-cycles/s, an overall `8.3391x` improvement.  The
checked full-output evolved-cycle gate, native/common histories, restart,
time-dependent-reference stage-time check, and independent full-source oracle
passed after the final change.

The implementation does not change the GH equations, 50-field state, spatial
discretization, RK4 integrator, CFL, dissipation, boundary conditions, or
diagnostic definitions. It records whether a reference provider is
time-dependent and avoids reconstructing a time-independent reference cache at
every RK stage. A time-dependent provider still rebuilt at all five observed
stage times. The component/cell traversal reorder was also retained because it
is portable and layout-consistent, but its measured throughput was `0.9792x`
the original baseline; it is not claimed as a performance improvement.

## Measured evidence

| Build/case | Aurora job | Active zone-cycles/s | Relative result |
|---|---:|---:|---:|
| Ref-GH checked baseline | 8773102 | `7.835008e4` | bounds checking enabled |
| Ref-GH performance baseline | 8773102 | `1.150275e5` | reference |
| Ref-GH traversal reorder | 8773363 | `1.126323e5` | `0.9792x` baseline |
| Ref-GH static cache | 8773414 | `3.033790e5` | `2.6935x` traversal; `2.6374x` baseline |
| Ref-GH diagnostic split | 8773745 | `3.550192e5` | setup-only change; no causal speedup claim |
| Ref-GH lean production source | 8774182 | `9.592259e5` | `2.7019x` preceding; `8.3391x` baseline |
| Mature Z4c control | 8773273 | `7.135622e6` | `7.4389x` final Ref-GH |

All cases used the same `64^3` active-cell, one-MeshBlock, one-PVC-tile,
four-stage RK4, CFL `0.05`, output-free benchmark shape. The Z4c numerical
warmup and measured case completed; its PBS script returned status 2 only
because the postprocessor used the wrong working directory. The rate is read
directly from the completed AthenaK log and is not inferred from that failed
postprocessing step.

The synchronized baseline profile attributed approximately 66 percent of
completed kernel time to repeated reference construction: metric jets 41.47
percent, reference connection 17.98 percent, coordinate spin derivative 5.22
percent, and remaining reference kernels. After static caching, the leading
kernels were scalar-source RHS at 72.02 percent and Pi RHS at 12.79 percent.
After the lean-source change they were 35.24 percent and 32.42 percent,
respectively, or 67.66 percent combined.  The static provider was constructed
once in the 20-cycle profile rather than at RK stages.  These percentages
include global profiler fences and are for localization, not production
throughput.

## Correctness gate

The final checked lean-source job reported:

- conditioned reference-cache oracle Linf: `2.36929e-14`;
- stationary initial RHS Linf: `1.10048e-16`;
- one-cycle field Linf: `9.992007e-16`;
- one-cycle native-constraint Linf: `2.461044e-14`;
- valid positive-time native/common histories, restart, and fault scan;
- static provider builds in the 20-cycle profile: 1;
- time-dependent provider builds in the focused evolved cycle: 5;
- time-dependent evolved maximum error: `9.992007e-16`.

The final checked and performance executables have SHA-256 hashes
`f242ab90e8495860dfd39e6b4b4a1a8eb7fa0c24379e261838867dfefcaf0e50`
and `c6b0cbc6c259879e384f60dd4456766e53149a11e61c5824e8d2f0168df9ee45`,
respectively. Kokkos used SYCL with `Kokkos_ARCH_INTEL_PVC=ON`, Level Zero was
selected, and GPU-aware MPI was enabled in the recorded environment.

## Scientific status

No new convergence or long-time evolution was run during this performance
iteration. The preceding campaign remains the scientific authority:

- smooth perturbed-trumpet convergence was fourth-order through `t=0.2M` at
  64, 96, and 128 cells per direction;
- the fine stationary run alone remained finite through `t=20M`;
- the medium run stopped near `t=10M`, and the coarse run did not start;
- the two available long-run resolutions had mixed error ratios at matched
  times and did not establish errors converging away.

Therefore convergent trumpet evolution through `t=20M` is **not established**.
The static-cache performance result must not be described as long-time
stability, production qualification, or completion of the three-resolution
campaign.

## Remaining work and blockers

The controlling objective is incomplete. In particular:

- Ref-GH is still `7.4389x` slower than the matched mature-Z4c control on this
  one-tile benchmark.
- Scalar-source and Pi RHS together account for `67.66%` of the synchronized
  profile and remain the next equation-preserving performance-audit target.
- The 1129-Real-per-point reference allocation has not yet been separated into
  production, transient-update, and diagnostic lifetimes; diagnostic
  overcompute and memory traffic remain unresolved.
- Lower-dimensional cache extents and fail-closed `RefGhToADM` output handling
  were audited as risks but not corrected in this iteration.
- No post-optimization 64/96/128 perturbed-trumpet rerun or stationary
  `t=20M` ladder was performed.

Do not restart long evolution until the remaining performance/code audit is
complete and the full checked gate is rerun after any further source change.

## Provenance and artifact locations

Committed compact evidence is under
`docs/fo_gh_artifacts/ref_gh_pvc_performance_20260821/`. It includes build
configuration, source/submodule/compiler/runtime provenance, GPU mapping,
executable hashes, checked logs, throughput tables, and synchronized kernel
profiles. No field dump, restart, executable, or profiler library is committed.

The full Aurora campaign root is:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_pvc_performance_convergence_20260821`

The relevant full run directories are `runs/baseline_8773102.*`,
`runs/z4c_control_8773273.*`, `runs/kernel_profile_8773335.*`,
`runs/iteration_order_8773363.*`, and `runs/static_cache_8773414.*` beneath
that root.  The final passing source/oracle/performance run is
`runs/lean_source_8774182.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.
