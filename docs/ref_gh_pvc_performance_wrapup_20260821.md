# Aurora Ref-GH PVC performance wrap-up

Status frozen on 2026-08-21 at commit `9da381a3311bf2ea133bd705994565a7d56fb86a`.
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

## Outcome

The equation-preserving static-reference validity change raised the matched
one-PVC-tile Ref-GH benchmark from `1.126323e5` to `3.033790e5` active
zone-cycles/s, a `2.6935x` speedup over the immediately preceding traversal
build and a `2.6374x` speedup over the original bounds-off baseline
(`1.150275e5`). The checked full-output evolved-cycle gate, native/common
histories, restart, and time-dependent-reference stage-time check passed after
the change.

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
| Mature Z4c control | 8773273 | `7.135622e6` | `62.03x` baseline Ref-GH; `23.52x` static-cache Ref-GH |

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
kernels were scalar-source RHS at 72.02 percent and Pi RHS at 12.79 percent;
the static provider was constructed once in the 20-cycle profile rather than
73 times. These percentages include global profiler fences and are for
localization, not production throughput.

## Correctness gate

The checked static-cache job reported:

- conditioned reference-cache oracle Linf: `2.36929e-14`;
- stationary initial RHS Linf: `1.10048e-16`;
- one-cycle field Linf: `9.992007e-16`;
- one-cycle native-constraint Linf: `2.461044e-14`;
- valid positive-time native/common histories, restart, and fault scan;
- static provider builds in the 20-cycle profile: 1;
- time-dependent provider builds in the focused evolved cycle: 5;
- time-dependent evolved maximum error: `9.992007e-16`.

The final checked and performance executables have SHA-256 hashes
`0ce1d09bd556a858828b836e9ed63d35576e6894b91442fc6cffb369768795c3`
and `c2810867854b387ee3c74b98bffb3263cd0dd3bc8b79ccbc172b251d489943af`,
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

- Ref-GH is still `23.52x` slower than the matched mature-Z4c control on this
  one-tile benchmark.
- The scalar-source RHS now dominates the synchronized profile and needs a
  direct equation-preserving code audit.
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
that root.
