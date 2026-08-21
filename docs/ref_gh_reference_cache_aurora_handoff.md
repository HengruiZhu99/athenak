# Aurora Ref-GH reference-cache handoff

Status frozen 2026-08-21 after the requested cancellation of PBS job `8770706`.
This report distinguishes completed evidence from partial campaign results. It
does not claim completion of the three-resolution `t=20M` campaign, long-time
convergence, production performance, or the full Goal Mode campaign.

## Implementation under review

The Ref-GH reference geometry is evaluated at the exact RK stage time by an
explicit `CopyU -> UpdateReferenceGeometry -> CalcRHS` task dependency. A
portable structure-of-arrays cache replaces construction of a large automatic
`ReferenceGeometry` object in each consumer kernel. Provider jets, frame and
coframe data, derivatives, connection, spin connection, curvature, and Ricci
data are produced by component-parallel Kokkos kernels and reused by the RHS,
diagnostics, and histories. The production evolution cache contains 313 Reals
per point; the diagnostic cache 336, provider cache 64, and reusable workspace
416 (1129 allocated Reals per point in total). The monolithic implementation
remains only as a validation oracle. A time-dependent reference-lapse test
checks the stage-time path. The mathematical GH system, fourth-order spatial
algorithm, four-stage RK4, damping, dissipation, and diagnostics were not
changed by this portability work.

Primary implementation commits are `9850a774`, `be40f0ea`, `ed191257`
(subsequently neutralized), and `6eed3d9d`. Campaign/analyzer commits are
`9238b05a`, `13b93916`, `327a5e4b`, and `36e43938`.

## Completed evidence

- Full-output PVC gate, job `8770430`, passed one evolved cycle on one Intel
  Data Center GPU Max 1550 tile. The conditioned cache-oracle error was
  `2.36929e-14`; initial RHS Linf was `1.10048e-16`; at
  `t=0.005118043`, field Linf was `9.992007e-16` and native constraint Linf
  was `2.461044e-14`. Restart, positive-time native/common histories, and the
  fault scan passed.
- Eight-rank mapping used eight distinct PVC tiles:
  `0.0,0.1,1.0,1.1,2.0,2.1,3.0,3.1`. Level Zero selected PVC on every rank.
- The stationary `t=1` ladder, job `8770508`, completed at 64, 96, and 128
  active cells per direction. Its errors are at roundoff/cache-accumulation
  scale and increase slightly with resolution, so it is a stationarity gate,
  not a truncation-convergence result.
- The smooth perturbed-trumpet runs completed through `t=0.2` at 64, 96, and
  128 cells per direction. Sixth-order interpolation to a fixed 32-cubed grid
  inside `r<1` gives dynamic-field orders `4.4013` (L2) and `4.9646` (Linf),
  and native-constraint orders `5.4195` (L2) and `4.1201` (Linf). This is the
  available fourth-order perturbed-trumpet convergence evidence. Binary32 Psi
  output did not converge and is secondary only.
- The fine stationary continuation completed through exactly `t=20`, cycle
  7844. It remained finite with field Linf `1.326694e-11`, native constraint
  Linf `3.584018e-10`, bad-state count zero, and a final effective CFL
  `0.04822058`. One resolution staying finite is not a convergence result.

## Partial and terminated evidence

Job `8770706` ran on node `x4406c2s5b0n0` in the `capacity` queue. At the stop
request it was running; `qdel 8770706` was issued once and PBS confirmed final
state `F`, `Exit_status=271`, wall time `08:09:16`.

- Fine: complete at `t=20.000000`, cycle 7844. Final restart `.00012.rst` is
  retained on Aurora.
- Medium: incomplete. Last log progress is `t=10.00830`, cycle 2940; last
  complete history row is `t=10.00149068506634`; the process then received
  signal 15. The latest complete restart is `.00006.rst`, corresponding to
  `t=8` under the configured two-M checkpoint cadence.
- Coarse: not started. No `dx16` run directory or result was produced.

The medium history remained finite through its last record (`bad-state=0`). At
matched times through `t=10`, however, the two available resolutions do not
show uniform error reduction. At `t=10`, medium/fine L2-squared ratios are
`0.5444` for stationary `PsiError` and `0.4161` for the reduction constraint,
so those measures are larger on the fine grid. GH, Pi, and Phi roundoff-scale
measures are smaller on the fine grid, while Q/Delta Linf are nearly equal.
The fixed-region, unmasked common ADM norms are also slightly larger on the
fine grid: medium/fine is `0.9572` for global H2, `0.9429` for Hinf, `0.9703`
for M2, and `0.9699` for Minf. With only two long-run resolutions and mixed
ratios, convergent trumpet evolution is not established.

## Performance and audit priority

The completed fine case measured `6.237938e5` active zone-cycles/s across eight
PVC tiles. The partial medium trajectory gives approximately `5.476e5`
active zone-cycles/s (`~5.48e5`). This is roughly two orders of magnitude below
expectation and requires a dedicated performance/code audit before further
long runs.

Possible cost centers are hypotheses only: CFL 0.05, four-stage RK4, 50 evolved
fields, the 313-Real staged production reference cache, GH tensor contractions,
ghost-zone overhead, one MeshBlock per tile, four unused Aurora tiles,
synchronization, and kernel-launch overhead. Compiler register/spill warnings
may be relevant but have not been tied causally to the measured rate.

## Failures and scientific limitations

Early PVC gates `8770307`, `8770332`, and `8770356` failed cache-oracle
thresholds; the final equation-preserving conditioned comparison passed in
`8770430`. The perturbed simulation job `8770619` completed its numerical
cases but returned nonzero because Aurora Python 3.6 rejected the analyzer;
analyzer-only job `8770660` then exposed `math.prod`, and `8770685` passed after
the compatibility fixes. These are preserved failures, not hidden retries.

There is no completed three-resolution long-time comparison, no long-time
convergence order, no Z4c comparison in this Aurora gate, and no production
performance qualification. The common ADM histories are deliberately unmasked
and the stationary domain is only `[-2,2]^3`, so the singular puncture dominates
those norms. The fine late-time native reduction/Psi errors show a small upward
trend; its cause has not been established. Do not infer a convergent stationary
trumpet from the perturbed `t=0.2` convergence test.

## Remote-review focus

Audit the staged cache extents/lifetimes and exact stage-time refresh, compare
the execution pattern to mature Z4c, profile the twelve staged reference-update
kernels and Ref-GH contractions, quantify ghost/work ratios, and verify whether
the one-block-per-tile decomposition is responsible for poor occupancy. Keep
performance hypotheses separate from demonstrated causes. Full artifact paths,
hashes, exact launch commands, compact tables, and cancellation evidence are in
`docs/fo_gh_artifacts/ref_gh_reference_cache_aurora_20260821/`.
