# Ref-GH relative-damped wormhole power-lag discriminator

## Claim boundary

This is a bounded, single-resolution diagnostic investigation.  It does not
establish stable or convergent trumpet evolution.  The prescribed case did not
reach `5.2M`; feedback failed closed before completing the reference
transition; and no case supplied the required completed-transition plus `2M`
post-transition dwell.  No exponent feedback was enabled or tested.

The continuation controller for reference activation, `xi(t)`, is distinct
from the exponent corrections `delta_q,delta_p`.  The earlier failed
fixed-core trajectory and the prescribed case here use prescribed `xi(t)` with
all exponent corrections frozen at exact zero.  They are not dynamically
q-controlled runs.

## Pre-existing evidence and sector job 8792829

The old `t=0--4M` prescribed fixed-core history and its last finite
continuation row at `t=4.953843M` were merged before launching new evolution.
Those pre-instrumentation histories do not contain separately measured
`q_phys` and `q_ref`.  The retrospective analysis therefore labels `-e_G/2`
only as a calibrated pure-power proxy for `Delta q`; a radially blended state
need not obey that identity exactly.

Using the analyzer's explicit heuristic—first magnitude above the larger of an
absolute floor and ten times the `t<=0.5M` baseline median—the old proxy crossed
`1e-3` at `0.85245M`.  Curl, reduction, and GH RMS crossed at `2.70349M`,
`2.90130M`, and `3.60076M`.  This ordering is consistent with a mismatch
preceding constraint growth, but is not a direct paired-power measurement and
does not prove causation.

Aurora job `8792829` independently completed the prescribed, STANDARD-Phi
sector discriminator from the same `t=2M` restart through `4.2M`.  Its
lapse-only, shift-only, and pure-wave-map GH RMS values were `5.50968e-3`,
`3.95049e-3`, and `5.12578e-3`, with positive fitted growth rates
`1.15819/M`, `1.56124/M`, and `1.12807/M`.  All localized their largest terms
near `r=0.47--0.54M`.  Neither added relative-lapse nor relative-shift term is
individually necessary for the growth.  That result implicated the prescribed
moving-reference/intermediate state broadly; it did not identify a cause.

## Paired-power diagnostic

The opt-in history diagnostic evaluates

```text
q_loc = -(1/6) X^k gamma^ij partial_k gamma_ij
```

for the physical and reference spatial metrics with the same device kernel,
native-cell list, reference center, weights, and shell membership.  It records
their difference without modifying `xi`, `delta_q`, or `delta_p`.  Every point
whose complete FD4 or KO stencil box can contain the puncture is discarded.
The four deliberately overlapping regions are:

- stencil-safe inner: `2h <= r < 0.30M`;
- fixed-core/blend: `0.30M <= r < 0.60M`;
- outer blend/source-maximum neighborhood: `0.50M <= r < 0.75M`;
- legacy estimator shell: `2h <= r < 8h`.

For physical, reference, and mismatch fits, the output includes weighted mean,
variance, effective sample count, extrema, weighted RMS residual, cell count,
and validity.  Shell populations at `h=M/16` are 264, 3168, 5032, and 1960;
effective sample counts are approximately 243, 2274, 4470, and 1220.

The common pure-power identity passed the unchanged source-unit tolerance with
maximum error `4.44089e-16`.  Both Serial and ASan+UBSan/Kokkos-bounds runs
finished without sanitizer reports.  The pre-existing naive direct-FD
same-shell estimator remains explicitly `FAIL`; its fixed-coordinate variant
remains `PASS`.  The new paired metric-jet diagnostic does not use that naive
coordinate finite difference.

At fresh matched data, all four `q_phys-q_ref` means are at binary64 roundoff.
Their individual `q` variances are nonzero—e.g. the inner and blend reference
means are `1.31519` and `1.07951`—which correctly exposes the blended profile
instead of forcing one global exponent.

## Aurora PVC gate 8792921

Job `8792921` built source commit
`9eff5b524c88cf3c2adf8fc4c219fb9c2e72ed29` on an Aurora compute node with
Kokkos SYCL/PVC and ran one full-output frozen, compatible-Phi RK4 cycle on the
328-block production tree.  Twelve MPI ranks mapped to twelve distinct PVC
tiles.  The executable SHA-256 is
`2a0f208b677c67d63f47820562b55a12d28406856aa64dc7190082830cd64324`.

The cycle reached `t=0.004138311M`, kept `xi=xi_dot=xi_ddot=0` exactly, and
wrote valid paired-power rows.  Initial mismatch means were at most
`1.22e-16`; evolved values remained finite at order `1e-8`.  This qualifies
the diagnostic execution path on PVC only.

## Three fresh compatible-Phi cases: job 8792940

One `debug-scaling` allocation used 36 nodes and 432 distinct PVC tiles.  The
nodes were partitioned into three disjoint 12-node/144-rank host sets.  All
cases used the same fresh isotropic-wormhole data, outer-24M 328-block SMR
tree, `dx_min=M/16`, relative-damped gauge/window, FD4, RK4, CFL 0.05, KO
0.02, boundaries, `0.02M` history cadence, and `0.5M` restart cadence.  All
used compatible Phi ordering and exact
`delta_q=delta_q_dot=delta_p=delta_p_dot=0`.

| case | exact outcome | final xi | GH RMS | reduction RMS | curl RMS |
| --- | --- | ---: | ---: | ---: | ---: |
| frozen | exited 0 at `5.2M`; finite | 0 | `6.03980e-5` | `1.34731e-5` | `1.11908e-4` |
| prescribed tau-8 | scheduler SIGTERM at `4.96109M`; last row finite, incomplete | 0.620136 | `7.81218e-2` | `3.00691e-3` | `1.36013e-1` |
| feedback | fail-closed constraint veto at stage `3.48005M`; last row `3.46372M` finite | 0.681260 | `3.81213e-2` | `1.75583e-3` | `1.00017e-1` |

The prescribed result must not be called a numerical endpoint failure or a
`5.2M` pass: PBS ended it at the one-hour limit with run status 143.  Its last
state closely approaches the previously observed `~4.97M` failure window and
has severe constraint/source growth, but no continuation was launched.

The frozen-reference control is not a physical fixed-point test.  Its
reference uses the pre-collapsed lapse `alpha=psi^-2`, not the stationary
isotropic Schwarzschild Killing lapse.  Thus physical equals reference at
`t=0` does not imply zero vacuum RHS.  The finite `5.2M` trajectory supports
only a bounded control result at one resolution.

## Does direct Delta q precede constraint growth?

Yes in both moving-reference cases under the documented heuristic, but the
frozen control shows it is not sufficient by itself:

| case | legacy/blend Delta-q onset | curl onset | reduction onset | GH onset |
| --- | ---: | ---: | ---: | ---: |
| frozen | `0.881M` | not crossed by `5.2M` | not crossed | not crossed |
| prescribed | `0.600M` / `0.641M` | `2.703M` | `2.901M` | `3.564M` |
| feedback | `0.641M` / `0.641M` | `2.001M` | `2.301M` | `2.482M` |

Inner-shell onsets are later (`1.001M`, `0.960M`, and `0.960M` for frozen,
prescribed, and feedback), while outer-shell onsets are `1.262M`, `0.741M`,
and `0.782M`.  These shell differences and the growing spatial variance rule
out a single pure-power description of the intermediate state.

At the frozen endpoint the legacy values are `q_phys=1.16322`,
`q_ref=1.19118`, and `Delta q=-0.0279635`; the GH RMS remains only
`6.04e-5`.  The legacy residual in the calibration relation,
`e_G+2 Delta q`, is `-0.002642`, but the outside-shell residual is `0.0709`.
At the prescribed last row, the legacy mismatch is `-0.323611` and its
variance is `0.185694`; in the blend/outer regions, mismatch variances are
`2.18255` and `2.61439` with maxima above ten.  There the calibration residual
is large (`0.428` and `1.801`), so `e_G` cannot be treated as an exact global
power mismatch.  Feedback shows the same loss of a pure-power description.

The largest moving-case GH, Pi-RHS, and covariant-source sectors remain near
`r=0.47--0.60M`.  At the prescribed last row, for example, GH-constraint and
Pi-RHS maxima are `36.58` at `r=0.533M` and `1682.48` at `r=0.540M`.
This associates the growth with the blend/window region, not the puncture or
outer boundary, but remains correlation rather than proof.

## Feedback interpretation

The existing feedback controller is initially faster than the tau-8 replay:
near `xi=0.5`, feedback has `xi_dot~0.248` at `t=2.50M`, while prescribed has
`xi_dot=0.125` at `t=4.00M`.  Its safety command first set `v_cmd=0` at
`t=2.96120M`, when `xi=0.613944`, `xi_dot=0.212365`, legacy
`Delta q=-0.226376`, and GH RMS `5.708e-3`.  Because the continuation state is
second order, the reference did not become fixed immediately.  By the last
history row, `xi` had advanced to `0.681260` while `xi_dot` decelerated to
`0.077732`; legacy mismatch grew to `-0.309874`, and GH RMS grew by a factor
of 6.7 to `3.812e-2`.  The constraint veto then failed closed at `3.48005M`.

Therefore feedback neither completes the transition nor avoids failure by a
stalled continuation.  Growth continues after the safety command freezes,
but no interval with exactly fixed `xi` is obtained, so this run cannot decide
whether a strictly fixed intermediate reference would continue to grow.  It
does show that the existing untuned feedback controller is not protective.

## Fixed-point distinctions and formulation uncertainty

- The invariant physical spacetime is Schwarzschild.
- A stationary coordinate solution depends on the slicing and shift; the
  chosen wormhole lapse is not the stationary Killing lapse.
- A controller equilibrium is a state satisfying its controller equations,
  not necessarily a stationary Einstein solution.
- A safety command freeze is a fail-safe response and, with second-order
  continuation dynamics, does not imply `xi_dot=0` immediately.

The evidence supports: direct power mismatch begins before constraint growth
on moving trajectories; a comparable but smaller mismatch can remain bounded
on the frozen trajectory; and the destructive growth localizes in the
radially blended intermediate reference/gauge region.  It does not establish
whether mismatch is causal, whether a genuinely fixed intermediate state is
unstable, or whether a different continuation design would succeed.  Since
mismatch preceding growth is now directly established, a separate bounded
exponent-feedback proposal is scientifically motivated, but it must first
specify equilibrium, signs, limits, and shell eligibility and pass source-unit
tests.  No such controller was activated here.

## Provenance and retained outputs

Compact histories, max-location tables, analyses, plots, scheduler provenance,
432-rank mapping, scripts, and restart manifests are under:

```text
artifacts/ref_gh_relative_damped_single_hole_20260830/power_lag_three_case_8792940
```

The full Aurora directory is:

```text
/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_relative_damped_power_lag_20260831_9eff5b52/three_case_8792940.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
```

Large restart data are not committed.  The directory retains 12 frozen
checkpoints totaling 122,929,386,785 bytes, 10 prescribed checkpoints totaling
102,441,159,599 bytes, and 7 feedback checkpoints totaling 71,708,810,685
bytes.  Their per-file SHA-256 values and sizes are committed in each case's
`restart_sha256.txt` and `restart_sizes.tsv`; `restart_manifest_sha256.txt`
authenticates those compact manifests.  Compute-only hash job `8792997`
exited zero.  No field dump or restart file is present in Git.
