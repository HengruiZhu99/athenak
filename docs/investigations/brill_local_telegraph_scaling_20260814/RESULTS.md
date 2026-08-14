# Figure-3 local telegrapher damping comparison

## Outcome

The domain-maximum baseline remains the best of the four prospectively fixed
prescriptions.  It reproduced the previous result and reached `t=10.16294 M`.
All three field-local alternatives failed the same unchanged strict-positive
chi boundary-prolongation gate earlier.  There was no coefficient tuning,
floor, clipping, smoothing, threshold relaxation, or selective rerun.

| damping scale `mu` | terminal `t/M` | terminal `tau/M` | last finite `C2` | rejected parent stencils |
|---|---:|---:|---:|---:|
| `max_domain |K|` (baseline) | 10.162940 | 6.201069 | 2.4013e11 | 6,412 |
| local `|K|` | 8.907275 | 5.472376 | 4.7651e12 | 13,184 |
| local `sqrt(Kij K^ij)` | 9.193945 | 5.751011 | 7.4024e14 | 12 |
| local `sqrt(gamma^ij d_i chi d_j chi)` | 8.358252 | 5.016554 | 3.3252e8 | 7,320 |

The rejected-stencil count is not a ranking metric: it is the inventory at the
first fatal boundary pass, and each prescription reaches that pass at a
different state and time.

At the matched `t approximately 8 M` slice, the combined constraint norms were
approximately 101 (baseline), 2.61e4 (local `|K|`), 3.20e3 (extrinsic norm),
and 77.7 (chi-gradient norm).  The chi-gradient choice is temporarily smallest
there but fails only `0.36 M` later; it is not a stability improvement.

## Scale-invariant parameterization

Every nonfixed mode uses the same coupled scaling requested for the
telegrapher system.  With `Kstar=max_domain |K|`,

```text
Q          = mu / Kstar
tau_eff    = tau / Kstar
kappa_eff  = kappa / Kstar
```

therefore

```text
Q/tau_eff       = mu/tau
kappa_eff/tau_eff = kappa/tau.
```

The production kernel evaluates the cancelled right-hand expressions.  This
avoids division by zero on the time-symmetric slice and does not multiply a
local `mu` by an additional `Kstar`.  All four cases keep `tau=kappa=1`.

## Baseline reproduction

The fresh `max_domain_abs_K` history has the same 854 rows as the preceding
baseline.  Its original primary columns 1 through 22 are bitwise identical,
including time, timestep, global constraints, `max|K|`, AMR inventory,
central lapse/proper time, and axis curvature.  It also has the identical
terminal time and the identical 6,412-parent strict-chi failure.  Appending
the two mu reductions changes a few auxiliary axis/off-axis decomposition
reductions at roundoff/order level (largest absolute difference 0.00772 in
large accumulated quantities); it does not alter the primary result.

## Spatial diagnostics

The plotted profiles are read directly from the AthenaK binary output, not
reconstructed from IrisK or from an external evolution.  The first file at
cycle zero contains the constructor value of the diagnostic work array, so
the profile comparison deliberately starts at cycle 64 (`t approximately
0.633 M`), after the first physical RHS evaluations.

All 12 selected raw snapshots contain zero negative and zero nonfinite mu
cells.  The local `|K|` prescription has the most visibly jagged equatorial
profile and noisy domain minimum as its many `K=0` surfaces move through the
AMR hierarchy.  The extrinsic norm is smoother, while the chi-gradient norm
develops the earliest large localized peak.  These are observations, not a
proof that profile nonsmoothness alone causes the later chi failure.

## Plots

- [Figure-3 comparison](figures/figure3_local_telegraph_mu_comparison.png)
- [Constraint comparison](figures/constraints_local_telegraph_mu_comparison.png)
- [Central lapse and max-K](figures/lapse_and_maxK_local_telegraph_mu.png)
- [Mu extrema](figures/telegraph_mu_extrema.png)
- [Spatial mu profiles](figures/telegraph_mu_spatial_profiles.png)

PDF versions and the machine-readable [analysis summary](data/analysis_summary.json)
are in the same directory.

## Provenance

- source commit: `2a8ad80e02279769a99fe279b7a33516bc6c8d0d`
- source tree: `67709c405a1169a15643cb933eec5353cd216243`
- pushed branch: `codex/cartoon-allbulk-brill-scaleinv-20260813`
- Perlmutter job: `56955603`, one A100 40 GB, seven numbered steps
- remote root manifest SHA: `1735a72c934f50a65d11657ea0410d650e9920efc2ca5c525d74f1f52f86d933`
- detached manifest-file SHA: `7a549f0aa7968386bc0a85038ababb4906de6dd46a88ca393e6c3df6aed0ec42`
- campaign summary SHA: `78d61defa94c822200dd8eedcf6c205fcbdab686f6c4e05d3c01ca8d91ad68c4`
- settled accounting SHA: `8a055fd6cb44ff87ec5849ba445db3bff8c4681be00b7500e99c152c4b69d530`
- full remote member and detached-manifest verification: PASS

This is a comparative diagnostic result, not production qualification of the
Figure-3 evolution.
