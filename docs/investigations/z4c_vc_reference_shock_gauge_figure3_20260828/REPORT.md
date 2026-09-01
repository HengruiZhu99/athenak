# Reference-gauge Figure-3 same-tree resolution investigation

## Final verdict

`EXACT_REPLAY_VERIFIED; FINE_TRIPLE_O4_COMPATIBLE_THROUGH_FIRST_PEAK; N1024_PEAK_MATCHES_PUBLICATION; CONSTRAINT_CONVERGENCE_NONUNIFORM; CFL040_KO010_FAILED_EARLY`

The same-tree N128/N256/N512/N1024 campaign is complete. N1024 reaches the
matched endpoint, reduces the N512 peak C/H/M squared integrals by roughly two
orders of magnitude, and places the unshifted first curvature peak directly
inside the published time and amplitude bands.

This strongly implicates bulk under-resolution as a major component of the
N256 failure. The fine N256/N512/N1024 triple is O4-compatible in central
curvature and lapse through the first peak. It still does not produce a
constraint-qualified Figure-3 reproduction: N1024 crosses the campaign's
strict C-integral gate near the peak, and no three-level order exists across
the later deep-minimum/rebound interval.

A controlled same-tree N1024 ablation changed CFL `0.15 -> 0.40` and KO
`0.50 -> 0.10` together. It agrees closely with the baseline through roughly
`tau=6`, then develops a constraint runaway, and fails the strict state gate at
coordinate time `16.72465 M` (`tau=8.68734` in the last history row), before
the published first peak. This result rules out that combined setting as a
cure, but cannot attribute the failure separately to CFL or KO.

## Qualified implementation and fixed setup

The corrected shock-avoiding/effectively-zero-shift implementation was
qualified on Aurora PVC in job `8789659`; eleven focused tests passed. The
narrow source fix copies host-owned gauge policy into a scalar before device
capture and does not change the gauge equation or production numerics.

```text
source commit       f8303c6be7eb214fa1e91b646123ee0d434b3698
source tree         7a585ca487b12351b084eb425bb812775849b001
AthenaK executable  aae7ccb8739fb4951221ad7be69ea0e220548b52d402086f57d7857fa2c97a13
CMake cache         8da40bcb47564d9184119ca207f9847a33a3d1b5bd2930627d705cda8fb36386
IrisK library       380a90d5b1d9762fe7f9076edcb27fb4a209f4cd8c070da376c36284a438c7a1
Brill coefficients  1b5f0efc3f080215ed7d7994194ba63ea123415bfd8e74c54ca1fd72680aea10
input               6c694cf871a3d694d745f0fb58b279b6cd07516463ac8ad54f1c91d2689c90ba
```

All four science cases use:

- shock-avoiding Bona--Masso lapse, `kappa=1`, unit initial lapse;
- prescribed zero shift and telegraph lapse disabled;
- native vertex-centered Cartoon SO(2), O4, q6 prolongation;
- RK4, CFL `0.15`, KO dissipation `0.50`;
- Z4c constraint damping `kappa1=kappa2=0`;
- outer boundary `128 M`;
- the exact accepted N256 AMR hierarchy and physical event times.

The only resolution change is cells per physical MeshBlock: N128 uses `16x16`,
N256 `32x32`, N512 `64x64`, and N1024 `128x128`. The root MeshBlock layout and
every accepted LogicalLocation remain identical. N1024 used the later
performance-equivalent source `02704c79ffe95312cfeba9acde3d38f8b9677dec` on
four Perlmutter A100s; a matched N512 interval agrees with Aurora to backend
roundoff scale.

## Exact hierarchy replay

Authority:

```text
path   evidence/aurora/authority/n256_reference_shock_authority.jsonl
SHA256 7055de601e6181e5ad7e1432b5c20a111b0ba67e0e8d5377c170ea80e7bedcde
```

N128, N512, and N1024 accepted the two authority events exactly:

| event | exact coordinate time | leaves | checksum |
|---:|---|---:|---|
| 1 | `0x1.97f84920e41a7p-7` | 200 | `24316947a3a67cd8` |
| 2 | `0x1.31fd0a8ca4018p-6` | 212 | `cf0d2384b11c1d42` |

The final tree then remained fixed at 212 leaves/physical level 5. Native AMR
decisions were logged only as shadow diagnostics and had no authority.

## Run outcomes

| case | final coordinate time | final proper time | execution disposition |
|---|---:|---:|---|
| N128 | 45.0 | 19.33240 | reached tlim, finite, constraint invalid |
| N256 | 30.0 | 11.28631 | reached tlim, finite, constraint invalid |
| N512 | 38.65233 | 14.98253 | gate exceeded, finite, constraint invalid at peak |
| N1024 | 38.65233 | 14.98066 | reached tlim, finite, constraint invalid at peak |
| N1024 CFL 0.40, KO 0.10 | 16.72465 | 8.68734 | failed: conformal metric not positive definite |

N512 used six two-node/24-rank `debug-scaling` segments, jobs `8789895`,
`8789956`, `8790025`, `8790135`, `8790202`, and `8790242`. N128 used jobs
`8790272` and `8790338`. No further jobs are active.

Segment N512/000 ended its Athena process cleanly but PBS killed the allocation
while a redundant second full artifact verification pass was running. The
science disposition and restart precede that kill; later segments use one hash
pass and have complete evidence seals. This limitation is preserved rather
than hidden.

N1024 ran from scratch in Perlmutter interactive job `57702588` on four
distinct A100 GPUs and completed in `02:37:46`; Athena, `srun`, and the
scheduler all exited zero.

The ablation ran in Perlmutter job `57706639` on the same four-A100 layout.
Both replay events and tree checksums matched the authority. The final gate
reported `nonpositive_metric_pivot_1` after RK stage 1 at
`(rho,z)=(0.0625,19.96875)`, with positive `chi=0.91660` but
`det(gtilde)=-0.29184`. Job `57706621` was an earlier zero-science startup
failure caused by the misspelled runtime key `time/cfl`; it is retained only
as provenance.

## Figure-3 comparison

No curve was shifted, fitted, or rescaled.

| case | first-peak tau | first-peak log10 abs(Kretschmann) | peak C squared integral |
|---|---:|---:|---:|
| N128 | 10.31396 | 4.29765 | 107.608 |
| N256 | 10.30333 | 5.01349 | 48.2330 |
| N512 | 10.30811 | 5.38112 | 4.09930 |
| N1024 | 10.30964 | 5.47867 | 0.0388727 |
| published | 10.30683--10.31384 | 5.47778--5.48688 | not available |

The two finest cases give:

| feature | N512 | N1024 | published range |
|---|---:|---:|---:|
| deep-minimum tau | 12.62280 | 12.61950 | 12.61674--12.73112 |
| deep-minimum log10 abs(Kretschmann) | -6.07875 | -7.88841 | -6.54553---5.20673 |
| rebound tau | 13.21629 | 13.21271 | 13.18978--13.21977 |
| rebound log10 abs(Kretschmann) | -2.81849 | -2.81258 | -2.95731---2.81225 |

N1024 directly matches the published first peak and rebound, while its deep
minimum is too deep. The first peak remains outside the strict constraint gate.

## Constraint evidence and location

The history measure is already the proper axisymmetric ring measure,
`2*pi*rho*sqrt(abs(det(gamma)))*drho*dz`, with canonical vertex ownership and
trapezoid endpoint weights. The jumps and growth are not a fictitious
collapsed-y normalization effect.

| squared integral | N128 max | N256 max | N512 max | N1024 max |
|---|---:|---:|---:|---:|
| C | 107.608 | 48.2330 | 4.09930 | 0.0388727 |
| H | 88.4516 | 41.1314 | 3.63079 | 0.0315101 |
| M | 25.3663 | 9.53227 | 0.682824 | 0.00453466 |
| Z | 0.643210 | 0.0550105 | 0.00462028 | 0.00400120 |

At N512, C first crosses `0.01`, `0.1`, and `1` at proper times `9.88652`,
`10.05721`, and `10.19025`. C and H peak at `tau=10.29711`, coincident with
the curvature peak.

At N1024, C first crosses `0.01` at `tau=10.17835`, peaks at `0.0388727`,
and never crosses `0.1`.

The baseline and ablation have identical initial constraint integrals. For
example, their initial C squared integral is
`1.2881308780515045e-08`. Across N128/N256/N512/N1024 the initial C values are
`1.30505e-1`, `6.85048e-4`, `2.88883e-6`, and `1.28813e-8`; the N512-to-N1024
amplitude order is `3.90`. Thus the initial data are demonstrably converging
at high order. The later constraint behavior is not uniform: over `tau=0--8`,
the N512/N1024 median amplitude orders are near zero for C and Z, while in
`tau=10--11.286` they are `2.69` (C), `4.00` (H), `3.74` (M), and only `0.27`
(Z). This is a resolution-independent evolution/diagnostic floor plus
collapse-window convergence, not a single asymptotic constraint order.

In the CFL/KO ablation, the C squared integral crosses `0.01` already at
`tau=7.54619`, then `1` at `tau=7.85416`, and reaches `4.97e4` in the last
history row. Its maximum C amplitude is `222.89`, versus `0.197` for the
baseline. The two trajectories differ in two numerical controls, so this is
an ablation of the combined setting only.

The N256 and N512 C/H maxima occur on the axis but far from a coarse-fine
interface; M is within one radial spacing of the axis. This is a geometric
observation, not proof of an axis-boundary source bug. The rho≈5 feature is not
the global peak once the constraint integrals become large.

## rho≈5 field diagnostics

All 25 evolved variables were stitched over the finest physical patch
`4<=rho<=6`, `-2<=z<=2` before differencing and Fourier analysis. Shared nodes
agree to roundoff. Matched proper-time slices are approximately 8.03, 9.50,
10.27, and 11.00.

At the near-peak slice, N512/N256 fourth-difference ratios are about `0.36` for
chi, `0.25` for Khat and the sampled conformal connections, and `0.23--0.48`
for representative Atilde components. Theta's normalized indicator is larger
before the peak, but its absolute fluctuation falls to 3.5--6.2% of N256 and
its power above half the N256 Nyquist stays below `3.1e-7`; by tau≈11 its
normalized indicator is smaller as well. No measured rho≈5 high-k branch grows
with resolution.

## Four-resolution assessment

Median Richardson orders for central fields at common proper time are:

| tau window | curvature | lapse |
|---|---:|---:|
| 0--8 | 4.86 | 3.93 |
| 8--10 | 3.34 | 3.36 |
| 10--11.286 | 2.10 | 1.40 |

This is early O4-compatible behavior followed by collapse-window degradation.
Constraint-amplitude pair orders are positive but inconsistent; for aggregate C
they are 3.46 versus 1.06 in tau 0--8 and 1.03 versus 2.38 in tau 10--11.286.
No single order describes both pairs. N256 ends before the late Figure-3
interval, so no three-level late-time order can be computed.

For the fine N256/N512/N1024 triple, median curvature orders are `4.10`,
`3.89`, and `5.91`, and lapse orders are `3.99`, `3.79`, and `3.34` in the
same windows. These are O4-compatible through the first peak; the curvature
value above four is treated as local superconvergence, not a higher-order
method claim.

## Interpretation

Observation:

- the exact same interface geometry becomes markedly better behaved when the
  cells per MeshBlock are doubled;
- N512 suppresses the rho≈5 numerical content and survives through Figure 3;
- N1024 directly matches the published first peak and reduces peak C/H/M by
  roughly two orders of magnitude relative to N512;
- global N256/N512 peak constraint locations are axis-adjacent, not
  coarse-fine-adjacent;
- N1024 still violates the strict C gate near the curvature peak.
- the combined CFL 0.40 / KO 0.10 N1024 setting is violently unstable before
  the first peak, despite identical initial constraints and hierarchy.

Inference:

- parent/bulk under-resolution is a major driver of the N256 runaway and is not
  ruled out; it is strongly supported by this experiment;
- a rho≈5 persistent interface mode is deprioritized as the primary failure;
- residual axis or AMR-interface error may coexist with bulk under-resolution.

The natural next step is a bounded N1024 peak constraint-localization audit,
separating active-axis stencils, same-level seams, and coarse-fine
contributions while retaining the exact tree. If time-step/dissipation
sensitivity must be isolated, vary only one of CFL and KO at a time; the
present combined ablation cannot identify which control caused the early
failure.

## Claim boundary

Supported:

- exact N128/N512/N1024 replay of the N256 hierarchy;
- fine-triple O4-compatible central fields through the first peak;
- direct N1024 first-peak agreement with the published band;
- substantially smaller peak constraints and weaker rho≈5 high-k content.

Not supported:

- a constraint-qualified Figure-3 reproduction;
- three-level convergence across the late minimum/rebound;
- exclusion of all axis/interface mechanisms;
- a unique source-level defect or a production numerical correction.

Detailed reports are in `N512_REPLAY.md`, `N1024_REPLAY.md`, and
`CONVERGENCE.md`. The five-trajectory plots and machine-readable analysis are
under
`analysis/aurora_n128_n256_n512_perlmutter_n1024_cfl040_ko010/final/`.
