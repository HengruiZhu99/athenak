# Reference-gauge Figure-3 same-tree resolution investigation

## Final verdict

`AURORA_PVC_QUALIFIED; EXACT_REPLAY_VERIFIED; STRONG_RESOLUTION_IMPROVEMENT; EARLY_O4_COMPATIBLE; PEAK_CONSTRAINT_INVALID; FULL_CONVERGENCE_NOT_ESTABLISHED`

The decisive same-tree N128/N256/N512 campaign is complete. Doubling the cells
per physical MeshBlock from N256 to N512 substantially delays and reduces the
constraint runaway, moves the first curvature peak toward the published
Figure-3 amplitude, suppresses the measured rho≈5 high-frequency content, and
allows the calculation to resolve the published deep minimum and rebound.

This strongly implicates bulk under-resolution as a major component of the
N256 failure. It does not produce a qualified Figure-3 reproduction: even N512
has order-unity squared constraint integrals at the first peak. Three-level
observed orders are near O4 early and degrade through collapse; constraint
orders are inconsistent between resolution pairs. No full convergence claim is
supported.

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

All three science cases use:

- shock-avoiding Bona--Masso lapse, `kappa=1`, unit initial lapse;
- prescribed zero shift and telegraph lapse disabled;
- native vertex-centered Cartoon SO(2), O4, q6 prolongation;
- RK4, CFL `0.15`, KO dissipation `0.50`;
- Z4c constraint damping `kappa1=kappa2=0`;
- outer boundary `128 M`;
- the exact accepted N256 AMR hierarchy and physical event times.

The only resolution change is cells per physical MeshBlock: N128 uses `16x16`,
N256 `32x32`, and N512 `64x64`. The root MeshBlock layout and every accepted
LogicalLocation remain identical.

## Exact hierarchy replay

Authority:

```text
path   evidence/aurora/authority/n256_reference_shock_authority.jsonl
SHA256 7055de601e6181e5ad7e1432b5c20a111b0ba67e0e8d5377c170ea80e7bedcde
```

Both N128 and N512 accepted the two authority events exactly:

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

N512 used six two-node/24-rank `debug-scaling` segments, jobs `8789895`,
`8789956`, `8790025`, `8790135`, `8790202`, and `8790242`. N128 used jobs
`8790272` and `8790338`. No further jobs are active.

Segment N512/000 ended its Athena process cleanly but PBS killed the allocation
while a redundant second full artifact verification pass was running. The
science disposition and restart precede that kill; later segments use one hash
pass and have complete evidence seals. This limitation is preserved rather
than hidden.

## Figure-3 comparison

No curve was shifted, fitted, or rescaled.

| case | first-peak tau | first-peak log10 abs(Kretschmann) | peak C squared integral |
|---|---:|---:|---:|
| N128 | 10.31396 | 4.29765 | 107.608 |
| N256 | 10.30333 | 5.01349 | 48.2330 |
| N512 | 10.30811 | 5.38112 | 4.09930 |
| published | 10.30683--10.31384 | 5.47778--5.48688 | not available |

N512 also gives:

| feature | N512 | published range |
|---|---:|---:|
| deep-minimum tau | 12.62280 | 12.61674--12.73112 |
| deep-minimum log10 abs(Kretschmann) | -6.07875 | -6.54553---5.20673 |
| rebound tau | 13.21629 | 13.18978--13.21977 |
| rebound log10 abs(Kretschmann) | -2.81849 | -2.95731---2.81225 |

The morphology and timing are strikingly close at N512, but the first peak is
not constraint qualified.

## Constraint evidence and location

The history measure is already the proper axisymmetric ring measure,
`2*pi*rho*sqrt(abs(det(gamma)))*drho*dz`, with canonical vertex ownership and
trapezoid endpoint weights. The jumps and growth are not a fictitious
collapsed-y normalization effect.

| squared integral | N128 max | N256 max | N512 max |
|---|---:|---:|---:|
| C | 107.608 | 48.2330 | 4.09930 |
| H | 88.4516 | 41.1314 | 3.63079 |
| M | 25.3663 | 9.53227 | 0.682824 |
| Z | 0.643210 | 0.0550105 | 0.00462028 through N512 final time |

At N512, C first crosses `0.01`, `0.1`, and `1` at proper times `9.88652`,
`10.05721`, and `10.19025`. C and H peak at `tau=10.29711`, coincident with
the curvature peak.

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

## Three-resolution assessment

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

## Interpretation

Observation:

- the exact same interface geometry becomes markedly better behaved when the
  cells per MeshBlock are doubled;
- N512 suppresses the rho≈5 numerical content and survives through Figure 3;
- global peak constraint locations are axis-adjacent, not coarse-fine-adjacent;
- N512 still has an order-unity constraint episode at the curvature peak.

Inference:

- parent/bulk under-resolution is a major driver of the N256 runaway and is not
  ruled out; it is strongly supported by this experiment;
- a rho≈5 persistent interface mode is deprioritized as the primary failure;
- residual axis or AMR-interface error may coexist with bulk under-resolution.

The natural next step is not gauge/KO tuning. It is a bounded shared-RHS audit
at the N512 peak, separating active-axis stencils, same-level seams, and
coarse-fine ghost/interface contributions while retaining the exact tree.

## Claim boundary

Supported:

- exact N128/N512 replay of the N256 hierarchy;
- clear resolution improvement and early O4-compatible central fields;
- N512 coverage of the published first peak, minimum, and rebound;
- substantially smaller/later constraints and weaker rho≈5 high-k content.

Not supported:

- a constraint-qualified Figure-3 reproduction;
- uniform convergence through collapse;
- exclusion of all axis/interface mechanisms;
- a unique source-level defect or a production numerical correction.

Detailed N512 and convergence reports are in `N512_REPLAY.md` and
`CONVERGENCE.md`. Machine-readable evidence is in `EVIDENCE_MANIFEST.json`.
