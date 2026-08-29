# N512 exact-replay discriminator

## Disposition

`REACHED_TAU_GATE; CLEAR_RESOLUTION_IMPROVEMENT; PEAK_STILL_CONSTRAINT_INVALID`

The N512 run replayed the accepted N256 AMR hierarchy exactly and reached
central proper time `14.98252698 M`, beyond the required `13.3 M` gate.  It
resolved the first peak, deep minimum, and rebound without a numerical failure.
It is a decisive improvement over N256, but it does not qualify as a scientific
Figure-3 reproduction because the first peak remains constraint contaminated.

## Frozen setup

No production numerical or gauge parameter was changed relative to N256:

- native vertex-centered Cartoon SO(2), O4 finite differences, q6 prolongation;
- RK4, CFL `0.15`, KO dissipation `0.50`;
- shock-avoiding Bona--Masso lapse with `kappa=1` and unit initial lapse;
- prescribed zero shift, telegraph lapse disabled;
- Z4c `kappa1=kappa2=0`;
- outer boundary `128 M`;
- root grid `256 x 512`, `64 x 64` cells per MeshBlock;
- the N256 authority's same root MeshBlock layout, physical MeshBlock bounds,
  LogicalLocation leaf set, and event times.

The authority SHA-256 is
`7055de601e6181e5ad7e1432b5c20a111b0ba67e0e8d5377c170ea80e7bedcde`.
The two replay events were accepted exactly at the recorded hexadecimal times,
with 200 and 212 leaves and checksums `24316947a3a67cd8` and
`cf0d2384b11c1d42`.  The hierarchy then remained fixed at 212 leaves and
physical level 5.

## Execution evidence

Six short two-node Aurora `debug-scaling` segments used 24 MPI ranks and all
PVC tiles:

| segment | job | science disposition | scheduler exit | final central proper time |
|---:|---:|---|---:|---:|
| 000 | 8789895 | healthy walltime restart | -29 | 6.98910 |
| 001 | 8789956 | healthy walltime restart | 0 | 8.26888 |
| 002 | 8790025 | healthy walltime restart | 0 | 9.57365 |
| 003 | 8790135 | healthy walltime restart | 0 | 10.93200 |
| 004 | 8790202 | healthy walltime restart | 0 | 12.40261 |
| 005 | 8790242 | healthy walltime restart | 0 | 14.98253 |

Segment 000's Athena process exited zero and wrote its restart and science
disposition before PBS killed the job during a redundant second full artifact
hash pass.  It is science-usable but lacks a completed root evidence seal.
The sealer was narrowed to one complete content-hash pass for later segments;
segments 001--005 all exited zero and have sealed manifests.

Manifest hashes for segments 001--005 are, respectively:

```text
c685988c980bf8f6f4a85ac383e149c5189912e6abd506d90299f4b08dd2bb7c
47275a90db713772fd107a59e9e66a7841fb1b295cd305a24d3d480486a4430c
71929509a2e9a9885f91ea51ae39e2180b4235700e5e3320f0be5dffaa2b19f1
622fea6b19e49b4a3652b3eb18cc8ee4915646515704b88a6205df97ab4f994b
5ff7b9f7a59fb1604c4a15e80377b3e3d4364efa0d3ed0af1e9235dd88733f90
```

## Direct Figure-3 comparison

No time, amplitude, or vertical transform was fitted.

| feature | N256 | N512 | published range |
|---|---:|---:|---:|
| first-peak proper time | 10.30333 | 10.30811 | 10.30683--10.31384 |
| first-peak `log10(abs(Kretschmann))` | 5.01349 | 5.38112 | 5.47778--5.48688 |
| deep-minimum proper time | not reached | 12.62280 | 12.61674--12.73112 |
| deep-minimum `log10(abs(Kretschmann))` | not reached | -6.07875 | -6.54553---5.20673 |
| rebound proper time | not reached | 13.21629 | 13.18978--13.21977 |
| rebound `log10(abs(Kretschmann))` | not reached | -2.81849 | -2.95731---2.81225 |

The N512 central curve is therefore much closer to the published morphology
and survives through the full requested interval.

## Constraint qualification

The constraint histories use the physical axisymmetric ring measure
`2*pi*rho*sqrt(abs(det(gamma)))*drho*dz`, with canonical vertex ownership and
trapezoid endpoint weights.  The result is not a collapsed-y normalization
artifact.

| squared integral | N256 maximum | N512 maximum | N512/N256 |
|---|---:|---:|---:|
| C | 48.2330 | 4.09930 | 0.0850 |
| H | 41.1314 | 3.63079 | 0.0883 |
| M | 9.53227 | 0.682824 | 0.0716 |
| Z | 0.0550105 | 0.00462028 through final time | 0.0840 |

For N512, C first crossed `0.01`, `0.1`, and `1` at proper times `9.88652`,
`10.05721`, and `10.19025`, respectively.  C and H peak near proper time
`10.29711`, essentially coincident with the curvature peak.  Thus the improved
curvature agreement is still not constraint qualified.

The C/H integral maxima occur at axis locations (`rho=0`, `z=-0.75`) and the M
maximum is also within one radial grid spacing of the axis.  They are far from
the nearest coarse-fine interface.  This is geometric classification, not
proof of an axis source bug.

## rho=4--6 high-frequency discriminator

All 25 evolved fields were stitched across the complete finest physical patch
`4 <= rho <= 6`, `-2 <= z <= 2` before differencing and Fourier analysis.
Duplicated shared vertices agreed to roundoff.  Matched slices were used at
proper times approximately 8.03, 9.50, 10.27, and 11.00.

Most variables have smaller fourth/second-difference indicators at N512.  At
the near-peak slice, representative N512/N256 ratios are about `0.36` for
`chi`, `0.25` for `Khat`, `0.23--0.48` for the principal `Atilde` components,
and `0.25` for the sampled conformal connection functions.  `Theta` has the
largest normalized indicator before the peak, but its absolute fluctuation is
only about 3.5--6.2% of N256 and its power above half the N256 Nyquist remains
below `3.1e-7`; by proper time 11 its normalized indicator is also smaller than
N256.  No measured rho≈5 high-k branch strengthens with resolution.

These observations strongly deprioritize a persistent rho≈5 same-level seam or
coarse-fine mode as the primary explanation for the N256 runaway.  They do not
prove that all interface operators are stable.

## Interpretation and claim boundary

Supported:

- doubling cells per physical MeshBlock on the identical AMR tree delays and
  reduces the constraint runaway;
- it moves the first peak toward the published amplitude;
- it removes the N256 survival limit and resolves the published minimum and
  rebound interval;
- the measured rho≈5 high-frequency content generally decreases.

Not supported:

- a constraint-qualified Figure-3 reproduction;
- convergence from N256/N512 alone;
- a unique bulk, axis, seam, or coarse-fine source bug;
- changing KO, gauge, transfer, or positivity gates.

Because N512 is clearly improved, the controlling workflow proceeds to the
N128 exact-authority replay and a claim-limited three-resolution analysis.
