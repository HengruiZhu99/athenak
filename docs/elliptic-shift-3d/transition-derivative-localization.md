# Transition derivative and source localization

This is a frozen, read-only initial-data diagnostic of the supported-puncture
v15 pilot. No engine, checkpoint, sampler or production derivative is changed.
The independent existing `IsotropicWormhole(M=1/3)` supplies analytic radial jets
on every actual active cell and interior ghost query; its values match the
selected initial file exactly. The Minkowski reference and physical equations
are unchanged. Full tensor constraints and 24 physical constraint/source
channels are evaluated, with the same independently refined radial shift jets
in every comparison. This isolates geometric/source differencing from shift
solve error. It is not a new coupled evolution or scientific acceptance.

## Dominant mechanism on the actual 47088-cell grid

The final pilot maximum at point126 has H=-.0435195. At the same initial point,
x=(.265666486,.044277748,.044277748), r=.272946, the diagnostic finds:

| Substitution into the original initial tensor state | H |
|---|---:|
| Supported baseline | -.04362334 |
| Exact interior ghost values | -.04334324 |
| Exact chi gradient only | -.04530404 |
| Exact chi Hessian only | +.00168070 |
| Exact chi gradient and Hessian | -7.98e-16 |
| Exact lapse jets only | -.04362334 |
| Exact curvature first derivatives only | -.04362334 |

Thus the dominant defect here is truncation/cancellation in the mapped active
second derivative, especially chi, rather than halo values. The small halo
contribution is retained; it is not assumed zero. Exact analytic active jets
produce global H/M maxima4.45e-14/5.17e-14 over every cell. No cells or puncture
neighborhoods are removed from norms. Replacing chi derivatives analytically
restores the initial all-cell H RMS to4.23e-15; this is a diagnostic oracle,
not an admitted production method or physical-RHS subtraction.

The initial full-domain H RMS is .00299250 and transition RMS .0155703. Sources,
physical positive-jet errors, unnormalized L2 and all regions are in
`evidence/transition-jet-analysis-20261027`. Every original positive value is
retained. Only interior ghost samples0<r<.85 are replaced in the ghost control;
outer ghosts keep the original scri closure and are labeled separately.

## Sixth-order logical-stencil falsifier

A standalone seven-point first/second stencil retains the full inverse-map
Jacobian and second chain-rule term. Only transition-layer chi/lapse derivatives
change; every other cell/field retains supported v15 derivatives. This is not a
global sixth-order scheme. The original ng2 baseline and its positive-halo cache
are embedded unchanged into an ng3 sampling grid; the added third ghost layer
uses ordinary interpolation unless the independent exact-halo control is used.

The original N6/refinement3 grid has47088cells. Related static-eight-core
N6/8/10 grids have10800/25600/50000cells. Their different core refinement is
explicit, and no apparent global RMS convergence order is inferred.

| Transition H RMS | N6 static | N8 static | N10 static |
|---|---:|---:|---:|
| Fourth/default halos | .187754 | .0200761 | .00541672 |
| Fourth/exact halos | .0156043 | .00556668 | .00240993 |
| Sixth/default added third halo | .206511 | .0192271 | .00447378 |
| Sixth/exact halos | .00773709 | .00190732 | .000580379 |

On the original N6/refinement3 grid, sixth-order transition derivatives reduce
all-cell H RMS .00299250→.00215457 and transition H RMS
.0155703→.00762120. At point126, H changes -.0436233→-.0194870;
exact-halo sixth-order gives-.0191785. Chi Hessian component error decreases
.08117→.04134 and lapse Hessian error .05431→.02704. This is useful but
insufficient. Static coarse cores have a separate halo limitation: sixth-order
with an inadequately accurate extra halo can worsen the result. No blanket
higher-order adoption is justified by the original-grid result.

Independent physical Cartesian constant, linear and anisotropic-quadratic
samples demonstrate curved-map truncation. N6 maximum linear error decreases
.2281→.05760, while deliberately omitting the second-chain term gives49.95.
N8/10 sixth-order linear errors are .01373/.004526. Constants are preserved
exactly by the anchored sixth-order formulas. The full chain rule is retained;
this is an approximation error diagnosis, not evidence of a missing chain term.
All physical lapse/source errors remain visible, including underresolved radial
height-transition structure that sixth-order improves only modestly.

Two setup failures remain separate: ng3 generated28920 positive-halo rows beyond
the20000 cap; a proposed standalone30000 allocation was itself rejected by the
public constructor's hard limit. No live header or guard was changed. The
successful third attempt preserves the original ng2 cache and adds the sampling
layer separately. All subsequent diagnostics complete within43seconds, with a
180-second cap per owned process and no shift solve/evolution.

## Physical Cartesian P5 hypothesis: rejected

A separate geometry-only QR probe retains all56 Cartesian monomials through
P5. It tests48 transition queries with125/175/256 nearest actual active donors.
Rank, condition<=1e8, reproduction<=1e-10 and value/gradient/Hessian functional
L1 bounds16/32/64 are checked without reducing degree or relaxing guards.
120/144 plans pass;24 are explicitly unsupported. The independent physical
constant/linear/quadratic error is at most5.35e-12, with bounded seeded-noise
response. Thus exact low moments and reasonable conditioning are achieved.

At point126,125 donors have rank53;175 have full rank but gradient L1=32.051
and are rejected. The256-donor plan passes with condition1131, reproduction
1.1e-13 and gradient/Hessian L1=15.61/26.62. Nevertheless its actual H is
-.199809 when fitting stored positive logs and-.0602398 when fitting direct
physical chi. Both are worse than the original FD4(-.0436233), and much worse
than FD6(-.0194870). Lapse Hessian errors also worsen. A small nonspherical
angular perturbation is included and does not rescue this candidate. Physical
values are never projected to fitted values. A broad footprint's approximation
error matters beyond rank/moments/noise bounds; nearest-neighbor P5 is not
adopted. `transition-physical-p5-r2-20261027` preserves matched log/direct tests.

## Evidence and bounded reproduction

- `transition-jet-localization-20261027`: frozen v15 source capsule, independent
  complete active/interior-ghost oracle and exact-jet substitutions.
- `transition-order-study[-r2]-20261027`: retained cache setup failures.
- `transition-order-study-r3-20261027`: qualified stencil study and Cartesian
  mapping controls, complete all-cell fields/source errors.
- `transition-jet-analysis-20261027`: every-cell/region/source norms, raw NPZs,
  mechanism and order-control plots.
- `transition-physical-p5[-r2]-20261027`: small-stencil log/direct falsifiers.
- `transition-physical-p5-analysis-qualified-20261027`: explicit unsupported geometries,
  guard audits, angular test and falsifier plot.

Source/initial hashes are checked before/after. Large files have lossless gzip
copies and compression manifests; local originals remain unchanged. Source
capsules are archived separately so diagnostics do not depend on later live
physics edits. Fresh output paths are required:

```
python3 tst/elliptic_shift_3d/run_transition_jets.py --output NEW_INITIAL
python3 tst/elliptic_shift_3d/run_transition_order.py --output NEW_ORDER
python3 tst/elliptic_shift_3d/analyze_transition_jets.py --initial NEW_INITIAL --order NEW_ORDER --output NEW_ANALYSIS
python3 tst/elliptic_shift_3d/run_transition_p5.py --output NEW_P5
python3 tst/elliptic_shift_3d/analyze_transition_p5.py --probe NEW_P5 --output NEW_P5_ANALYSIS
```

The order/P5 runners intentionally bind the historically frozen initial capsule;
when relocating, update that recorded capsule path rather than silently using
new live headers. No scientific acceptance, generic regularity, late collapse or
production adoption is claimed. The next derivative-basis hypotheses require
independent constant-preservation and nonstationary-source checks before use.
