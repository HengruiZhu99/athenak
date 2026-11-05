# Short boosted error growth and interface controls

## Preserved larger-block comparison

Job9040972 completed both fresh600-step cases to.075M. The frozen collector
admitted both endpoints; all303 worker artifacts are locally hash/length
verified in large-block-results/local-verification.json. These short results
do not establish long stability or justify adopting either grid.

The24³/two-refinement layout retains the finest puncture spacing, but evolves
2.478 times as many cells. Its max-rank stepping time is1.788 times larger and
native command time1.834 times larger. Per-cell throughput improves1.386 times;
total GPU-hour cost does not improve. This is not pure block repacking or strong
scaling. The fourfold stable performance gate remains unmet.

At fixed r>.3, endpoint H/M are .7011433/5.3860483 for12³ and
.1319296/1.2266421 for24³. Both grow sharply from their initial exterior norms.
All three saved horizon rows pass the online finder, with nearly identical
mass, and both have the same central floor/sign counts. Horizon accuracy and
finite evolution do not validate the exterior field accuracy.

The independent read-only native constraint reconstruction reproduces both
GPU full-domain report norms to relative1.6e-12, retains both checkpoint hashes,
and uses no analysis operator override. Its radial volume integrals show:

| Layout | Fraction exterior H² in .94–.97 | Fraction exterior M² in .94–.97 |
|---|---:|---:|
|12³|99.9714%|99.8735%|
|24³|99.4075%|99.4922%|

The large error is concentrated near the original mesh transition r=.95,
outside the peak initial slicing gradient near.9. This narrows the numerical
suspect but does not prove a sole interface cause. The12³ layout combines an
eightfold radial-spacing jump with an angular-resolution drop at.95; the24³
layout has a fourfold radial jump there and uniform shell angular subdivisions.
Errors in both layouts mean angular coarsening alone is not sufficient to
explain the observation.

Evidence: large-block-results/, large-block-native-analysis/, reproducible
analyze_large_block_pair.py and analyze_large_block_radial.py. Native radial
figures were visually inspected. The separate interpolated ray figure in
large-block-fields/ has a significant limitation: high-order interpolation
near patch/block corners has large weights (up to1.83e5) and6/8-order scaled
differences up to.074. It is qualitative; do not use those corner extrema for
pointwise convergence. The volume-localization percentages above do not use
that ray interpolation.

## Existing fresh control job9041071

Four debug-scaling nodes/48ranks, same frozen executable/source/gauge/initial
data and dt1/24000, two fresh N12 cases to600steps/.075M. Both debug queues
were empty before one submission; package and37 locally available plan
bindings were verified after mirroring, and the executable was rehashed remotely.
No rebuild, smoke, checkpoint reuse, or extra long evolution was submitted.

1. `radial-graded`: replace radial interface.94375 with.975. Final interfaces
   are .93125,.9375,.95,.975,1; angular2→1 remains at.95. The final spacing
   ratios become2,2,1. This coarsens .9375–.95 by2 and refines .95–1 by2;
   outermost spacing and therefore the mesh-dependent scri penalty rate also
   change. There are814 blocks and1,482,624 evolved cells.
2. `angular-moved`: keep original radial interfaces; move angular2→1 from.95
   to.9375. There are796 blocks and1,451,520 evolved cells. Angular coarsening
   occurs earlier, with radial spacing and outermost spacing unchanged.

Neither is a full factorized experiment. Changes in error amplitude and
location will guide a further matched control if needed. The parser's
outermost-angular-factor1 restriction and all guards remain unchanged.

Plan campaigns/boosted-tail-interface-pair-v1/plan.json, SHA256
7933efa7102497bfe0adb838a6ca1595fcb06a3db1e601ca4f789692e0041245.
Saved submission, package hashes, queue checks and local binding verification
are in tail-interface-controls/. Collect only after scheduler terminal using
the submitted tail-interface-pair-v1/collect.py; preserve/hash-verify every
artifact and compare fresh histories before any further launch.

At submission the guarded total footprint was43,834,507,264bytes, with
2,200,000,000 planned and1,050,000,000 atomic reserves; headroom2,915,492,736
under the authorized50GB cap. No output retirement was needed.
