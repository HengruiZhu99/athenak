# Scratch full-geometry ghost projection experiment

This directory changes only an ignored include overlay. No production source was edited. It projects full analytic-baseline-plus-extrapolated-deviation metric and traceless curvature at the ghost targets already required by native derivative stencils; donor plans, strictly interior support and nonrecursive filling are unchanged. Lambda is independent and is not reset from the metric.

## Algebra and admission

For full unscaled ghost metric g and A, s=det(g)^(-1/3), τ=g⁻¹:A, g′=s g, A′=A−g τ/3. The projected full g/A are subtracted against the same analytic reference or fixed reconstruction baseline used to assemble them, then consumed by actual native LoadMeshJet. Non-SPD metrics and nonfinite scale/trace throw; there are no floors or repairs.

The explicit rejection audit uses a strictly interior SPD determinant-one donor and a negative extrapolation weight; the invalid extrapolated ghost is rejected. All native snapshots below had valid unprojected SPD ghosts.

## Actual native algebraic and derivative defects

Exact double restart payloads were validated against every matching active float32 field output. Rebuilt original ghosts match the saved native ghosts bitwise (max difference 0). These evolving snapshots use κ5 controls; applying projection here replays a saved state and does not assert a changed evolution.

|degree|time|max ghost det residual, original|max ghost trace A, original|outer derivative RMS original (det first, det second, TF first)|outer derivative RMS projected|
|---:|---:|---:|---:|---|---|
|2|0|0|0|2.08124e-16, 5.19405e-16, 1.11089e-16|2.08124e-16, 5.19405e-16, 1.11089e-16|
|2|0.125326|0.0242014|0.130159|0.0202453, 0.505384, 0.0945259|0.000233112, 0.00294174, 0.00203914|
|2|0.250225|0.0503455|0.168437|0.066082, 1.673, 0.229514|0.00155367, 0.0316313, 0.00895868|
|2|0.5|0.0304251|0.0801981|0.0519369, 1.24454, 0.115612|0.00277348, 0.0908269, 0.0365446|
|4|0|0|0|2.08124e-16, 5.19405e-16, 1.11089e-16|2.08124e-16, 5.19405e-16, 1.11089e-16|
|4|0.125326|0.0969587|1.52233|0.0753613, 1.88937, 0.874954|0.158289, 2.55863, 1.94525|
|4|0.250204|0.708467|205.635|0.456161, 14.7264, 25.2873|1.60069, 20.0568, 154.094|

The derivative statistic at each node takes the maximum absolute direction/component identity residual, then computes the RMS over active r>.9 nodes. The identities are tr(g⁻¹∂g)=0; tr(g⁻¹∂²g)−tr(g⁻¹∂g g⁻¹∂g)=0; and tr(g⁻¹∂A)−tr(g⁻¹∂g g⁻¹A)=0. Finite differences need not obey these nonlinear chain rules exactly. Point projection reaches ghost det/trace roundoff but worsens derivative continuation for the quartic replay.

## Fixed point and actual native tangent gates

Repeated Prepare changes the N24/36/48 stationary reference by exactly 0. Actual reference RHS max is ≤2.63e−15; H≤6.10e−15, M≤1.06e−15, Z≤1.24e−16. Fixed analytic wormhole reconstruction M=.05 remains H7.64e−15/M1.45e−15/Z1.16e−16 after repeated Prepare. This is a reconstruction consistency check, not black-hole evolution evidence.

The actual finite nonspherical lapse .1 and shift .02 pulse, width .5, is the existing versioned tangent test. It retains native stencils, algebraic active projector, constraints and full analytic reference jets; signed Euler probes use dt1e−6. All initial constraints remain roundoff.

|N|original Hdot|projected Hdot|original Mdot|projected Mdot|original Zdot|projected Zdot|
|---:|---:|---:|---:|---:|---:|---:|
|24|0.4724136448|0.4723111216|1.734798039|1.732785688|0.04535568143|0.04532315243|
|36|0.08225204804|0.08238544138|0.6131455667|0.6129473046|0.008352754442|0.008488912795|
|48|0.02609270744|0.02611688303|0.2480366357|0.2480936639|0.002027781946|0.002026433405|

H/M tangent changes are below 0.17%; Z rises1.63% at N36 and changes below0.1% at N24/N48. There is no consistent instantaneous improvement. The stronger reduction of quadratic evolving derivative defects motivates the full native time comparison.

## Build and native evolution provenance

Apple clang21.0.0, -O3 -DNDEBUG -std=c++17 -arch arm64 and the original Kokkos static libraries. audit-results.json records every audit compiler command/source/header/executable hash. native-build/manifest.json records all original object/library hashes, six privately recompiled TUs and exact dependency/source snapshots. Runtime source TUs match commit27c19d20696ea6dd4704032c51dfd026218f64f2. The tangent test includes sorted radial bins from commit2564570b.

The baseline relink is identical to original runtime5ba555211db81985274f8ebe789869b8f6300f638ac8dfd7ba7736b541160788 outside Mach-O LC_UUID and ad-hoc signature bytes; all code/data/symbol bytes compare exactly. The normalized comparison proof is saved. All six native TUs whose dependency files include CartesianPatch were recompiled with the private overlay first in the include path.

The immutable projected runtime is e9b15a545065f8fe998c2a3b3e7b7b43ac2af2dd584500489b496dc89217fbb3. Current native κ10 wide N24 d2, pole.03, t2 run is in native-kappa10-t2; exact inputs, executable, histories, fields and checkpoints are retained. It requests t2, about2.682 outward crossings using independently converged reference ∫dr/outgoing=.745764383923422. history-comparison.json records matched common-time history interpolation against parent clean-wide-kappa10-long. The run failed before requested t2, exit−6, with `invalid full ghost matrix/trace in scratch projection: 14`. The exact failing stage time/cycle is not printed. Last saved valid time is1.8502979340961283/cycle4329; no following scheduled1.875 snapshot was produced. No invalid state was repaired.

This closure changes a nonlinear boundary extension, not the interior20-field principal system. No characteristic closure or stability claim follows from these checks.

## Final negative verdict

The last saved valid projected H/M/Z are1.56233946/.992251411/.165518559. Clean history interpolated to that exact time gives1.56574467/1.02718909/.194585520, candidate/control ratios.997825/.965987/.850621. The original control reaches requested t2; this candidate does not. All75 projected saved snapshots have finite active fields, positive α/χ and SPD regular spatial metric. Last eigen minimum is.944440, αmin1.26458, χmin.554653. The guard checks extrapolated full ghost geometry, independently of those active-state checks.

Both clean and projected cycle4329 states have the same bulk H peak r=.378886 and99.600%/99.764% squared H inside.9. The bulk spike is shared, not newly localized by projection. Native matched histories and full radial budgets preserve the rapid nonmonotonic growth. At the saved last-valid state both unprojected ghost continuations still pass SPD, but determinant minima fall to.150382/.151036 and ghost traces rise to19.3439/19.2080. Projected point det/trace remain roundoff; outer derivative identities still retain truncation errors. The later strict guard rejects14 invalid ghost matrices/traces; the failed-stage arrays are not saved, so the exact offending matrices and stage time are not inferred.

Reject this candidate as stabilization. Point algebraic consistency does not remove the shared finite-time instability or guarantee admissible polynomial metric continuation. Retain quadratic continuation and no production integration. Complete receipt/source/build/input/history/field evidence is indexed by final-report.json.
