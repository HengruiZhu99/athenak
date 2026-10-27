# Targeted radial discrete-consistency diagnostic

This is a bounded static diagnostic, not a solve/evolution success. It uses the
actual M=1/3 embedded isotropic wormhole data and the independent8193-node shift
integral from `wormhole-localization-20261027/focused-exact.dat`. The existing
datum receipt records a maximum4097/8193 shift-ratio difference2.28e-14. No
curvature, lapse or gauge source is chosen to make a numerical shift stationary.

The grid has N8, eight Cartesian blocks, eighteen shell blocks, shell radii
.62/.70/1, 13,312 active cells and four ghosts. Separate sequential runs select
ordinary or physical-radial interface interpolation. No Krylov solve, timestep,
background RHS subtraction, lapse modification or production mode is involved.
Every process has a180-second timeout; actual build/run costs are1.67/2.28/2.61s.
Original1D jobs are untouched.

Reproduction from the repository root:

```sh
python3 tst/elliptic_shift_3d/run_radial_metric_divergence.py \
  --build build-shift-3d \
  --data docs/elliptic-shift-3d/evidence/wormhole-localization-20261027/focused-exact.dat \
  --output build-shift-3d/radial-consistency-reproduced
```

The driver reuses the existing Kokkos compile/link flags without editing CMake.
`manifest.json` contains the complete compiler argument vector, source/data and
binary hashes, process exits, timings and complete results. No process timed out
and source/binary bytes remained unchanged during each run. Matching copies of
the recorded diagnostic/kernel sources are under `source/`. The later unrelated
telegrapher-support helper was recovered from HEAD for this archive because
that HEAD version exactly matches the diagnostic's recorded hash. The first
four-mode diagnostic is preserved separately at
`../radial-metric-divergence-20261027`; its exact pre-oracle source is also archived.

## Negative result: matching DV alone is not a fix

The physical curvature rate remains `C_ij=-2 alpha A_ij` from the actual samples
at every cell and the actual zero outer A datum. Operator/source variants are:

1. Existing compact operator, exact analytic source.
2. Existing compact operator, product-rule source from sampled alpha/A derivatives.
3. Centered `DV_D[H_D(beta)]` with the same `DV_D[C]` source.
4. Forward/backward-adjoint `DV_D[H_D(beta)]` with the same `DV_D[C]` source.

With radial-interface interpolation, all-cell physical-volume residuals are:

| Variant | RMS | Point maximum |
| --- | ---: | ---: |
|Compact operator, analytic source|2.3444553|10.961854|
|Compact operator, sampled product-rule source|2.9846584|12.625569|
|Consistent centered composition|3.0071849|18.981015|
|Consistent forward/backward pair|86.473824|508.09143|

With ordinary interfaces the biased pair is worse still: RMS441.93022,
maximum6107.5359 at r=.625. The centered composition gives RMS6.13566.
This rejects adoption of either prototype as the radial fix. Removing a flat
checkerboard null in a symbol calculation does not establish useful mapped,
strong-gradient behavior. The compact operator remains the control.

## Independent first-jet oracle localizes the defect

Two additional controls replace **only beta's first spatial jets** by the
independently supplied radial shift jets. They still evaluate the full
`ShiftMetricRate(g,beta)` formula and retain the same sampled physical alpha/A,
outer DV stencil, tensor halo transfers and direct vector boundary treatment.
This does not set H=-C in the code and is not a usable unknown-shift operator.
The relation follows from the independent radial metric-freezing solution.

The exact continuous vector equation is satisfied to1.14e-13 pointwise. With
radial interfaces, the exact-beta-jet controls reduce the centered/biased
residual RMS to2.54e-14/9.07e-14, and maxima to2.56e-13/1.02e-12.
Their metric-rate defect `H(beta)-2 alpha A` is5.08e-16 RMS.

In contrast, sampled centered first derivatives have metric-rate defect
.118980 RMS, maximum.647069 at r=.597117. The biased first derivative increases
that defect to.696913 RMS, maximum4.80534 at r=.606109. Scri metric-rate defects
are only about5e-14, and independent reconstruction of the actual vector ghost
boundary has maximum6.72e-30. Thus, for this diagnostic, the dominant failure
arises while constructing the shift-induced metric rate from first derivatives
near the transition, rather than from a scri datum or from the algebraic
Frechet variation itself. This does not independently prove the outer DV
stencil accurate for arbitrary nonzero tensor rates; the smooth general-metric
MMS and finite-metric variation audit remain separate controls.

`*.points.csv` retains every active cell. Full physical-volume RMS/L2, conformal
volume RMS, point maxima, six radial regions, all-block interface bands and a
scri band are recorded in JSON. Acceptance cells are never excluded. Weights
are this diagnostic's own finite-grid `DeltaV/chi^(3/2)`; normalization of a
solver's stored weights does not change these RMS ratios.

The next small falsifiable control is to replace only internal beta halo values
in the first H construction with exact sampled values, retaining active beta,
physical source and outer DV unchanged. That separates interface interpolation
from underresolved same-patch first derivatives before any new evolution run.
