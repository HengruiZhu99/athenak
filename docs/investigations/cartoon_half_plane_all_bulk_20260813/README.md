# Half-plane all-bulk SO(2) Kerr experiment

Final verdict: **ALL-BULK STABLE BUT NOT CONVERGENT**.

This report closes the experiment requested from baseline
`c697e0d577fdc41360767f6c73d3ae7e8bedc1b2`.  The source-only all-bulk change
is `2403600b68712c39c2b095a1e6267cb73b07c37c` (tree
`fd1ad1382183d0652c95ff2e38fa7264df8882ab`).  The later analyzer-only JSON
fix is `c9d020bc4a4432bc47c6e0b28bc3c452db44a4db`.

The complete terminal artifact is outside Git at:

`artifacts/axisymmetric_cartoon_z4c_2026-08-10/continuation/cartoon_all_bulk_three_grid_v4_analysis_20260813`

Its root manifest SHA-256 is
`040b7d26ef3f940f2eeb2b1df80e83172e65015b3e80f3a4c5e9aa3ca26937c6`;
the detached-manifest-file SHA-256 is
`d9aa2be3620b7dc80673b7354c2d7dc658fe07d487c9105bb0085b7efb63cc9f`.
`FINAL_REPORT.md` in that artifact contains all eleven requested sections,
with raw data, plots, tables, strict summaries, and both Slurm provenance
layers.

## 1. Files changed

The science commit changes `src/z4c/cartoon_derivatives.hpp`, two operator
design documents, and seven focused/generated/static test files.  The analyzer
fix changes only `cartoon_half_plane_kerr_convergence.py`.

## 2. Removed production logic

Production no longer contains or calls `NearAxisCell`, `TargetLayer`,
`RegularCoefficientDerivative`, `EvenCoefficientDerivative`,
`OddCoefficientDerivative`, `QuadraticCoefficientDerivative`, or
`QuadraticDifferenceCoefficientDerivative`.  No layer-conditioned
`s=rho^2` reconstruction remains.

## 3. No active axis point

Every production provider is constructed with
`CartoonAxisLocation::cell_centered`; the first evolved cell remains
`rho=h/2`.  The explicit diagnostic-axis mode is not used by production.

## 4. All-bulk operator

All active cells use the same centered-parity bulk scalar, vector, and
symmetric-tensor SO(2) quotient identities.  Cartesian3D continues to delegate
to the unchanged Cartesian finite-difference primitives.

## 5. Manufactured result

Fixed physical `rho=0.5M` observes O2/O4/O6 orders `2.00202`, `4.01794`, and
`6.01950`.  At fixed `rho/h=0.5,1.5,2.5,3.5,4.5`, however, the aggregate
observed orders are:

- O2: `1.00353, 1.02360, 1.06340, 1.12298, 1.20380`;
- O4: `3.00937, 3.05592, 3.17905, 3.41412, 3.76419`;
- O6: `5.28528, 5.79523, 6.23194, 6.49450, 6.65647`.

The prospective fixed-layer designed-order gate therefore fails.

## 6. Fresh h48 prequalification

Job `56909189` passed its build/focused-test gates, and the h48 science step
completed `0:0` from `t=0` to `t=5M` in 56 seconds, crossing the previous
failure near `4.56M`.  Its later outer failure was a horizon-carrier parsing
bug after science completion.

## 7. Three grids

Job `56910419` completed fresh h32, h48, and h64 science steps `0:0` in 39,
56, and 86 seconds.  Every history ends exactly at `5M`; accepted horizon
coverage extends past `4.93M` on every grid.  The outer analyzer then failed at
strict JSON serialization; the source-bound offline replay repaired no
numerical data.

## 8. Constraints

All 36 requested `C/H/M/Z` combinations pass: global, axis tube, off-axis,
layers 0--4, and `Linf`.  The minimum time-monotone and positive-order
fractions are both `0.947368`.  At the former seam (`rho=2.5h`), all four
constraint families decrease monotonically across h32/h48/h64.

## 9. Horizons

Direct, flow, and reflection residual gates pass; spin error converges, and
the mean/minimum radii have positive three-grid orders.  Area errors
`0.0510974, 0.0454695, 0.0559315` and horizon-mass errors
`0.00187286, 0.000351272, 0.000489844` worsen from h48 to h64.  Simultaneous
horizon-invariant convergence therefore fails.

## 10. Previous layer-2 failure

The old h48 evolution failed at `4.55625M` with a catastrophic near-axis
Gamma/A--Ricci mode in the special-layer region.  The all-bulk h48 reaches
`5M`, and layer-2 constraints converge on all three grids.  The old runaway is
absent over the requested interval, but this does not cure the independent
manufactured-order and horizon-invariant failures.

## 11. Verdict

**ALL-BULK STABLE BUT NOT CONVERGENT**.  No gauge, damping, KO dissipation,
CFL, floor, clipping, spatial-order, AMR, or threshold tuning was used.
