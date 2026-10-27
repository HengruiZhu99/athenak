# Completed coupled 3D family versus the matched selected radial pilot

All three cases completed genuine full3D tensor evolution with RK4 to
t=0.0002 and remain **scientific FAIL**. Angular averaging is used only for
these postprocessing diagnostics. No evolution field was radially projected,
and no source, input, PDE header or original output was modified here. The
cases have different shell grids, block resolutions, enrichment and solver
controls; these figures are **not a uniform-resolution convergence study**.

The reference is the existing independent `selected-taper-pilot-20261027`
radial N2049 trajectory, with its N1025 numerical-resolution control. Both use
the matched telegraph taper width.24, outer radius.8, AF-regular window,
kappa.005, tau.02, nu1, M=1/3, Minkowski reference, dt=.0001 and2 steps. Its
recorded spatial control is centered/zero-KO. This is a short matched pilot,
not reproduction of the selected long t10 collapse benchmark. Reference
parameters, original commands and every numerical input are hashed in
`comparison.json`.

## Physical fields and explicit quadrature

The eight compared physical channels are alpha, chi, g_rr, A_rr, Khat, Theta,
independent Lambda_r and beta_r. Tensor/vector radial components are Cartesian
contractions used for comparison, not imposed spherical symmetry. Physical
Cartesian beta is also reconstructed from the stored delta_beta minus x and
compared in every cell, including its tangential part.

Every original active cell enters the full-domain norm. The supplied weights
are exactly the pilot's **Penrose spatial proper-volume quadrature**:

    w_p = map_cell_volume_p * sqrt(det(g_tilde_p)) / chi_p^(3/2),
    map_cell_volume_p = 1 / (N³ * abs(det(dq/dx)_p)),
    gamma_bar_ij = g_tilde_ij / chi,
    RMS(e) = sqrt(sum_p w_p e_p² / sum_p w_p).

There is no extra Omega^(-3) factor, floor, clipping, radial exclusion or
renormalization after dropping cells. This is Penrose volume, not the physical
Omega^(-2) spatial metric's divergent volume at scri. Unnormalized L2,
maximum error, signed bias and volume fractions are retained too. Beta vector
errors use Euclidean lengths of common Cartesian components before volume
weighting; they are not metric-length norms.

| Case | Active cells | Penrose volume quadrature | alpha RMS error | beta_r RMS error |
|---|---:|---:|---:|---:|
|N8 enriched,3 shells|13312|7.50848|2.57499e-5|3.18224e-2|
|N8 enriched,7 shells, radial lines|25600|7.50800|4.29455e-7|5.16875e-4|
|N6 enriched,7 shells, core AMR3|47088|40.6552|3.82664e-7|4.12883e-4|

Seven shells improve the shift and transition comparison substantially. The
AMR case is not uniformly better: A_rr/Khat/Lambda_r RMS and several maxima
worsen relative to N8 seven shells. In particular, its lower shell resolution
and different interfaces prevent attributing every change to refinement.
The N1025-minus-N2049 reference change is much smaller than the observed3D
errors in every channel. For example beta_r's reference-control RMS is
4.04e-11,2.12e-11,7.78e-12 on these three distinct meshes. This difference is a
control, not a proved continuum error bar or a Richardson order estimate.

## Constraints, boundaries and solver residuals

The raw all-cell CSV quadrature independently reproduces the reported H, M
and Theta norms. Z and original elliptic residuals remain explicitly labeled
stdout-reported global quantities: the CSV lacks the metric derivatives and
local source/shift jets needed to recompute them. Independent Lambda is not
substituted for the metric-connection difference or Z.

| Recorded diagnostic | N8,3 shells | N8,7 shells | N6,AMR3 |
|---|---:|---:|---:|
|H RMS|.510563|.509059|.0298108|
|M RMS|.140161|.0283855|.0328553|
|Theta RMS|9.75267e-6|5.43691e-6|9.31226e-7|
|Z RMS, global only|4.04412e-4|1.18209e-4|9.19132e-5|
|Original shift RMS maximum over stages|5.97101e-11|7.06070e-11|2.72146e-11|
|Scri tensor pole, target1e-7|3.93105e-6|1.46164e-6|1.44183e-5|

The original shift solves satisfy their1e-10 RMS and1e-7 maximum-component
controls. The prescribed scri clock and future-null/tangential ghost residuals
also pass their original bounds. All three nevertheless fail the unchanged
1e-7 tensor-pole gate. An accurate algebraic solve therefore does not certify
the coupled formulation at these resolutions.

Regional H/M/Theta and all eight field errors are retained for the core,
Cartesian interior, each shell layer, transition, scri, block faces/edges/
corners and Cartesian coarse-fine stencil neighborhoods. These overlapping
regions supplement, and never replace, the all-cell acceptance norm. Regional
Z and local original shift residuals are marked unavailable rather than
invented from incomplete data.

The AMR volume quadrature is5.42 times the N8 value: the core contributes84.4%
of its sampled Penrose volume, versus15.9% for N8. Thus a cross-mesh RMS ratio
also changes the sampled measure near the unresolved inner infinity.
Unnormalized Hamiltonian L2 falls from1.39486 to.190078, a factor7.34, while
normalized RMS falls by17.1. This distinction is retained instead of labeling
the latter as a convergence factor. The full Hamiltonian maximum still reaches
.406127.

## Actual final Hamiltonian localization

The earlier localization record at node34308 has H=.406305 at **t=0**. The
final t=.0002 CSV maximum is a symmetry-related coarse-fine node44676:

    xyz = (-.0598958333, .00260416667, .0598958333),
    r = .08474552133,
    H = .406126702181,
    Cartesian level =3, block206, logical(i,j,k)=(0,0,5),
    h = .005208333333, rho/h =16.2711401.

At t=.0002 the old node34308 instead has H=.406124435386. Both lie inside the
radius.1 neighborhood and outside the separate rho/h<=8 enrichment support
criterion. The independent geometric support reconstruction exactly matches
the recorded enriched-cell counts136,136,2728. The final maximum lies in the
identified coarse-fine stencil-neighbor mask. This localizes a failure; it
does not by itself prove the stencil's causal mechanism or solve it.
`AMR-maximum-H-audit.json` preserves both times/locations and the unchanged
older metadata.

## Figures and retained numerical data

* `physical-profiles.png`: all eight matched radial profiles; full3D angular
  means and raw1-sigma scatter from64 directions at each of19 radii.
* `physical-error-budget.png`: full-cell RMS/maxima, numerical reference
  control, angular scatter and angular-mean errors.
* `constraint-regions.png`: all-cell/regional H/M/Theta and the separately
  labeled global Z/original-shift/scri diagnostics.
* `hamiltonian-localization.png`: every active cell's |H| versus radius plus an
  AMR diagnostic slice with Cartesian block boundaries. Only the illustration
  selects a slice; all numerical norms retain every cell.

Raw angular values, minima/maxima and variance decomposition are preserved;
scatter is not assumed to be a physical nonspherical mode, since mapped-grid
extraction can contribute. `*-all-cell-errors.npz` retains every coordinate,
weight, physical/reference field, error, N1025 control, H/M/Theta and full beta
comparison. `*-raw-angular.npz` retains every direction. Original raw CSVs
remain untouched and are bound by their input hashes.

The new script `tst/elliptic_shift_3d/compare_coupled_family.py` completed in
16.28 seconds under a60-second subprocess bound, with every input hash unchanged.
Matplotlib/font caches stayed inside the new analysis directory and were
removed. `executed-analysis-source.py` matches the numerical run's recorded
script hash. A documented presentation-only revision removes an inappropriate
negative automatic axis range from the |H| plot; no data or norm changed.
No scientific threshold, field, gauge or source was altered.

Reproduce into a fresh output directory with:

```sh
python3 tst/elliptic_shift_3d/compare_coupled_family.py \
  --output build-shift-3d/new-coupled-family-comparison
```
