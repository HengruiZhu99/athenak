# Offline news from native scri data

The recorder supplies accepted endpoint geometry and its native derivative
support. The offline tools reconstruct the Penrose four-metric and estimate
radiation in a transported inertial frame. They do not modify AthenaK state,
change the evolution scheme, or extrapolate a finite-radius waveform.

This is a **preliminary extraction at the simulation's resolution**. Analytic
extractor checks do not establish convergence of the evolved N12 spacetime.
Frame, shear, nullness, angular-fit, and temporal-sampling errors must accompany
the waveform. A failed or large diagnostic must not be hidden by relabeling
native-coordinate field modes as radiation.

## Definition and frame

The reported normalization is

```
H = lim R (h_plus - i h_cross)
N = dH/du_B
```

`N` is dimensionless. The time axis is `u_B/M0`, with `M0=1/3` for this binary.
The polar axis is Cartesian **x**, along the initial head-on separation, so the
dominant axisymmetric component is `(ell,m)=(2,0)`. The initial scri cut is assigned
`u_B=0`, and its native angles set the initial inertial axes. No strain integration
constant, supertranslation adjustment, or memory subtraction is inferred.

The conformal extraction and inertial-frame construction follow the geometric
framework of [Babiuc et al., arXiv:0808.0861](https://arxiv.org/abs/0808.0861) and
[Handmer et al., arXiv:1605.04332](https://arxiv.org/abs/1605.04332). Our strain/news
normalization is twice the complex conjugate of the news convention in Eq. (3.22)
of the first reference. This factor and polarization sign are checked against an
explicit radiative asymptotic metric, rather than inferred from a numerical plot.

Let `g` be the archived Penrose metric, `Omega` its prescribed conformal factor,
and `n^a=g^{ab} partial_b Omega`. Starting from a round unit-sphere cut, the code
transports `lambda=log(omega)`, inertial direction labels `y^I`, and time `u_B`:

```
v^i = n^i/n^t
(partial_t + v^i partial_i) lambda = -Box(Omega)/(4 n^t)
(partial_t + v^i partial_i) y^I = 0
(partial_t + v^i partial_i) u_B = exp(lambda)/n^t
```

This chooses the preferred conformal gauge for `g'=omega^2 g` and
`Omega'=omega Omega`. In the continuum vacuum limit the transported cuts are
round. The implementation reports their actual deviation from roundness.
Its initial-round-cut requirement is checked explicitly; arbitrary initial-cut
uniformization is not implemented.

The full inverse coordinate Jacobian constructs angular vectors at fixed
`u_B` and `Omega'`, including their native-time components. Their polarization
is anchored to inertial north. A small Gram--Schmidt correction makes the dyad
null and normalized in the actual reconstructed metric; the uncorrected cut
metric defect remains in the diagnostics.

In the admitted inertial conformal frame the estimate is

```
N = -R'_ab mbar'^a mbar'^b
```

The trace term in the Schouten tensor vanishes in this contraction. The
transformation of `R_ab` includes both first and second derivatives of `lambda`,
including time and mixed derivatives. The geometric identification assumes the
usual vacuum regularity at scri; finite violations in the evolved data remain
an extraction uncertainty.

## Numerical path

- `scri_collect.py` restores accepted endpoint jets using the archived redundant
  samples and normal lifts. Independent reconstruction from their donor support
  is available in `scri_reader.py` and must be checked when admitting a dataset.
  The temporary metric cache requires fixed native surface topology and retains
  the source record/cycle/time index. Source archives remain read-only.
- `scri_geometry.py` removes storage offsets, forms the four-metric, and computes
  its connection and Ricci tensor. Time derivatives use degree-six polynomials
  on seven actual accepted times; spatial derivatives use the recorded native
  endpoint construction.
- `scri_frame.py` transports the frame with RK4 and cubic interpolation of the
  sampled geometric coefficients. Scalar angular interpolation uses all real
  harmonics through the requested cutoff; it is separate from evolution.
- `scri_news.py` evaluates the nodal news estimate, interpolates it in spin `-2`
  harmonics on transported angles, and resamples each inertial generator to a
  common Bondi-time interval. Six-point polynomial time interpolation avoids
  extrapolation. Final modes use Gauss--Legendre/azimuthal quadrature. The first
  and last three native records are excluded from the radiation interval.

`--stride 2` checks sensitivity to time sampling; `--lmax 6` against `--lmax 8`
checks sensitivity to angular interpolation. Neither replaces a second spatial
resolution of the actual binary evolution. Unresolved angular content is
reported as a nodal fit residual.

## Running and checking

```sh
python3 tst/z4c_hyperboloidal/scri_geometry.py
python3 tst/z4c_hyperboloidal/scri_frame.py
python3 tst/z4c_hyperboloidal/scri_news.py
python3 tst/z4c_hyperboloidal/scri_checkpoint.py --checkpoint CHECKPOINT --record RANK_ZERO_RECORD --output CHECK_REPORT
python3 vis/python/scri_collect.py --root RUN_ROOT --output METRIC_CACHE
python3 vis/python/scri_news.py --cache METRIC_CACHE --output NEW_RESULT_DIRECTORY
```

Only NumPy is required. Collection can resume as complete rank-local records
appear. Partial final output is retried, while malformed earlier records and
mixed rank/time identifiers are errors. The metric cache is disposable analysis
data; preserve the original recorder files for other reconstructions.
Collection also rejects mixed evolution identities, source manifests, field
definitions, conformal profiles, or recorder configurations. Use `--mass` on
`scri_news.py` when `M0` differs from this binary's default `1/3`.

The analytic checks cover Ricci conventions on curved product/FLRW geometries,
zero radiation, a known quadrupolar shear derivative, time/angle-dependent
conformal rescaling, rotating coordinates, nonuniform time differences, mixed
derivatives, scalar/spin harmonic normalization, and common-Bondi-time extraction
from angle-dependent native-time cuts. The latter recovers the known `(2,0)`
news to about `2e-12` in the local test.
