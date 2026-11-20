# Characteristic extraction from hyperboloidal checkpoints

The `athenak-cce` Python package reconstructs a timelike spherical worldtube from the physical metric on accepted AthenaK checkpoints. SpECTRE evolves the exterior characteristic initial-boundary value problem and outputs the strain, news, five leading Weyl coefficients, and angular derivative of inertial retarded time at null infinity. This provides an independent extraction alongside [native endpoint news](scri-news.md).

## Extract the worldtube and evolve the exterior

Install the Python package into an analysis environment and unpack a [SpECTRE CCE release](https://spectre-code.org/tutorial_cce.html) for the compute platform. These commands use the refined head-on parameters. Each output path should be new.

```sh
python -m pip install ./vis/python/athenak_cce
athenak-worldtube metric.h5 --radius 0.90 --mass 0.3333333333333333 \
    /path/to/trajectory/segment-*/rst/*.rst
athenak-cce metric.h5 /path/to/CceExecutables exterior
```

Input is the shared, double-precision version-1 native Z4c restart with a cubed-sphere v2/v3 atlas. Supply one trajectory on a fixed topology. The reader restores the determinant-one spatial metric and the native lapse and shift offsets from bulk samples. For a tube at compact radius `r0`, it evaluates the physical ADM metric with the archived conformal profile. In units of the supplied mass, the coordinates are `T=t/M0` and `(X,Y,Z)=(y,z,x)/(M0*Omega(r0))`. This constant spatial scale applies throughout the worldtube stencil, so SpECTRE's radial derivative and extraction radius refer to the same Cartesian coordinates. Its polar axis is the AthenaK x axis. The Bondi areal radius follows from the angular metric.

The converter uses up to six radial nodes in the selected block and a solid-angle-weighted harmonic fit for each Cartesian component. It takes the time derivative of the modal coefficients on five actual checkpoint times, including one-sided stencils at the endpoints. SpECTRE consumes all 30 metric, radial-derivative and time-derivative datasets. The extraction precision depends on angular and radial resolution, checkpoint cadence, and the initial outgoing null hypersurface. Compare worldtube radii and inspect initial transients.

`athenak-cce` takes initial-null-data and error-control settings from the installed release. It writes both resolved YAML files and accepts a custom SpECTRE evolution input with `--config`. The default observes all supplied modes, retains at least the release evolution bandlimit and six extra angular degrees, filters the highest two, and uses fourth-order barycentric rational time interpolation. The HDF5 reduction file retains the independent CCE news and all original SpECTRE conventions. Inspect the text evolution log for progress and open the reduction file after the writer exits. Concurrent HDF5 reads can block the next SpECTRE output.

## Choose and check a remnant frame

Choose a late cut in a settling interval with some data on both sides. The cut and half-width use the **CCE output Bondi time in units of M0**. The example below illustrates the arguments. Select the cut from the reduction file for each run.

```sh
athenak-bondi exterior/CharacteristicExtractReduction.h5 analysis \
    --cut 100 --half-width 5 --mode 2 0 --keep-axes
```

The SXS `scri` library converts the CCE fields to Moreschi–Boyle conventions and determines the BMS transformation from Moreschi supermomentum and the center-of-mass charge. For the nonspinning head-on problem, `--keep-axes` retains the collision axis. For a spinning remnant omit that flag to include spin alignment. The time origin matches the SpECTRE output. Use `--spacing` to interpolate the analysis onto regular Bondi cuts. The original reduction file preserves every adaptive CCE observation.

The analysis writes the original-frame diagnostics, the BMS transformation, transformed diagnostics, a curvature plot and a JSON record of the cut and residuals. The default frame-fit criterion is that all three residuals at the cut are at most 0.001: speed, center-of-mass displacement in units of M0, and the norm of the proper Moreschi supermomentum modes divided by the monopole magnitude. Fits over tolerance are labeled `frame-fit`. Inspect the reported mass variation, late radiation, extraction-radius dependence and balance residuals separately to assess whether the cut represents the physical remnant.

In the NPZ files, `strain` and `news` start at ell=2. The five `psi` arrays and `moreschi` start at ell=0, with mode index `ell*(ell+1)+m`. Weyl arrays use Moreschi–Boyle normalization. Plots show SpECTRE normalization, twice the stored psi4. The news is the CCE observation transformed with the fitted BMS transformation. Diagnostics include the strain/news and curvature/news consistency checks, Bondi mass, four-momentum, angular momentum, spin, boost and center-of-mass charges, four Poincaré fluxes, the four-momentum balance defect, and six Bondi constraint norms in scri's order. The original HDF5 also contains `EthInertialRetardedTime`, which describes the coordinate cuts of that CCE evolution.

See the [SpECTRE characteristic extraction formulation](https://arxiv.org/abs/2110.08635) and the [scri asymptotic Bondi data tutorial](https://scri.readthedocs.io/en/latest/tutorial_abd.html).

For an independent comparison with the existing native news archive, run
`python -m athenak_cce.native news.npz native-psi4.png`. It differentiates
the complex news modes with respect to the archived Bondi time using scri and
plots the result in the native initial extraction frame. Each extractor
determines its own inertial cuts, so applying a CCE frame transformation to
the native archive requires first identifying the BMS map between the two
initial frames.

## Verification

Run `python -m pytest tst/z4c_hyperboloidal/test_cce_{frame,worldtube}.py`
after installing the package. The analytic ADM fixture checks both harmonic
phase and derivative units. The frame test deliberately translates, boosts,
and supertranslates a unit-mass Schwarzschild solution and checks the recovered
charges. Set `SPECTRE_CCE_RELEASE` to the unpacked release directory to also run
the native-checkpoint-to-CCE test on tilted Schwarzschild slices for the mass monopole and all
six vanishing radiation or nonspherical Weyl outputs.

## Refined N12 head-on example

The two extractions of the 2026-11-19 head-on use 203 shared checkpoints
spanning native time 0–120 M0. SpECTRE v2026.11.17 evolved through ell=20 and
observed through ell=8. The compact radii 0.84 and 0.90 correspond to mean
Bondi areal radii of roughly 24–27 and 28–31 M0, respectively. Both use
`CauchySecondOrder` initial data. At 0.84 the initial angular solve reached
2.13e-10 in max abs(J0), so its input uses `MaxPartiallyFlatJ0: 1e-9`.
The 0.90 extraction passed the release default of 5e-12.

Both frame fits retain the collision axis and use cut 130 M0, half-width 4 M0
and analysis spacing 0.125 M0. The table uses SpECTRE curvature conventions.
The minimum is taken between 20 and 130 M0.

| Compact radius | Minimum Re(M0 Psi4_20) | Time of minimum / M0 | Cut mass / M0 | Proper supermomentum residual at cut |
| --- | ---: | ---: | ---: | ---: |
| 0.90 | -0.01512 | 68.86 | 1.1097 | 2.94e-7 |
| 0.84 | -0.01553 | 68.33 | 1.1163 | 8.59e-6 |

Using the output time origins, the relative L2 difference between the complex
quadrupoles over 20–130 M0 is 26.8%. The two main troughs differ by 2.7% in
depth. On the outer tube the Bondi energy increases by 0.0306 M0 while the
news carries away 0.00143 M0 over that interval. The fitted outer
supermomentum residual grows to about 0.04 at either edge of the fit window.
These errors limit interpretation of the absolute charges and of a single
frame as stationary throughout the late waveform.

As a normalization check, the independent unit-mass tilted Schwarzschild
fixture passed through the converter and the same SpECTRE release. It gave
a mass of 1.000143 across its cuts, maximum news mode amplitude 1.73e-11,
and maximum Psi4 mode amplitude 2.52e-11.
