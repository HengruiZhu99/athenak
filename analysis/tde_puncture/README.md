# Residual puncture stability control

`problem/bh_background=schwarzschild_puncture` selects isotropic Schwarzschild
wormhole data with M=1, psi=1+1/(2r), alpha=psi^-2, zero shift and extrinsic
curvature, and coordinate horizon r=0.5. The existing residual Z4c operator
and `residual_gauge=background_adapted` evolve perturbations. Kappa1=0.1,
kappa2=0 are unchanged. No custom core extension, state freezing, interior
sponge, or coordinate excision is used. A cell center at the puncture is
rejected rather than regularized. Both background providers cover ghost cells.

This is an experimental equilibrium of the **subtracted equations**, not a
stationary solution of unmodified Einstein evolution with this positive lapse.
Subtraction removes continuum gauge/geometry relaxation as well as background
truncation error. It therefore does not reproduce ordinary wormhole-to-trumpet
moving-puncture evolution. Stability would not by itself validate the continuum
physical response of this fixed-background construction. In particular, a
physical star is a diagnostic of the existing TOV superposition, not a solved
binary constraint problem or a validated disruption calculation.

## Reproduce

Build `PROBLEM=z4c_tov_ks`, double precision, MPI. Then:

```
OMP_NUM_THREADS=1 python3 tst/regression/z4c_puncture_background.py \
  --exe /absolute/path/to/athena --output /new/regression/directory
python3 analysis/tde_puncture/make_controls.py --output /new/control/directory
```

The regression checks independent analytic background values (including ghosts),
zero state/RHS/geometry differences at every audited vacuum stage, refinement
transfers, nonzero gauge/matter response, and final active-cell MPI bitwise parity.
It is a three-step correctness regression, not a long stability test.

Long control decks use identical SMR meshes: outer faces at +/-256M,
BH dx=0.0625M (16 cells across the initial coordinate horizon diameter), star
dx=0.25M. The M_BH=2e5 solar masses, one-solar-mass gamma=4/3 TOV model has
M_star/M_BH approximately 5e-6 and radius approximately 2.356M. Initial
areal separation 206.670586M=1.5r_t, isotropic separation 205.669371M,
parabolic test-particle E=1, L=6.666667M, areal periapsis 20M. The generator
checks the coordinate transformation analytically. Its conversion to the
existing boosted TOV inputs neglects the star's small self-metric corrections.
The background is held fixed, so these geodesic values describe the initial
Schwarzschild slice; the modified evolution must not be assumed to preserve
the physical Schwarzschild geodesic thereafter.

Atmosphere/star controls keep matter feedback enabled, with density floor
1.6e-21 and pressure floor 1.6e-33 in BH units. Zero-residual vacuum controls
suppress only matter feedback; the fluid floor still evolves. All runs retain
metric-validity checks, exterior physical-constraint histories, unexcised
Theta/gauge maxima, and density/residual slices. Reaching 100M is a pilot,
not full disruption or proof of long-time stability. Constraint amplification,
finite values, target completion versus walltime, and stellar density must be
reported separately.

The first local regression passed all 12 combinations of uniform/SMR,
zero/lapse-pulse/atmosphere, and 1/4 MPI ranks. The regression atmosphere is
1e-14, below the boundary matter limit; an initial 1e-12 test correctly hit that
limit and was not classified as a metric instability. Long results are pending.
