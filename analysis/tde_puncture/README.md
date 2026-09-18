# Residual puncture stability control

See [RESULTS.md](RESULTS.md) for the measured outcomes and remaining growth
mode. Raw histories, exact control decks, stage-audit summaries and provenance
are under `results/`; binary visualization data remain in the documented run
directories rather than being committed wholesale.

The matter-control candidate is
`problem/bh_background=schwarzschild_trumpet`: the analytic R0=M=1 trumpet of
[Dennison & Baumgarte](https://arxiv.org/abs/1403.5484), equations 15--20.
Areal radius R=r+1, coordinate horizon r=1, psi=sqrt(R/r), lapse r/R,
Cartesian shift x^i/R^2, K=1/R^2, and conformal A_ij=(2 delta_ij/3-2 n_i n_j)/R^2.
This is stationary vacuum geometry; the independent spherical ADM check
`verify_trumpet_continuum.py` reduces both constraints and all independent
metric/extrinsic-curvature evolution components to exactly zero symbolically.

The existing residual Z4c operator and `residual_gauge=background_adapted`
evolve perturbations, retaining the background-adapted 1+log coefficient and
Kappa1=0.1, Kappa2=0. The analytic R0=M slice is **not** the stationary standard
advective 1+log slice (whose limiting areal radius is about 1.312M). The adapted
gauge preserves its stationary background lapse and shift by construction.
For these inputs its lapse equation is
`dt(delta_alpha) = beta_bg . grad(delta_alpha) - 2 alpha_bg delta_Khat`,
with the existing background-adapted Gamma-driver shift and eta=2.
This deliberately adapted residual gauge is not ordinary nonlinear advective
1+log applied to a freely relaxing wormhole.
No custom core extension, state freezing, sponge, or coordinate excision is used.
A cell center at the puncture is rejected rather than regularized. Both
background providers cover ghost cells and use immutable stencil inputs.

The alternate `schwarzschild_puncture` selects isotropic wormhole data:
psi=1+1/(2r), alpha=psi^-2, beta=K_ij=0, coordinate horizon r=0.5.
It is an experimental equilibrium of the **subtracted equations**, not a
stationary solution of unmodified Einstein evolution with this positive lapse.
It does not reproduce ordinary wormhole-to-trumpet moving-puncture relaxation.
Its small-domain perturbation control grew and was stopped; it is retained
only as a diagnostic. The physical atmosphere/star comparisons use the
continuum-stationary trumpet, not this held-fixed wormhole.

## Reproduce

Build `PROBLEM=z4c_tov_ks`, double precision, MPI. Then:

```
OMP_NUM_THREADS=1 python3 tst/regression/z4c_puncture_background.py \
  --background schwarzschild_trumpet \
  --exe /absolute/path/to/athena --output /new/regression/directory
python3 analysis/tde_puncture/make_controls.py --output /new/control/directory
```

The regression checks independent analytic background values (including ghosts),
zero state/RHS/geometry differences at every audited vacuum stage, refinement
transfers, nonzero gauge/matter response, and final active-cell MPI bitwise parity.
It is a three-step correctness regression, not a long stability test.

Long control decks use identical SMR meshes: outer faces at +/-256M,
BH dx=0.125M (16 cells across the initial coordinate horizon diameter), star
dx=0.25M, with a static refinement corridor covering its swept path through
100M (1268 MeshBlocks total). The deck generator rejects longer targets
because that corridor has not been designed for them. The M_BH=2e5 solar masses, one-solar-mass gamma=4/3 TOV model has
M_star/M_BH approximately 5e-6 and radius approximately 2.356M. Initial
areal separation 206.670586M=1.5r_t, isotropic separation 205.670586M,
parabolic test-particle E=1, L=6.666667M, areal periapsis 20M. The generator
checks the coordinate transformation analytically. Its conversion to the
existing boosted TOV inputs neglects the star's small self-metric corrections.
The star is initialized by the existing approximate boosted-TOV metric
superposition, not a binary constraint solve. The input solar mass/radius label
refers to the isolated TOV model; integrated baryon mass on the background
and finite-resolution morphology must also be checked.

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
limit and was not classified as a metric instability. This is the wormhole regression; trumpet regression and long results are recorded separately.

The initial integrated baryon mass in the local star stage test was
5.073394746e-6 M_BH, or 1.01468 solar masses. This is distinct from the
isolated TOV gravitational mass 4.99989e-6 M_BH.
