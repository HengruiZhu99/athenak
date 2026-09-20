# Direct-constraint pulse: ready inputs and local preflight

The primary input is **`theta_loweta_coremask.athinput`**: a regular
Gaussian Theta monopole of amplitude1e-6/M and width384M, evolved with G=1,
kappa1=0, shift_eta=.02 and lapse-residual damping=.01. It targets50000M with
dt3.2M on the existing64³ uniform cube. The optional matched ablation
`theta_loweta_lapse01_coremask.athinput` restores only lapse damping to.1.
Both retain the wide radial residual sponge starting512M and reaching rate
.001/M at1792M, with the nearest boundary at2048M. No physical fields are
reset or clipped by this setup.

These are **vacuum Minkowski-background tests**, not black-hole or stellar
evolutions: the inherited input has bh_mass=0, coord/minkowski=true,
force_minkowski_metric=false, and zero matter feedback. The spacetime fields
still evolve. The mesh has eight32³ blocks, dx64M, D6/RK3, linear ghosts and
the unchanged original zero_rate characteristic boundary. No C++ or existing
job was changed by this preparation.

## Seed and why it differs from the existing gauge control

`SeedOuterSpongeThetaPulse` adds to the active residual Theta only:

```
Theta(r) = 1e-6 * exp[-r²/(2*384²)]
outer_sponge_test_theta_pulse_radius = 0
outer_sponge_test_theta_pulse_width = 384
outer_sponge_test_theta_pulse_dipole_axis = 0
vacuum_gauge_pulse_amplitude = 0
characteristic_test_family = none
```

The Gaussian is smooth at the origin. Its one-sigma width is six cells;
its diameter at half maximum is904.25M, about14.1 cells. The active-grid peak
is9.896374e-7/M, far above roundoff. `seed-design.json` records the exact-grid
sampling and `initial-profile.json` independently reads the initialized MPI
checkpoint using the existing radial-profile helper.

| Radius | Theta at initialization |
|---:|---:|
|0M|1e-6/M|
|512M, sponge onset|4.1111e-7/M|
|1792M, ramp end|1.8664e-11/M|
|2048M, nearest boundary|6.6584e-13/M|

Initially69.07% of the volume integral of Theta² lies in the protected core
and30.93% in the ramp. The Gaussian is not compact: the layer is deliberately
seeded as well as receiving the outgoing pulse. Consequently an early boundary
signal cannot automatically be interpreted as a reflected wave. The existing
GPU lapse-pulse control is different: its compact support lies entirely inside
r512M (maximum radius399.1M); it has no Gaussian tail in the sponge.

This is an intentional **constraint violation**, not a constraint-consistent
gauge perturbation. Khat remains initially zero, so physical K=Khat+2Theta
changes by2Theta. In the initial flat continuum geometry, A=0 and Q=0,
H=(8/3)Theta² and M_i=-(4/3)partial_i Theta. The initial histories confirm
nonzero Hamiltonian and momentum norms. Gauge and metric residuals are initially
zero and subsequently respond physically to the seeded violation.

An x/r dipole at radius0 was deliberately not used: the current seed lacks the
additional radial factor needed for a regular l=1 scalar at the origin. An
off-center Gaussian shell also leaves a radial cusp unless its origin tail is
negligible. Neither is required for this first low-frequency scalar test.

## Time integration and checks already performed

The preserved pgen source safeguard reports maximum sponge rate.001/M and
absolute source_dt limit1000M. This does **not** automatically bound shift_eta:
the current z4c spatial estimator is dx-only. Thus the standard eta2/lapse.1
comparison retains cfl=.009375, dt=.6M (eta*dt=1.2). The low-eta cases use
cfl=.05, dt3.2M (eta*dt=.064); the temporal comparison uses cfl=.025, dt1.6M.
Lapse damping.1 at dt3.2 gives a source product.32. These source products are
within the RK3 negative-real-axis stability interval. The small wave CFL still
requires measured validation; it is not established solely by a damping cap.

All local tests used eight MPI ranks, one block per rank and one thread per
rank, on the exact64³ domain. Immutable executable SHA256:

`67ff395e3c7af43425ef13fe4f0ebecef00256db9450f6ac912f5ab39620f1f8`.

- Standard and low-eta zero controls completed three cycles with every saved
  active/ghost residual exactly zero. Their direct-Theta pulse gates were finite.
- The optional lapse.1 ablation passed its own zero and pulse gates. Both final
  diagnostic-mask zero inputs also passed three cycles exactly.
- Low-eta dt3.2 and dt1.6 both reached320M, with all raw checkpoint payloads
  finite and full/ghost metric principal minors positive. Relative active
  residual L2 difference was8.4050e-8; the Theta L2 difference was1.00624e-7.
  This is early-time timestep consistency at one spatial resolution, not a
  measured convergence order or a long-time stability result.
- The dt3.2 case took101 cycles because of a final1.25e-12M target-alignment
  step; the dt1.6 case took200. The saved final checkpoint dt for the former
  is that tiny final step, not the intended main timestep.

See `preflight-results.json`, `coremask-validation.json` and each run's
`provenance.json`, `exit_code.txt`, `run.log`, and `checkpoint-validation.json`.
The temporal pair cost about101s/262s locally; neither was walltime-limited.
No local50000M run was launched. Its feasibility on Aurora must use measured
GPU throughput, and a clean walltime checkpoint is not target completion.

## Core diagnostic mask and measurements

The `*_coremask.athinput` files set the existing output-only options
`history_excise_ks_horizon=true`, radius512 and spin0. This is a **core diagnostic
mask**, not physical excision. Coordinate excision and inner cleaning remain
off. The masked/unmasked low-eta three-cycle pulse checkpoints have identical
evolved payload bytes on all eight ranks: MHD, magnetic faces and all25 Z4c
fields, including ghosts. The initial masked core/exterior Theta² integrals
sum to the unmasked value within3.5e-14 relative reduction error.

Every10M, use the actual history labels as follows:

- `Theta-int2`: protected-core integral of Theta². Plot its square root or
  normalize the integral to its own initial value. It is not a core RMS unless
  a separately measured core proper volume is supplied.
- `Theta-norm` (truncated `Theta-norm2`): exterior integral of Theta².
  `sqrt(Theta-norm/Volume)` is **exterior** RMS because Volume is exterior-only.
- Sum `Theta-int2 + Theta-norm` for the whole-domain squared norm. Do not
  divide this by the exterior Volume and call it a global RMS.
- H and M have corresponding exterior/interior integrals. Track them with
  Theta, lapse, shift and metric residuals; small Theta alone is not sufficient.

Full-volume per-rank Theta dumps every64M and constraint dumps every128M
support radial profiles and propagation plots. Raw double checkpoints every
1000M preserve all evolved variables and ghosts; all outputs use per-rank files.
`../radial_profiles.py` separates r<=512,512<r<1792,r>=1792 and an overlapping
physical-face band. Evolved Gamma residual is not the physical Q=Gamma-Gamma_metric
constraint; use the actual constraint output for that distinction.

Monitor outward-moving extrema and signed radial Theta profiles first, then
core recurrence and late growth over5000–10000,10000–20000 and20000–50000M.
The broad Gaussian and pre-existing boundary tail prevent assigning a sharp
arrival time. A spherical leading-order proxy using psi=rTheta can separate
outgoing/incoming signals via partial_t psi minus/plus partial_r psi; label it
as a flat scalar-wave proxy, not the full nonlinear Z4c characteristic flux.
Compare a core recurrence with boundary-localized growth and propagation before
calling it a reflection coefficient. No outgoing-only initial velocity was
imposed: the seed excites the coupled system and both wave directions.

The practical late-time budget is gamma<=ln(2)/50000=1.38629e-5/M, fitted only
above measured noise. Quantify residual secular drift and finite-time norm
amplification as well; saturation or merely finishing does not establish
stability. Any invalid full/ghost metric, nonfinite payload, or inconsistent
checkpoint invalidates a continuation. On a clean cap stop, preserve the final
checkpoint and report its actual time separately from the50000M target.

## Usable inputs and verification

Primary: `theta_loweta_coremask.athinput` and `zero_loweta_coremask.athinput`.
Ablation: `theta_loweta_lapse01_coremask.athinput` and its matching
`zero_loweta_lapse01_coremask.athinput`. The filename `lapse01` means0.1.
The standard eta2/.1 counterpart and unmasked versions remain separate inputs;
the short timestep inputs were not changed by the output mask.

`input-manifest.json` contains every immutable input hash. The primary hash is
`8da1b3ee0a101841c196e00346a527a458e0461ef8d2845fb4d70f6894953bb8`.
The ablation hash is
`605c2a02d2a155b4e2c5ed8d4873e41fea70281048112f8cf1a6921722a4f2bc`.
`prepare_inputs.py` refuses to overwrite an existing different input.

Use the existing fail-closed raw validator, with the regression-reader path:

```
export ATHENA_REGRESSION_PATH=/path/to/athenak/tst/regression
python3 ../gpu/check_minkowski_checkpoint.py RUN --ranks 8
python3 ../radial_profiles.py RUN --ranks 8 --output profiles.json
```

Add `--exact-zero` for zero gates. Require all eight rank headers to agree,
every payload field to be finite, positive full/ghost metric minors and a
checkpoint matching the final application record. Check `target_reached` and
`stopping_reason` explicitly. Initial snapshots and finite active histories
cannot substitute for a final raw checkpoint check.

This compact archive omits raw checkpoints and binary outputs. The preparation
script uses the packaged `gpu/current` seed. Historical `run_preflight.py` and
readback scripts retain the original host-specific executable/reader locations;
use the recorded inputs with a verified local build on another host. Running
the raw-checkpoint validators requires the retained external data. The two
masked long inputs were subsequently submitted as Aurora job8842248; its
long-time results are documented separately from these short preflights.
