# PC-GH qualification log

## Claim policy

This is an append-oriented evidence log.  A build, a short run, or a finite residual
does not by itself qualify the solver.  `PASS` below means only that the named bounded
gate passed.  `OPEN` and `BLOCKED` are not silently replaced by weaker criteria.

Current evidence source commit: `55290efdf310a03dcce8d5cbc84a0ded595a8d13`.
Required baseline: `d3148a1b87c9b28008c92388055d6aebd56c381a`.

## Reproducibility envelope

Unless a row states otherwise, current numerical evidence used:

- input tree: this commit and the named file under `tst/inputs`;
- compiler: GCC `13.3.0` via `/usr/bin/c++`;
- build: CMake `Debug`, `-g`;
- backend: Kokkos Serial with Kokkos debug and bounds checks enabled;
- MPI: disabled;
- timestepper: RK2;
- spatial order: 2;
- KO coefficient: 0;
- damping `kappa`: 0.

Temporary run directories under `/tmp` are not durable artifacts.  The committed input,
generator, table, source, and exact numbers below are the durable provenance record.

## Gate ledger

| Gate | Status | Evidence / remaining condition |
|---|---|---|
| 1 symbolic identities | PASS | complete `analysis/pc_gh_symbolic/run_all.py` pass at current commit |
| 2 flat algebra | PASS | exact projection/map tests plus 10,000 seeded SPD inverse, projection, and ADM round-trip trials pass |
| 3 Minkowski pointwise | PASS | exact state, ADM round trip, RHS, and diagnostics at floating-point zero |
| 4 wave convergence | PASS | exact shifted harmonic wave at 32/64/128 cells; every exercised state/constraint sector is at least order 1.96 and curls remain exact |
| 5 robust Minkowski | PASS | seeded cell-scale perturbations at 32/64/128 cells through `t=2`; bounded amplification and no positive late-time fitted rate |
| 6 stationary trumpet pointwise | PASS | continuum target, decomposed three-precision conditioning audit, maximum locations, and second-order RMS residual ladder pass on their stated punctured domains |
| 7 stationary trumpet evolution | BLOCKED | Gauge A0 has a positive projected frozen mode; no derived nonperiodic diagnostic/characteristic outer BC also remains |
| 8 perturbed trumpet | OPEN | waits on gates 6 and 7 |
| 9 Bowen-York to trumpet | OPEN | existing ADM initial-data conversion has not been exercised as a qualified transition |
| 10 Gauge A1 | FAILED | bounded feedback linearization cannot affect the positive invariant tangential trace-free Q subspace; no production implementation authorized |
| 11 Gauge B | DEFERRED | no scaled driver derivation or combined symmetrizer yet |
| 12 boosted puncture | OPEN | waits on single-hole qualification |
| 13 spinning puncture | OPEN | waits on single-hole qualification |
| 14 binary | OPEN | waits on single-hole qualification |
| 15 AMR | OPEN | plumbing exists; reduction/curl injection has not been measured |

## 2026-09-01 — complete symbolic suite

Command:

```bash
/tmp/athenak-pc-gh-sympy-20260901/bin/python \
  analysis/pc_gh_symbolic/run_all.py
```

Result: all established regularization, projection, conformal-Ricci, corrected-primary,
gradient, four-dimensional oracle, FO-GH map, Gauge A0 generator, and source-policy
checks passed.  The suite deliberately prints expected failures for the three rejected
supplied regression targets.

The isolated Python requirements are `sympy==1.14.0` and `scipy==1.18.1`.

## 2026-09-01 — Gate 2 randomized flat algebra

Evidence source commit: `55290efdf310a03dcce8d5cbc84a0ded595a8d13`.
Script: `analysis/pc_gh_symbolic/verify_flat_algebra_randomized.py`.

Seed `20260901` generated 10,000 SPD conformal metrics over six overall metric
scales, arbitrary symmetric `Atilde` and three `Q` tensors, and `chi` over eight
decades.  The test independently evaluates the production cofactor inverse formulas,
the simultaneous unit-determinant/trace projections, and the PC-GH to ADM to PC-GH
metric/curvature round trip.

Maximum binary64 discrepancies were:

```text
cofactor inverse absolute       6.293703e-10
projected determinant           1.156852e-13
relative Atilde trace           1.108481e-15
relative Q trace                1.128805e-15
ADM metric round trip           5.826450e-13
ADM curvature round trip        1.674039e-11
chi round trip                  4.456524e-11
K round trip                    4.279577e-12
```

All scale-aware assertions passed.  Classification: `PASS` Gate 2 as a broad seeded
binary64 regression; the exact determinant/Q projection and constrained FO-GH map
remain separately proved by the symbolic suite.

## 2026-09-01 — exact harmonic Minkowski

Input: `tst/inputs/pc_gh_minkowski.athinput`.

Grid: `8 x 4 x 4`, one mesh block, periodic.  CFL `0.25`, `nlim=0`, harmonic gauge.

Result: `PASS: exact PC-GH Minkowski state, ADM round trip, RHS, and diagnostics`.
This is a pointwise construction/RHS gate, not an evolution-stability result.

## 2026-09-01 — initial nonlinear harmonic gauge-wave smoke test

Input: `tst/inputs/pc_gh_gauge_wave.athinput`.

Grid: `32 x 1 x 1`, two mesh blocks, periodic.  CFL `0.1`, `t=1`, 321 cycles,
amplitude `0.01`, harmonic gauge.

Result: `L1=1.593071e-04`, `Linf=2.593841e-03`.  This confirms a finite nonlinear
periodic evolution after the Gauge A0 source branch was added.  It does not satisfy the
required three-resolution, all-sector convergence gate.

## 2026-09-01 — Gate 4 shifted harmonic-wave convergence

Evidence source commit: `7bde88a31884eefa7e89089c63754f4e07c29841`.
Input: `tst/inputs/pc_gh_gauge_wave.athinput`.
Analyzer: `analysis/pc_gh_symbolic/analyze_gauge_wave_convergence.py`.

The exact solution is the flat metric under the harmonic null-coordinate map
`T=t+F(x-t)`, `X=x+F(x-t)`.  Unlike the earlier diagonal gauge wave, it has nonzero
shift and therefore exercises `beta` and `B` together with the lapse, conformal metric,
curvature, GH, `X`, `Y`, and `Q` sectors.  Common parameters were amplitude `0.01`,
periodic unit domain, two-to-eight 16-cell mesh blocks, RK2, CFL `0.1`, second-order
centered differences, harmonic gauge, `kappa=0`, and KO amplitude zero.  All runs ended
at `t=1`:

| N | cycles | aggregate L1 | aggregate Linf |
|---:|---:|---:|---:|
| 32 | 334 | 3.860982294920e-4 | 5.861907791622e-3 |
| 64 | 667 | 9.706938117387e-5 | 1.475440672817e-3 |
| 128 | 1333 | 2.429808464991e-5 | 3.692661258194e-4 |

The worst adjacent observed order in each required family was:

| family | minimum order | limiting diagnostic |
|---|---:|---|
| aggregate | 1.990224 | aggregate Linf |
| primary | 1.979089 | `Lambda^x` Linf |
| X | 1.977550 | `X_x` Linf |
| Q | 1.977492 | `Q_xxx` Linf |
| Y | 1.999437 | `Y_x` Linf |
| B | 1.990224 | `B_x^x` Linf |
| GH | 1.977918 | `C_perp` Linf |
| ADM | 1.962997 | Hamiltonian RMS |
| reduction | 1.995012 | Q-reduction Linf |

All curl diagnostics stayed below `1e-12`; on this compatible periodic one-dimensional
solution the discrete derivative operators commute, so exact preservation is the
appropriate result rather than a fitted order.  Algebraic-projection RMS corrections
decreased from `4.819972e-9` to `1.752859e-9` to `5.281400e-10`.

Classification: `PASS` Gate 4 for the second-order Serial configuration.  This does not
qualify higher spatial orders, MPI, GPU, AMR, or a physical gravitational-wave family.
The detailed table SHA-256 was
`6d3ea31a552793fa3d6dee76b916e1770fe4464752f28954dde7cb425a82aca3`.

## 2026-09-01 — Gate 5 robust Minkowski and KO defect discriminator

Evidence source commit: `7bde88a31884eefa7e89089c63754f4e07c29841`.
Input: `tst/inputs/pc_gh_robust_minkowski.athinput`.
Analyzer: `analysis/pc_gh_symbolic/analyze_robust_minkowski.py`.

The generator applies seed `20260901`, amplitude `1e-10`, independent cell-scale noise
to every continuum sector while constructing positive `A`/`chi`, an exactly
unit-determinant positive conformal metric, and metric-trace-free `Atilde` and `Q`.
Thus GH, physical, reduction, and curl constraints are all directly excited without a
floor or a derivative-field reset.

The first 32-cell attempt diagnosed an engineering defect: a positive user KO amplitude
was passed directly to the raw second-order `Diss` operator, whose Nyquist symbol is
positive.  The resulting anti-dissipation amplified `1e-10` noise to state RMS
`1.689157e-2` by `t=2`.  Production setup now applies the alternating sign and
`2^(-2p)` normalization.  `verify_ko_symbol.py` proves the normalized symbol is
`-sin(theta/2)^(2p)` for every supported stencil.  The discriminator was rerun without
changing its seed or user KO amplitude.

Common corrected-run parameters: periodic unit domain, 16-cell mesh blocks, RK4,
CFL `0.25`, second-order centered differences, harmonic gauge, `kappa=1`, user KO
amplitude `0.02`, and `t=2`:

| N | cycles | peak normalized diagnostic amplification | max late fitted rate | final state RMS | min SPD principal minor |
|---:|---:|---:|---:|---:|---:|
| 32 | 257 | 2.660800 (`Mhat`) | -0.096926/M (`redX`) | 5.267459e-11 | 0.9999999997691 |
| 64 | 513 | 2.674121 (`Mhat`) | -0.170527/M (`Cperp`) | 4.400557e-11 | 0.9999999998020 |
| 128 | 1025 | 1.576566 (`Mhat`) | -0.189207/M (`curlY`) | 3.729399e-11 | 0.9999999998567 |

Every GH, ADM, reduction, and curl family was sampled at 41 times.  The maximum
endpoint growth rate changes from `+0.178353/M` at 32 cells to `-0.018215/M` and
`-0.182190/M` under refinement; every fit over the second half of every run is
negative.  `A`, `chi`, and all conformal-metric SPD principal minors remain positive.

Classification: `PASS` the bounded Gate 5 resolution-growth search for this seeded
one-dimensional Serial configuration.  It is not a proof for arbitrary noise seeds or
three-dimensional/GPU/MPI configurations.  Final table SHA-256:
`5160e8e7f1526710aa25487b11075013248987ea7bb86f1ae1eed6e45b059760`.

## 2026-09-01 — stationary Gauge A0 continuum target

Generator: `analysis/pc_gh_symbolic/generate_gauge_a0_table.py`.
Table: `inputs/pc_gh/gauge_a0_m1.dat`, `M=1`, 4097 points,
`r/M in [1e-8,1e4]`.

The generator reproduced the committed table byte-for-byte and passed the residual and
inner-exponent checks recorded in `docs/pc_gh_regularization_audit.md`.

## 2026-09-01 — stationary Gauge A0 discrete residual ladder

Input: `tst/inputs/pc_gh_trumpet_a0.athinput`, with command-line overrides
`audit_r_min=2.0` and equal `nx1=nx2=nx3=N`.

Common setup: domain `[-8,8]^3`, `16^3` mesh blocks, periodic, `M=1`, center zero,
CFL `0.1`, `nlim=0`, Gauge A0, audit shell `2M <= r <= 4M`.

The columns are maxima then component-sample RMS for primary RHS, gradient RHS,
GH/physical constraints, and reduction/curl/algebraic constraints:

| N | max P | max G | max GH/P | max R/C/A | rms P | rms G | rms GH/P | rms R/C/A |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 6.805345e-3 | 2.980813e-3 | 1.417438e-2 | 6.884724e-3 | 6.031425e-4 | 2.466420e-4 | 1.504585e-3 | 1.240393e-3 |
| 48 | 3.780944e-3 | 1.718759e-3 | 8.046744e-3 | 3.573120e-3 | 2.864677e-4 | 1.157426e-4 | 7.155474e-4 | 5.797452e-4 |
| 64 | 1.928830e-3 | 1.085954e-3 | 3.915768e-3 | 2.220801e-3 | 1.609426e-4 | 6.491906e-5 | 4.014704e-4 | 3.251500e-4 |
| 80 | 1.357528e-3 | 7.419769e-4 | 2.853682e-3 | 1.505075e-3 | 1.037794e-4 | 4.184119e-5 | 2.608615e-4 | 2.092320e-4 |

Observed RMS orders for `48 -> 64` are `2.004, 2.010, 2.009, 2.010`; for
`64 -> 80` they are `1.966, 1.969, 1.932, 1.976`.  The max norms are less regular
because the maximizing cell changes near the hard inner audit shell.  Per-field serial
output identifies the maximum and its location; family maxima and RMS are MPI-global
when built with MPI.

Classification: second-order discrete pointwise evidence on a bounded shell.  It is not
an outer-boundary pass or a stationary evolution result.

## 2026-09-01 — stationary Gauge A0 source-cancellation audit

Script: `analysis/pc_gh_symbolic/audit_gauge_a0_cancellation.py`.

The script evaluated the production table representation on a radial/tangential tensor
basis at 73 radii from `1.1e-8 M` through `100 M`.  It logged 387 named temporaries,
additive RHS terms, sums, and term scales in binary64, long double, and 100-digit
arithmetic.

Result: no additive RHS term has fitted inner power below `-0.25`.  The maximum
100-digit total RHS on the bounded open table domain is `5.118e-5` in radial `Lambda`
at `r=2.0797956529e-8 M`.  The worst binary64 discrepancy is `2.827e-8` absolute and
`3.547e-7` relative to the corresponding additive-term scale.  The table was regenerated
byte-for-byte with SHA-256
`122d84a52b4f19ea5c7e4c13a4e0bc8a9d488265d5d9df306bdd360978928eb5`.

The raw angular derivative of the direction-dependent trumpet `Atilde` tensor scales as
`r^-1.000001`.  It is logged rather than hidden; all additive RHS terms containing it
remain bounded.  This closes the mandatory Gauge A0 temporary-cancellation gate on the
punctured table domain.  It does not cover Bowen-York data.

## 2026-09-01 — Gauge A0 frozen-operator hard stop

Input: `tst/inputs/pc_gh_frozen_operator.athinput`.
Extractor: `pc_gh_trumpet_a0` with `frozen_operator=true`.
Analyzer: `analysis/pc_gh_symbolic/analyze_frozen_operator.py`.

The extractor centrally perturbs all 55 fields in the production kernel.  A constant
perturbation gives the complete lower-order Jacobian; a sinusoidal perturbation gives
the actual centered-FD response.  The analyzer additionally linearizes the production
algebraic projection and restricts the operator to its rank-50 tangent space.

At `r=M`, `k=0`, and local spacing `dx/M=0.025`, the projected operator has:

```text
max Re(lambda) = 3.29527194199e-1 / M
min Re(lambda) = -4.41193267394e-1 / M
spectral radius = 5.63162099634e-1 / M
eigenvector condition number = 6.0582494e1
Euclidean logarithmic norm = 2.03212816094 / M
non-normality = 5.03609775e-1
```

The positive mode is not a perturbation-size artifact: raw `max Re(lambda)` agrees to
the shown digits for central-difference amplitudes `1e-6`, `1e-7`, and `1e-8`.  At
`k=0`, raw values converge from `0.3330987443/M` at `dx/M=0.1` to
`0.3337008071/M` at `0.05` and `0.3338520383/M` at `0.025`.

At `r=M`, `dx/M=0.025`, the projected rightmost values are `0.3365117983/M` for
radial `kM=1` and `0.3084139301/M` for tangential `kM=1`.  The projected `k=0`
rightmost mode remains positive at all sampled radii:

| r/M | max Re(lambda) raw at k=0 |
|---:|---:|
| 0.5 | 0.5717866 |
| 1.0 | 0.3338520 |
| 2.0 | 0.1808149 |
| 4.0 | 0.1080776 |

The radial ladder used `dx/r=0.025`; the table lists raw values because those matrices
predated background-state output needed for the rank-50 projection.  The separately
re-extracted `r=M` matrix confirms that projection changes the positive rate only
slightly.  KO cannot change a `k=0` mode because its Fourier symbol vanishes there.

Classification: `FAILED` frozen-operator clearance for Gauge A0.  The Euclidean
logarithmic norm is diagnostic only.  The explicit pulled-back FO-GH symmetrizer is
given in `docs/pc_gh_derivation.md`; for its induced norm, as for every induced norm,
the logarithmic norm obeys

```text
mu_E(L) >= max Re(lambda) >= lambda_Q = 0.27162910729 / M > 0.
```

Thus the formulation-energy logarithmic-norm *sign* gate also fails independently of
the matrix representation chosen for that symmetrizer.  No evolution, parameter
tuning, or KO escalation is authorized from this evidence.

## 2026-09-01 — bounded Gauge A1 discriminator

The bounded feedback was linearized before implementation.  Its complete Jacobian
update is recorded in `docs/pc_gh_derivation.md` and can be applied with `--mu-l` and
`--mu-s` in `analyze_frozen_operator.py`.

At `r=M`, `k=0`, and `dx/M=0.025`, a nonnegative grid
`mu_L,mu_S in {0,0.1,0.3,1,3,10,30}` never makes the projected rightmost rate
nonpositive.  Shift feedback lowers it from `0.329527/M` but reaches a positive floor
`0.271629/M`; lapse feedback does not remove that floor.

The floor is analytic, not just a finite scan.  A projected tangential trace-free `Q`
subspace is invariant under the lower-order operator and has

```text
lambda_Q = (5 B_tangential - 2 B_radial) / 3
         = 0.27162910729 / M   at r=M.
```

Gauge A1 acts only on `A/Y` and `beta/B`, so this eigenvalue is independent of both
feedback coefficients.  Classification: `FAILED` Gauge A1 frozen clearance.  The
feedback is not implemented in production.

## Current hard stop

Do not start a stationary evolution campaign.  Gauge A0 has a robust positive projected
frozen mode whose invariant subspace is identified above, and only periodic physical
boundaries exist.  A periodic finite box is not acceptable as a production single-hole
outer boundary.  The mandated baseline is `(gamma_1,gamma_2)=(-1,0)`, and the
qualification plan forbids adding `gamma_2` until that baseline qualifies.  Because it
did not qualify, reduction-constraint damping is not an authorized next step.
