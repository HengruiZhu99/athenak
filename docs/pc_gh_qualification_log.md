# PC-GH qualification log

## Claim policy

This is an append-oriented evidence log.  A build, a short run, or a finite residual
does not by itself qualify the solver.  `PASS` below means only that the named bounded
gate passed.  `OPEN` and `BLOCKED` are not silently replaced by weaker criteria.

Latest moving-puncture implementation evidence source commit: `f5b334f0`.
Latest SMR localization/horizon-adapter evidence source commit: `377a1edf`.
Latest qualification analysis/figure commit: `668ee7bf`.
Latest puncture-regular 55-field implementation and qualification: this commit,
described in the final 2026-09-02 section below.
Required baseline: `d3148a1b87c9b28008c92388055d6aebd56c381a`.

## Reproducibility envelope

Unless a row states otherwise, the legacy numerical evidence below used:

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
| 1 symbolic identities | PASS | complete suite, including the puncture-regular 55-field equivalence, gauges, characteristic, and asymptotic checks |
| 2 flat algebra | PASS | exact projection/map tests plus 10,000 seeded SPD inverse, projection, and ADM round-trip trials pass |
| 3 Minkowski pointwise | PASS | exact regular state, ADM round trip, RHS, and diagnostics in both Release Serial and two-rank MPI builds |
| 4 wave convergence | PASS | regular 55-field shifted harmonic wave at 32/64/128 cells; minimum exercised order 1.859 and curls remain exact |
| 5 robust Minkowski | PASS | regular 55-field seeded perturbations at 32/64/128 cells through `t=2`; peak amplification 4.48 and all late fitted rates are negative |
| 6 stationary trumpet pointwise | PASS | regular 55-field N=32/48/64/80 residual ladder and `1e-12`/`1e-14`/`1e-16` byte-identical floor audit pass |
| 7 stationary trumpet evolution | BLOCKED | Gauge A0 has a positive projected frozen mode; no derived nonperiodic diagnostic/characteristic outer BC also remains |
| 8 perturbed trumpet | OPEN | waits on Gate 7 |
| 9 Bowen-York to trumpet | FAILED (partial improvement) | both regular 55-field gauges now reach `6M` at M/16--M/24, but `rho` has a negative inner power and its inner maximum grows approximately as `N^1.12`; full-domain reductions/curls are nonconvergent |
| 10 Gauge A1 | FAILED | bounded feedback linearization cannot affect the positive invariant tangential trace-free Q subspace; no production implementation authorized |
| 11 Gauge B | DEFERRED | no scaled driver derivation or combined symmetrizer yet |
| 12 boosted puncture | BLOCKED | waits on the failed single-hole convergence gate; existing Z4c-only pgen is not an audited PC-GH initial-data path |
| 13 spinning puncture | BLOCKED | waits on the failed single-hole convergence gate and constraint-satisfying TwoPunctures data |
| 14 binary | BLOCKED | waits on the failed single-hole convergence gate and constraint-satisfying TwoPunctures data |
| 15 AMR | OPEN | Z4c transfer plumbing is reused and reduction changes are measured, but no dynamic-AMR qualification exists |

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

## 2026-09-01 — Bowen-York regularity and production-path audit

Evidence source commit: `f74a19ae4d425bccbbd1bff78db72bbe49502f42`.
Input: `tst/inputs/pc_gh_bowen_york.athinput`.
Analyzers: `analysis/pc_gh_symbolic/audit_bowen_york_cancellation.py` and
`analysis/pc_gh_symbolic/analyze_bowen_york_residuals.py`.

The three-precision analytic audit evaluated 217 named quantities over 81 radii for
time-symmetric, momentum, spin, and combined Bowen-York leading fields.  No stored
field, temporary, or additive RHS term had fitted inner power below `-0.25`.  The
maximum 100-digit `|H+Atilde^2|` identity residual was `9.875e-101`; worst normalized
RHS-sum discrepancies were `7.612e-14` in binary64 and `2.403e-17` in long double.
The momentum/spin cases omit the regular TwoPunctures correction and are therefore
regularity models, not constraint-satisfying data.

The required baseline contains the `z4c_two_puncture.cpp` adapter and CMake hook, but
does not track the external `twopuncturesc` headers/library or register them as a
submodule.  The original dirty checkout exposes an untracked symlink to a local
external installation; that user-owned dependency was inspected read-only and
intentionally excluded from this branch's build provenance.  Consequently no
constraint-satisfying nonzero-momentum/spin TwoPunctures build or runtime result is
claimed here.

The production pgen fills exact time-symmetric isotropic Schwarzschild ADM data on all
cells and ghosts, calls the actual `ADMToPcGh` and `PcGhToADM` paths, then evaluates the
actual RHS and constraints on `1M <= r <= 4M`.  Common setup was `[-8,8]^3`, `16^3`
mesh blocks, periodic, RK2, second-order centered differences, harmonic gauge,
`kappa=0`, KO amplitude zero, and `nlim=0`:

| N | rms state primary | rms state gradient | rms RHS primary | rms RHS gradient | rms GH/physical | rms reduction/curl/algebraic |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 7.416731e-17 | 9.065557e-4 | 2.281639e-3 | 2.791716e-17 | 8.676208e-3 | 2.137871e-16 |
| 48 | 8.953469e-17 | 3.755427e-4 | 1.072975e-3 | 6.937500e-17 | 4.013403e-3 | 3.305478e-16 |
| 64 | 9.991160e-17 | 2.209600e-4 | 6.443014e-4 | 1.094164e-16 | 2.446672e-3 | 4.099355e-16 |
| 80 | 1.116420e-16 | 1.412928e-4 | 4.190921e-4 | 1.549547e-16 | 1.592116e-3 | 5.373368e-16 |

The exact sectors remain below `5.4e-16`.  Least-squares orders over all four grids are
`2.018570`, `1.841383`, and `1.839332` for state gradients, primary RHS, and
GH/physical constraints; corresponding `64 -> 80` orders are `2.003853`, `1.927351`,
and `1.925509`.  The analyzer requires four levels to control hard-shell sampling,
monotone residual decrease, fitted order at least `1.8`, finest-pair order at least
`1.85`, and maximum-location tables at every resolution.  Residual-table SHA-256:
`7d42b2e295f20055827438a921a57fe390762d68f176281d301b4f1da4dc8984`.

Classification: `PASS` for bounded analytic Bowen-York source conditioning and the
exact time-symmetric ADM-to-PC-GH pointwise ladder in the stated Serial configuration.
This is not a TwoPunctures momentum/spin test, a finite-time evolution, a trumpet
transition, or a pass of Gate 9.

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
subspace is invariant under the complete pointwise-frozen lower-order operator and has

```text
lambda_Q = (5 B_tangential - 2 B_radial) / 3
         = 0.27162910729 / M   at r=M.
```

Gauge A1 acts only on `A/Y` and `beta/B`, so this eigenvalue is independent of both
feedback coefficients.  Classification: `FAILED` Gauge A1 frozen clearance.  The
feedback is not implemented in production.

This pointwise `Q` eigenvector is off the first-order reduction manifold at `k=0`, so it
is not itself a physical Fourier perturbation.  Direct reduction-constraint propagation
removes the `B_ell{}^ell=t` contribution and gives the smaller but still positive exact
rate

```text
lambda_R = 2 (B_tangential - B_radial) / 3
         = 0.11738365338 / M   at r=M.
```

`analysis/pc_gh_symbolic/verify_reduction_constraint_growth.py` proves both rates and
their difference exactly.  The bounded Gauge-A1 source does not enter the conformal
metric source and therefore cannot change `lambda_R`.  This diagnoses a local
reduction-constraint growth mechanism; it does not establish a global eigenmode or an
evolution growth rate.

## 2026-09-01 — independent frozen-operator re-audit handoff

A fresh local extraction independently reproduced the previously recorded rightmost
rates.  For `k=0`, the raw rates at `dx/M={0.1,0.05,0.025}` are
`{0.3330987443,0.3337008071,0.3338520383}/M`; the projected `dx/M=0.025` rate is
`0.3295271942/M`.  Repeating the finest extraction with relative perturbations
`1e-6`, `1e-7`, and `1e-8` agrees to the displayed digits.  For radial `kM=1`, the
raw rates at the same spacings are `{0.3218119199,0.3225780310,0.3227709150}/M`,
and the finest projected rate is `0.3365117983/M`.

The extractor now also linearizes the signed production constraint residuals, and
the analyzer reports GH, physical, algebraic, first-order reduction, and curl
responses of the rightmost projected mode.  At `k=0`, `dx/M=0.025`, that mode has
eigenvalue `(0.3295271942+0.1970486050 i)/M`.  One representative eigenvector,
normalized to maximum component one, has production GH and physical-constraint
response norms `1.6986` and `0.4581`, while its algebraic-constraint response is
`2.8e-16`.  Its reduction-constraint norms `(X,Q,Y,B)` are
`(0.1467,1.6037,0.1266,0.3492)`.  Thus this rightmost representative is tangent to
the algebraic conformal manifold but is not an admissible physical/GH/
reduction-constraint perturbation.

This re-audit was stopped at the user's request for remote independent review.  The
new decomposition is diagnostic infrastructure, not a completed root-cause audit,
and it does not supersede the hard stop below.

## 2026-09-01 — direct moving-puncture gauge decision tree

Evidence source commit: `f5b334f0`. The two new exact scripts prove the requested
primary and STANDARD Y/B gauge equations and construct the 50-by-50 algebraic-tangent
principal symbol. The direct `z4c_mp` characteristic polynomial is

```text
-v^30 (v-1)^2 (v+1)^2 (3v^2-4) (2 alpha chi-v^2)
    (alpha^2 chi-v^2)^6 / 3
```

It is diagonalizable at generic positive `alpha,chi`, but exact ranks find defects at
`alpha^2 chi=4/3` (multiplicity `7/6`), `alpha=2` (generically `7/5`), and
`alpha chi=2/3` (`2/1`). The unit-mass initial wormhole crosses the last surface at
`r/M=7.151725902759133`. The direct gauge therefore fails the complete-basis condition
for spectral/SAT promotion.

The prescribed follow-on `z4c_mp_hyperbolic` switch has longitudinal speed squared
`(4-S alpha^2 chi)/3`. With the implemented smoothstep from `S=0` at
`alpha chi=0.1` to `S=1` at `alpha chi=0.5`, exact ranks and inequalities establish a
complete basis only on the conditional domain `0<alpha<=1`, `0<alpha chi<=1`. The
evolution does not enforce those inequalities, so this is not a global
strong-hyperbolicity claim.

### Build and symbolic commands

The compiler was `/usr/bin/c++`, GCC `13.3.0`; both builds were Release, Kokkos Serial,
MPI off, double precision. Commands were:

```bash
cmake -S . -B /tmp/athenak-pcgh-z4cmp-release-20260901 \
  -D CMAKE_BUILD_TYPE=Release
cmake --build /tmp/athenak-pcgh-z4cmp-release-20260901 -j2

cmake -S . -B /tmp/athenak-z4c-mp-control-20260901 \
  -D PROBLEM=z4c_one_puncture -D CMAKE_BUILD_TYPE=Release
cmake --build /tmp/athenak-z4c-mp-control-20260901 -j2

/tmp/pc-gh-z4c-mp-sympy/bin/python analysis/pc_gh_symbolic/run_all.py
/tmp/pc-gh-z4c-mp-sympy/bin/python \
  analysis/pc_gh_symbolic/verify_z4c_mp_gauge.py
/tmp/pc-gh-z4c-mp-sympy/bin/python \
  analysis/pc_gh_symbolic/analyze_z4c_mp_principal.py
```

The complete symbolic suite passed, including all prior gates and the expected printed
counterexamples to rejected historical equations.

### Minkowski gates

Exact Minkowski passed for both gauges with:

```bash
PCGH=/tmp/athenak-pcgh-z4cmp-release-20260901/src/athena
$PCGH -i tst/inputs/pc_gh_minkowski.athinput \
  -d /tmp/pcgh-z4cmp-direct-gates-current-d1HHge pc_gh/gauge=z4c_mp
$PCGH -i tst/inputs/pc_gh_minkowski.athinput \
  -d /tmp/pcgh-z4cmp-hyp-gates-s81wTD pc_gh/gauge=z4c_mp_hyperbolic
```

The established robust-Minkowski input was then run at `N={32,64,128}` with:

```bash
RUN=/tmp/pcgh-z4cmp-direct-gates-current-d1HHge
for N in 32 64 128; do
  $PCGH -i tst/inputs/pc_gh_robust_minkowski.athinput -d "$RUN" \
    pc_gh/gauge=z4c_mp mesh/nx1="$N" meshblock/nx1=16 \
    job/basename="pc_gh_z4cmp_direct_robust_n${N}"
done
/tmp/pc-gh-z4c-mp-sympy/bin/python \
  analysis/pc_gh_symbolic/analyze_robust_minkowski.py \
  "$RUN/pc_gh_robust_minkowski-final.dat" \
  --history 32:"$RUN/pc_gh_z4cmp_direct_robust_n32.pcgh.hst" \
  --history 64:"$RUN/pc_gh_z4cmp_direct_robust_n64.pcgh.hst" \
  --history 128:"$RUN/pc_gh_z4cmp_direct_robust_n128.pcgh.hst"

RUN=/tmp/pcgh-z4cmp-hyp-gates-s81wTD
for N in 32 64 128; do
  $PCGH -i tst/inputs/pc_gh_robust_minkowski.athinput -d "$RUN" \
    pc_gh/gauge=z4c_mp_hyperbolic mesh/nx1="$N" meshblock/nx1=16 \
    job/basename="pc_gh_z4cmp_hyp_robust_n${N}"
done
/tmp/pc-gh-z4c-mp-sympy/bin/python \
  analysis/pc_gh_symbolic/analyze_robust_minkowski.py \
  "$RUN/pc_gh_robust_minkowski-final.dat" \
  --history 32:"$RUN/pc_gh_z4cmp_hyp_robust_n32.pcgh.hst" \
  --history 64:"$RUN/pc_gh_z4cmp_hyp_robust_n64.pcgh.hst" \
  --history 128:"$RUN/pc_gh_z4cmp_hyp_robust_n128.pcgh.hst"
```

The direct run directory was `/tmp/pcgh-z4cmp-direct-gates-current-d1HHge`; the
switched directory was `/tmp/pcgh-z4cmp-hyp-gates-s81wTD`. Both passed the existing
bounded gate:

| gauge | N | peak amplification | maximum late fitted rate `/M` | final state RMS |
|---|---:|---:|---:|---:|
| direct | 32 | 2.390899 | -0.096936 | 1.269404e-10 |
| direct | 64 | 2.546140 | -0.171004 | 1.060175e-10 |
| direct | 128 | 1.456351 | -0.189202 | 1.058574e-10 |
| switched | 32 | 2.390899 | -0.096947 | 7.118217e-11 |
| switched | 64 | 2.546140 | -0.171004 | 6.458413e-11 |
| switched | 128 | 1.456351 | -0.189202 | 6.110251e-11 |

This qualifies only exact and robust Minkowski for these Serial configurations.

### Matched one-puncture commands and results

The PC-GH and Z4c controls used the same unit-mass time-symmetric ADM wormhole, periodic
`[-8M,8M]^3` box, one uniform block, `N={16,24,32}`, second-order spatial operators,
RK4, CFL `0.1`, user KO amplitude `1.0`, and `eta=2/M`. The formulations have different
native constraint systems: PC-GH retained its already-established `kappa=1`, while the
explicit default Z4c control has `damp_kappa1=damp_kappa2=0`. Constraint definitions
also differ, so cross-formulation comparisons below are qualitative. The periodic,
coarse box is a robustness discriminator, not a physical outer-boundary or accurate
trumpet calculation.

Short `N=16`, `t=0.25M` commands were:

```bash
$PCGH -i tst/inputs/pc_gh_one_puncture.athinput \
  -d /tmp/pcgh-z4cmp-onepuncture-JMDUJc
/tmp/athenak-z4c-mp-control-20260901/src/athena \
  -i tst/inputs/z4c_one_puncture_control.athinput \
  -d /tmp/z4c-mp-onepuncture-control-y9UyfL
```

Both completed. PC-GH ended with positive `A=0.1695696`, `chi=0.1718281`, and minimum
SPD principal minor `0.9995141`; this is only a short-run gate.

The `20M` runs used the following exact command pattern, with all three spatial and
mesh-block dimensions set to each `N` and the run directories listed after it:

```bash
$PCGH -i tst/inputs/pc_gh_one_puncture.athinput -d "$RUN" \
  time/tlim=20 output1/dt=0.25 \
  mesh/nx1="$N" mesh/nx2="$N" mesh/nx3="$N" \
  meshblock/nx1="$N" meshblock/nx2="$N" meshblock/nx3="$N" \
  job/basename="$TAG" pc_gh/gauge="$GAUGE"

/tmp/athenak-z4c-mp-control-20260901/src/athena \
  -i tst/inputs/z4c_one_puncture_control.athinput -d "$RUN" \
  time/tlim=20 output1/dt=0.25 \
  mesh/nx1="$N" mesh/nx2="$N" mesh/nx3="$N" \
  meshblock/nx1="$N" meshblock/nx2="$N" meshblock/nx3="$N" \
  job/basename="$TAG"
```

Direct PC-GH directories were
`/tmp/pcgh-z4cmp-onepuncture-t20-JdqrU7`,
`/tmp/pcgh-z4cmp-onepuncture-n24-q9ocll`, and
`/tmp/pcgh-z4cmp-onepuncture-n32-E3mzus`. Switched PC-GH directories were
`/tmp/pcgh-z4cmp-hyp-onepuncture-n16-j9sfx8`,
`/tmp/pcgh-z4cmp-hyp-onepuncture-n24-AmlmHq`, and
`/tmp/pcgh-z4cmp-hyp-onepuncture-n32-VUUJjZ`. Z4c directories were
`/tmp/z4c-mp-onepuncture-n16-Z4taxt`,
`/tmp/z4c-mp-onepuncture-n24-1i0NMz`, and
`/tmp/z4c-mp-onepuncture-n32-4C2382`.

The exact `(GAUGE,N,RUN,TAG)` substitutions were:

| formulation | gauge | N | run-directory suffix | tag |
|---|---|---:|---|---|
| PC-GH | `z4c_mp` | 16 | `pcgh-z4cmp-onepuncture-t20-JdqrU7` | `pc_gh_one_puncture` |
| PC-GH | `z4c_mp` | 24 | `pcgh-z4cmp-onepuncture-n24-q9ocll` | `pc_gh_one_puncture_n24` |
| PC-GH | `z4c_mp` | 32 | `pcgh-z4cmp-onepuncture-n32-E3mzus` | `pc_gh_one_puncture_n32` |
| PC-GH | `z4c_mp_hyperbolic` | 16 | `pcgh-z4cmp-hyp-onepuncture-n16-j9sfx8` | `pc_gh_hyp_one_puncture_n16` |
| PC-GH | `z4c_mp_hyperbolic` | 24 | `pcgh-z4cmp-hyp-onepuncture-n24-AmlmHq` | `pc_gh_hyp_one_puncture_n24` |
| PC-GH | `z4c_mp_hyperbolic` | 32 | `pcgh-z4cmp-hyp-onepuncture-n32-VUUJjZ` | `pc_gh_hyp_one_puncture_n32` |
| Z4c | input default | 16 | `z4c-mp-onepuncture-n16-Z4taxt` | `z4c_one_puncture_n16` |
| Z4c | input default | 24 | `z4c-mp-onepuncture-n24-1i0NMz` | `z4c_one_puncture_n24` |
| Z4c | input default | 32 | `z4c-mp-onepuncture-n32-4C2382` | `z4c_one_puncture_n32` |

Every PC-GH run reached `20M` with finite constraint histories sampled at 81 output
times. Positivity and SPD were checked by the final diagnostic at the endpoint, not
recorded as time-series minima; the endpoint values were:

| gauge | N | max GH | max ADM | max reduction/curl | max algebraic | min A | min chi | min SPD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | 16 | 0.0842200 | 0.167283 | 0.0017914 | 3.75e-7 | 0.48957 | 0.43272 | 0.95666 |
| direct | 24 | 0.180386 | 0.366192 | 0.0035310 | 5.77e-7 | 0.35007 | 0.30873 | 0.90804 |
| direct | 32 | 0.347373 | 0.753324 | 0.0147963 | 3.29e-5 | 0.193998 | 0.171497 | 0.83826 |
| switched | 16 | 0.0843156 | 0.167636 | 0.0017241 | 1.47e-7 | 0.48859 | 0.42825 | 0.96820 |
| switched | 24 | 0.180642 | 0.367116 | 0.0035262 | 7.33e-7 | 0.34883 | 0.30529 | 0.91236 |
| switched | 32 | 0.346936 | 0.753620 | 0.0149539 | 2.49e-5 | 0.192884 | 0.169548 | 0.84512 |

At the final history sample, the maximum component RMS in each PC-GH family,
normalized by the diagnostic volume, was:

| gauge | N | GH | ADM | reduction | curl | algebraic projection |
|---|---:|---:|---:|---:|---:|---:|
| direct | 16 | 0.0133210 | 0.0261511 | 9.238e-4 | 1.797e-4 | 1.873e-7 |
| direct | 24 | 0.0214784 | 0.0422763 | 8.254e-4 | 4.278e-4 | 5.091e-8 |
| direct | 32 | 0.0334931 | 0.0669451 | 0.0015938 | 0.0018847 | 2.367e-6 |
| switched | 16 | 0.0133403 | 0.0261949 | 9.419e-4 | 1.796e-4 | 5.351e-8 |
| switched | 24 | 0.0215190 | 0.0423683 | 8.403e-4 | 4.299e-4 | 6.257e-8 |
| switched | 32 | 0.0335164 | 0.0670557 | 0.0016064 | 0.0019029 | 1.820e-6 |

The matched Z4c final volume-normalized RMS diagnostics were:

| N | aggregate C | H | M | Z | Theta |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.188204 | 0.113941 | 0.079275 | 0.046772 | 0.086041 |
| 24 | 0.220833 | 0.159700 | 0.074440 | 0.053596 | 0.078942 |
| 32 | 0.192036 | 0.138884 | 0.063806 | 0.048699 | 0.063496 |

The direct and switched PC-GH endpoint GH/ADM norms increase monotonically with
resolution, and the switch changes them only at the sub-percent level. Z4c's aggregate
constraint is nonmonotone and improves from `N=24` to `N=32`, but its diagnostics are
not the same PC-GH quantities. No convergence order or continuum growth rate is
claimed from these three coarse periodic grids.

Classification: direct `z4c_mp` is a useful finite-difference control but has a proven
defective shell. The switched variant repairs the reachable principal-symbol
coincidences only conditionally, yet does not repair the observed resolution trend.
Gate 9 therefore `FAILED`; this result does not establish whether the trend originates
near the under-resolved puncture, at the periodic boundary, or in the continuum
formulation. The sequential perturbed, boosted, spinning, and binary gates remain
blocked. The existing `z4c_boosted_puncture` path is Z4c-specific and its initialization
policy has not been audited for PC-GH; constraint-satisfying spin/binary data also
depends on an external `twopuncturesc` tree absent from this branch.

## Current hard stop

Gauge A0 remains stopped: do not start its stationary evolution campaign. It has a
robust positive projected frozen mode whose invariant subspace is identified above,
and only periodic physical boundaries exist.

The new moving-puncture path is also stopped before perturbed/boosted/spinning/binary
promotion. The apparent `20M` completions above were invalidated by the strict
non-finite final diagnostic: `fmax` had hidden NaNs while the CFL calculation advanced
with a spurious large timestep. The superseding SMR study below is the qualification
evidence. A periodic finite box is not an acceptable production single-hole outer
boundary. Do not tune KO, add unauthorized constraint damping, weaken norms, or treat
survival as qualification.

The single recommended next action is to isolate the puncture-interior semidiscrete
failure without changing the evolution equations, then decide explicitly whether an
interior regularization or turduckening policy is in scope.

## 2026-09-02 — M/16--M/24 SMR localization qualification

This study supersedes the low-resolution uniform-box and M/32 one-puncture evidence
above. No old run or M/32 result is included in the figures or classification here.
The symbolic, wave, Minkowski, and robust-Minkowski gates retained their inexpensive
resolutions. The sibling worktree branch starts from
`090f6e238139de44f299d3b49621ad0e16b2cc8f`.

### Mesh, transfer, and initial-data controls

All black-hole cases used the periodic `[-8M,8M]^3` box and the same four physical
static-refinement levels (232 MeshBlocks including the root and refined hierarchy).
The root/MeshBlock pairs were `16^3/8^3`, `20^3/10^3`, and `24^3/12^3`; the exact-z=0
Cartesian sampling grids were `64^2`, `80^2`, and `96^2`, and the even horizon dump
grids were 34, 42, and 50. A zero-step full-volume output was read back before the
evolutions:

| target | MeshBlocks in output | measured finest spacing |
|---|---:|---:|
| M/16 | 232 | 0.0625 M |
| M/20 | 232 | 0.0500 M |
| M/24 | 232 | 0.0416666666667 M |

`analysis/pc_gh_localization/measure_finest_spacing.py` performs this check from the
stored MeshBlock geometry. The PC-GH run task list now follows the Z4c update,
restriction, communication, boundary, prolongation, algebraic-projection, ADM, and
timestep ordering and calls the same `RestrictCC(..., true)` and
`ProlongateCC(..., true)` implementations.

PC-GH and Z4c were initialized independently from the same unit-mass, time-symmetric
wormhole data. `analysis/pc_gh_localization/compare_initial_data.py` compared their
t=0 Cartesian ADM outputs: all six metric components and `psi4` agree to a maximum
normalized absolute difference `1.47344e-8`, and all six extrinsic-curvature components
agree exactly (zero). This is float-output roundoff, not a distinct initial-data set.

### Commands and ordering

The matched Z4c controls were completed first, then all direct `z4c_mp` cases, then all
`z4c_mp_hyperbolic` cases. MPI rank counts were 8 or 10, never more than 12. The command
pattern was:

```bash
mpirun -np "$RANKS" build-z4c-mpi-release/src/athena \
  -i tst/inputs/z4c_one_puncture_control_smr.athinput \
  mesh/nx1="$N" mesh/nx2="$N" mesh/nx3="$N" \
  meshblock/nx1="$NB" meshblock/nx2="$NB" meshblock/nx3="$NB" \
  z4c/horizon_0_Nx="$NAH" \
  output2/numpoints_x="$NS" output2/numpoints_y="$NS" \
  output3/numpoints_x="$NS" output3/numpoints_y="$NS"

mpirun -np 8 build-mpi-release/src/athena \
  -i tst/inputs/pc_gh_one_puncture_smr.athinput \
  mesh/nx1="$N" mesh/nx2="$N" mesh/nx3="$N" \
  meshblock/nx1="$NB" meshblock/nx2="$NB" meshblock/nx3="$NB" \
  pc_gh/gauge="$GAUGE" pc_gh/horizon_0_Nx="$NAH" \
  output1/dt=0.005 output2/dt=0.005 output3/dt=0.005 \
  output2/numpoints_x="$NS" output2/numpoints_y="$NS" \
  output3/numpoints_x="$NS" output3/numpoints_y="$NS"
```

Here `(N,NB,NS,NAH)` was `(16,8,64,34)`, `(20,10,80,42)`, or
`(24,12,96,50)`. The input time limit remained `6M`.

### Matched Z4c result

All Z4c controls reached `t=6M`: 961, 1201, and 1441 cycles at M/16, M/20, and M/24.
Their native chi-excised, physical-volume RMS endpoints were:

| finest spacing | C | H | M | Z | Theta |
|---|---:|---:|---:|---:|---:|
| M/16 | 2.73163e-2 | 1.24786e-2 | 4.89648e-3 | 3.56899e-3 | 2.27055e-2 |
| M/20 | 2.66568e-2 | 1.23595e-2 | 4.95954e-3 | 2.11319e-3 | 2.27017e-2 |
| M/24 | 2.67319e-2 | 1.27402e-2 | 5.23011e-3 | 1.45254e-3 | 2.27264e-2 |

Z converges at fitted order 2.22, while the aggregate, H, M, and Theta measures are
approximately flat/nonmonotone. The exact-z=0 chi slices are smooth; there is no
per-block normalization. PC-GH's raw Cartesian interpolation has only a tiny
puncture-crossing overshoot at t=0 (minimum between `-3.7e-6` and `-5.1e-6` across the
suite); native cell-centered chi remains positive. Constraint slices show the expected
refinement-interface and periodic-boundary truncation structures.

The periodic mismatch is present at t=0. Using the conservative fastest coordinate
speed `sqrt(2)`, a boundary signal can reach `r=2M` at approximately `4.24M` and the
initial coordinate horizon radius `0.5M` at approximately `5.30M`, both before `6M`.
Late Z4c behavior is therefore a control, not clean outer-boundary physics.

### PC-GH result and localization

Both PC-GH gauges failed the strict finite-state/finite-constraint diagnostic. The
listed first-bad time is the first history sample containing any non-finite diagnostic;
all later synthetic times caused by the corrupted CFL estimate are excluded.

| gauge | finest spacing | last fully finite sample | first non-finite sample |
|---|---|---:|---:|
| `z4c_mp` | M/16 | 0.0216506 M | 0.0270633 M |
| `z4c_mp` | M/20 | 0.0129904 M | 0.0173205 M |
| `z4c_mp` | M/24 | 0.0108253 M | 0.0180422 M |
| `z4c_mp_hyperbolic` | M/16 | 0.0216506 M | 0.0270633 M |
| `z4c_mp_hyperbolic` | M/20 | 0.0129904 M | 0.0173205 M |
| `z4c_mp_hyperbolic` | M/24 | 0.0108253 M | 0.0180422 M |

At the common finite target `t=0.01M`, the two gauges agree to the displayed precision.
Coordinate-volume RMS values for the chi mask and `r>2M` are:

| spacing | GH chi / r>2 | ADM chi / r>2 | reduction chi / r>2 | curl chi / r>2 |
|---|---:|---:|---:|---:|
| M/16 | 2.1185e-4 / 2.1271e-4 | 2.3782e-2 / 2.3876e-2 | 4.0720e-3 / 4.0886e-3 | 5.8432e-4 / 5.7106e-4 |
| M/20 | 2.3559e-4 / 2.3654e-4 | 2.6436e-2 / 2.6543e-2 | 3.6304e-3 / 3.6451e-3 | 5.1317e-4 / 5.0622e-4 |
| M/24 | 2.5720e-4 / 2.5824e-4 | 2.8848e-2 / 2.8965e-2 | 3.3036e-3 / 3.3170e-3 | 4.6335e-4 / 4.5947e-4 |

The full-domain coordinate RMS differs little from the chi/radial values before the
failure (the largest effect is a 2--12% ADM reduction at each run's last finite sample).
Thus the old large physical-volume norm was puncture-weight sensitive, but the clean
coordinate-volume evidence does not show that all error is confined inside the mask.
The direct Cartesian series stays smooth in chi through its last finite output; the
first corrupted output begins near the puncture and the subsequent state becomes
non-finite globally. A uniform fine-grid run and a CFL-0.02 run failed at the same
physical time scale, so neither SMR transfer nor the CFL number explains the failure.

Classification:

```text
SEMIDISCRETE / PRINCIPAL / REDUCTION ISSUE
```

This is a puncture-triggered semidiscrete concern, not an `EXTERIOR PC-GH EVOLUTION
ROBUST` result. It occurs orders of magnitude before any causal outer-boundary return.
The hyperbolic switch provides no measurable improvement before failure. No evolution
equation, KO coefficient, damping, eta, projection frequency, or switch parameter was
changed.

### Horizon status and figures

The generic horizon adapter successfully wrote the same reconstructed-ADM
AHFinderDirect input packages for Z4c and PC-GH. This host has no Cactus/Einstein
Toolkit/AHFinderDirect executable, so the dumps could not be converted into area,
areal-radius, irreducible-mass, centroid, shape, residual, or iteration histories.
Moreover, PC-GH fails before the first scheduled `0.5M` follow-up dump. Horizon-property
drift and a true dynamic outside-AH mask therefore remain unmeasured; the online `ah`
history label is only the documented conservative spherical `r>0.5M+buffer` proxy.

Committed figures and machine-readable summaries are under
`docs/figures/pc_gh_localization_20260902/`. They include Z4c `t=6M` evolution,
three-resolution convergence and Cartesian slices, plus PC-GH pre-failure convergence,
localization, finiteness-window, and Cartesian-slice plots. No obsolete low-resolution
or M/32 data appear in them.

## 2026-09-02 — puncture-regular 55-field replacement and requalification

This section supersedes the PC-GH failure result immediately above. The source started
from `1c67ad8f` on branch `codex/pc-gh-localization-horizon-20260902`; all changes and
runs were made in the dedicated
`/Users/hz0693/research/athenak-pcgh-localization-20260902` worktree. Unrelated and
pre-existing untracked artifacts were preserved.

### Formulation and implementation changes

The storage ABI remains 55 components, but the old
`{chi,gtilde,K,Atilde,Lambda,pi,A,beta,X,Q,Y,B}` state is completely replaced by

```text
{w,gtilde,K,Atilde,Z,Cperp,rho,beta,p,Q,L,B}.
```

The production map and inverse identities are those in `docs/pc_gh_derivation.md`:
`chi=w^2`, `alpha=rho*w`, `A=rho^2*w^2`, `X=2*w*p`, `Y=alpha*L`,
`pi=Cperp-K`, and `Lambda=GammaTilde(Q)-Z`. No `partial_i rho` field is evolved.
The direct `w`, `rho`, `p`, and `L` equations; denominator-free `H`, `E`, `m`,
`alpha*M`, `S`, and `T`; direct `Cperp` and `Z`; and both shift variants are
implemented as written in the reviewed goal. The hyperbolic `B_i^j` equation includes
the full standard-order derivative, including `S'(z) partial_i(z) M^j` with
`z=rho*w^3` and `partial_i z=w^2(L_i/2+2*rho*p_i)`.

The complete change inventory is:

- `src/pc_gh/pc_gh.{hpp,cpp}`: new ABI, names, gauge composites, configuration,
  strict state/RHS/constraint/speed/SPD diagnostics, and complete boundedness output;
- `pc_gh_calcrhs.cpp`, `pc_gh_constraints.cpp`, `pc_gh_newdt.cpp`,
  `pc_gh_projection.cpp`, `pc_gh_tasks.cpp`, and `pc_gh_update.cpp`: denominator-free
  equations and constraints, regular-variable CFL, Z4c-aligned task graph, unchanged
  evolved `Z` under projection, and reduction-change monitors around transfer and
  projection;
- `pc_gh_adm.cpp`: guarded initialization-only `rho=alpha/w`, no full-volume ADM
  reconstruction in the timestep dependency chain, and masked diagnostic output;
- `src/utils/horizon_dump.hpp` and `src/z4c/horizon_dump.cpp`: interpolate regular
  conformal PC-GH fields first, reconstruct physical fields only outside the declared
  mask, and fill the masked cube with an output-only flat extension;
- `src/outputs/{outputs.hpp,history.cpp,basetype_output.cpp}`: new state/constraint
  labels and diagnostic ranges;
- PC-GH problem generators and `tst/inputs/pc_gh_*.athinput`: new exact state,
  zero-step 232-block/all-axis finest-spacing enforcement, and updated audit outputs;
- `analysis/pc_gh_symbolic`: independent 55-field proof, updated source/KO/principal
  checks and analyzers; `analysis/pc_gh_localization`: semantic history parsing,
  changed initial-data comparison, boundedness/power/localization plots; and the three
  PC-GH design/audit documents.

The task graph is now exactly update, restriction, communication, physical boundary,
prolongation, algebraic projection, diagnostic/output conversion, timestep. It calls
the same Z4c cell-centered restriction/prolongation implementations. Across the six
full runs the measured maximum changes were: restriction exactly zero in all four
reduction families; prolongation `dRw <= 2.04e-7`, `dRQ <= 6.28e-6`,
`dRalpha <= 2.36e-6`, and `dRB <= 1.23e-6`; projection changed only Q, with
`dRQ <= 6.54e-6`. This is instrumentation evidence, not a dynamic-AMR qualification.

### Symbolic, division, guard, and inexpensive numerical gates

The exact command

```bash
/tmp/athenak-pcgh-regular-sympy/bin/python \
  analysis/pc_gh_symbolic/run_all.py
```

passed in 60.63 seconds. The isolated Python 3.9 environment used SymPy 1.14.0,
NumPy 2.0.2, Matplotlib 3.9.4, and SciPy 1.13.1; SciPy 1.18.1 does not support the
host Python version. `verify_puncture_regular_55.py` proves the transformed
regularization identities, `Cperp`/`Z` equations, both shifts and differentiated B
equations, equivalence to the old positive-field equations on the algebraic/reduction
manifold, tangent-map similarity, exact characteristic polynomial, eigenspace ranks,
persistence of the direct-gauge defective surfaces, absence of a new finite positive
`rho` degeneracy, and wormhole/stationary-trumpet powers. The complete suite, including
all older independent tensor/oracle checks, ends with
`PASS: all currently established PC-GH symbolic gates`.

The requested raw `rg -n '/|sqrt\(|pow\('` audit covers the five named PC-GH kernels,
all PC-GH problem generators, outputs, and horizon paths and is stored at
`qualification-runs-20260902/regular55-final/verification/source-division-rg.log`
(640 matches, including comments and path/index slashes). Every quotient is classified
in `docs/pc_gh_regularization_audit.md`. `verify_source_policy.py` independently
passes over all nine production PC-GH files and rejects any preferred-source quotient
by `w`, `rho`, `alpha`, `chi`, or `A`.

The only puncture-field division is the defining initialization conversion
`rho=alpha/guarded_w`. The resolution-independent default floor is `1e-14`; input
values are checked before the guard, the unguarded global minimum and activated-cell
count are printed, and a negative/nonfinite input fails. No black-hole cell activated
the guard. Stationary-A0 N=32 runs at `1e-12`, `1e-14`, and `1e-16` produced
byte-identical residual files (SHA-256
`2523d48c15bbd61c0888eabd458131b0216549ab07c7b72c2e3e68c5c6d60d9e`) and maxima
files (`b11d72b5edb3db254b25412882876fb78a053e27d7440f32b3cfcc0839a0327d`).

Final Release builds used Apple clang 21.0.0; the MPI build used Open MPI 5.0.10.
The incremental Serial and MPI builds both passed in 3.6 seconds combined. The final
Serial and two-MeshBlock/two-rank MPI Minkowski tests both printed
`PASS: exact PC-GH Minkowski state, ADM round trip, RHS, and diagnostics`.

The inexpensive pre-black-hole results were:

- shifted gauge wave N=32/64/128: aggregate L1 errors
  `3.002128e-4`, `7.552022e-5`, `1.890239e-5`; minimum exercised sector order
  1.859491 (`Cperp` Linf), aggregate order at least 1.991, exact curls;
- robust Minkowski N=32/64/128 through `t=2`: state RMS
  `5.999411e-11`, `5.138255e-11`, `4.525745e-11`, peak amplification at most
  4.482354, every late-time fitted rate negative, SPD minimum above
  `0.999999999769`;
- stationary A0 N=32/48/64/80: maximum primary RHS decreases
  `1.45674e-2 -> 3.86234e-3`; the corresponding RMS decreases
  `1.37955e-3 -> 2.39809e-4`, while gradient, GH/ADM, and reduction/curl families
  form the same decreasing ladder. These are residual/asymptotic checks, not an
  authorization to run the known-defective stationary Gauge A0 evolution.

Durable logs are under
`qualification-runs-20260902/verification-regular55/` and
`qualification-runs-20260902/regular55-final/verification/`.

### Black-hole command, geometry, and ordering

The matched Z4c controls were not rerun. Direct PC-GH was completed first, followed by
hyperbolic PC-GH. Every black-hole command used the MPI-linked
`build-mpi-release/src/athena` with 12 ranks. The preflight/short/full pattern was:

```bash
mpirun -np 12 build-mpi-release/src/athena \
  -i tst/inputs/pc_gh_one_puncture_smr.athinput \
  mesh/nx1="$N" mesh/nx2="$N" mesh/nx3="$N" \
  meshblock/nx1="$NB" meshblock/nx2="$NB" meshblock/nx3="$NB" \
  pc_gh/gauge="$GAUGE" pc_gh/horizon_0_Nx="$NAH" \
  output2/numpoints_x="$NS" output2/numpoints_y="$NS" \
  output3/numpoints_x="$NS" output3/numpoints_y="$NS" \
  problem/expected_finest_spacing="$DX" time/tlim="$TLIM"
```

Preflight used `nlim=0,tlim=0`; its two-line geometry file was read and checked with
awk immediately before both the `tlim=0.1` short gate and the `tlim=6` full run.
The exact tuples were `(N,NB,NS,NAH)=(16,8,64,34),(20,10,80,42),(24,12,96,50)`.
The M24 command used the full-precision literal `0.0416666666666667`; the deliberately
tight pgen correctly rejects the shorter decimal when it differs by more than its
roundoff tolerance.

| target | blocks | recorded finest dx1=dx2=dx3 | min unguarded initial w | guarded cells |
|---|---:|---:|---:|---:|
| M/16 | 232 | `6.25000000000000000e-02` | 9.54121e-3 | 0 |
| M/20 | 232 | `5.00000000000000028e-02` | 6.35214e-3 | 0 |
| M/24 | 232 | `4.16666666666666644e-02` | 4.53077e-3 | 0 |

The PC-GH and reused Z4c `t=0` Cartesian data agree exactly for `w` versus the Z4c
initial lapse and for every conformal metric, curvature, lapse, and shift component.
The nonzero `interpolate(w)^2-interpolate(chi)` maxima are `5.61038e-5`,
`2.73394e-5`, and `1.48271e-5`, decrease with refinement, and are below the declared
`2e-4` interpolation-consistency tolerance. The JSON comparisons are stored in each
direct full-run directory.

### Short gates and full-run endpoints

All six strict short gates passed through `0.1M`; direct CPU times were 6.82, 16.26,
and 33.13 seconds, while hyperbolic times were 4.34, 10.61, and 24.46 seconds for
M/16, M/20, and M/24. The gauges agree to the shown short-run precision:

| gauge | spacing | max GH | max ADM | max reduction/curl | min SPD |
|---|---|---:|---:|---:|---:|
| direct | M/16 | 7.2886e-3 | 2.2020e-1 | 1.0618e-2 | 0.999750 |
| direct | M/20 | 9.0361e-3 | 1.6224e-1 | 8.0904e-3 | 0.999750 |
| direct | M/24 | 1.0763e-2 | 1.3156e-1 | 6.6653e-3 | 0.999750 |
| hyperbolic | M/16 | 7.2886e-3 | 2.2020e-1 | 1.0618e-2 | 0.999750 |
| hyperbolic | M/20 | 9.0361e-3 | 1.6224e-1 | 8.0904e-3 | 0.999750 |
| hyperbolic | M/24 | 1.0763e-2 | 1.3156e-1 | 6.6653e-3 | 0.999750 |

Every full run reached exactly `t=6M` without a nonfinite state, RHS, constraint,
characteristic speed, determinant, or eigenvalue, and without loss of SPD:

| gauge | spacing | cycles | CPU s | max GH | max ADM | max red/curl | max alg | min alpha^2 | min chi | min SPD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | M/16 | 1171 | 266.52 | 1.25108 | 2.05803 | 1.16594 | 9.94e-5 | 0.115868 | 1.74439e-3 | 0.867341 |
| direct | M/20 | 1466 | 641.22 | 1.24677 | 2.05812 | 1.60281 | 6.95e-5 | 0.119579 | 1.10995e-3 | 0.866192 |
| direct | M/24 | 1761 | 1481.84 | 1.22479 | 2.01923 | 1.96331 | 1.46e-7 | 0.119900 | 7.57387e-4 | 0.865247 |
| hyperbolic | M/16 | 1171 | 266.03 | 1.25160 | 2.06008 | 1.17097 | 1.21e-4 | 0.116157 | 1.74651e-3 | 0.870748 |
| hyperbolic | M/20 | 1466 | 640.91 | 1.24770 | 2.06070 | 1.61037 | 8.88e-5 | 0.119870 | 1.11165e-3 | 0.870295 |
| hyperbolic | M/24 | 1761 | 1509.47 | 1.22616 | 2.02218 | 1.97394 | 1.73e-5 | 0.120195 | 7.58728e-4 | 0.869958 |

Thus the new last-finite sample is `6M` and no first-bad sample exists in any case,
versus the old first-bad times `0.0270633M`, `0.0173205M`, and `0.0180422M`.
Survival improved by at least factors 220, 346, and 333 at M/16, M/20, and M/24.

### Constraint localization and boundedness

The endpoint coordinate-volume RMS values below list `(M/16,M/20,M/24)`; `chi` means
the declared regular conformal mask and `all` is the full domain:

| gauge/family | chi-mask RMS | full-domain RMS |
|---|---|---|
| direct GH | `(5.1569,5.1383,5.1326)e-3` | `(6.0976,5.6710,5.4633)e-3` |
| direct ADM | `(1.00936,1.00765,1.00694)e-2` | `(1.05766,1.03679,1.02603)e-2` |
| direct reduction | `(6.7449,4.7125,3.5474)e-4` | `(1.4226,1.4902,1.5011)e-3` |
| direct curl | `(2.0201,1.3612,0.9892)e-3` | `(5.5278,5.6473,5.6119)e-3` |
| hyperbolic GH | `(5.1575,5.1387,5.1330)e-3` | `(6.0966,5.6709,5.4636)e-3` |
| hyperbolic ADM | `(1.00951,1.00778,1.00699)e-2` | `(1.05783,1.03695,1.02612)e-2` |
| hyperbolic reduction | `(6.7056,4.6661,3.5244)e-4` | `(1.4228,1.4909,1.5028)e-3` |
| hyperbolic curl | `(2.0118,1.3432,0.9520)e-3` | `(5.5401,5.6596,5.6222)e-3` |

Masked reductions converge at approximately order 1.54--1.63 on the finest pair and
masked curls at 1.75--1.89. GH/ADM are nearly flat rather than cleanly convergent;
full-domain reductions grow slightly and full-domain curls are nonmonotone. The
hyperbolic switch changes these values only at the sub-percent level.

All requested global time-series bounds are stored in the six
`*.pcgh-boundedness.dat` files and summarized in
`docs/figures/pc_gh_localization_20260902/regular55/qualification_summary.json`.
The principal resolution-growing maxima are:

| gauge | spacing | max rho | max L | max B | min w | min rho | min metric eigenvalue | max primary RHS | max gradient RHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | M/16 | 2.7742 | 2.1502 | 1.1019 | 9.6901e-3 | 0.999907 | 0.931068 | 0.9551 | 0.6390 |
| direct | M/20 | 3.5892 | 2.4688 | 1.2073 | 6.4401e-3 | 0.999969 | 0.930422 | 1.4036 | 0.8535 |
| direct | M/24 | 4.3567 | 2.7726 | 1.2787 | 4.5869e-3 | 0.999974 | 0.929858 | 1.8881 | 1.0888 |
| hyperbolic | M/16 | 2.7795 | 2.1525 | 1.1027 | 9.6901e-3 | 0.999907 | 0.932918 | 0.9582 | 0.6409 |
| hyperbolic | M/20 | 3.5952 | 2.4713 | 1.2081 | 6.4401e-3 | 0.999964 | 0.932758 | 1.4075 | 0.8556 |
| hyperbolic | M/24 | 4.3636 | 2.7753 | 1.2798 | 4.5869e-3 | 0.999970 | 0.932621 | 1.8928 | 1.0913 |

The table below records every fitted `t=6M`, `r<=0.5M` puncture power and the fitted
resolution exponent `q_N` of the inner maximum (`max_inner ~ N^q_N`). A negative
puncture power violates the required no-divergence condition; a positive `q_N` marks
a growing sampled inner maximum.

| field | direct powers N16,N20,N24 | direct q_N | hyperbolic powers N16,N20,N24 | hyperbolic q_N |
|---|---|---:|---|---:|
| w | `(0.949,0.916,0.945)` | 0.026 | `(0.949,0.915,0.944)` | 0.026 |
| rho | `(-0.320,-0.373,-0.433)` | **1.122** | `(-0.320,-0.373,-0.433)` | **1.121** |
| p | `(0.297,0.267,0.266)` | 0.135 | `(0.296,0.266,0.265)` | 0.135 |
| L | `(-0.076,-0.144,-0.187)` | 0.375 | `(-0.077,-0.144,-0.187)` | 0.374 |
| Cperp | `(-1.206,-1.262,-1.260)` | -0.648 | `(-1.207,-1.263,-1.259)` | -0.646 |
| Z | `(-1.188,-1.181,-1.349)` | -0.054 | `(-1.188,-1.181,-1.350)` | -0.053 |
| K | `(-0.186,-0.121,-0.066)` | -0.322 | `(-0.188,-0.123,-0.068)` | -0.323 |
| Atilde | `(1.201,1.104,1.093)` | 0.156 | `(1.200,1.103,1.092)` | 0.157 |
| beta | `(0.877,0.819,0.830)` | 0.032 | `(0.876,0.819,0.830)` | 0.032 |
| Q | `(-0.426,-0.425,-0.442)` | -0.140 | `(-0.428,-0.426,-0.443)` | -0.138 |
| B | `(-0.256,-0.229,-0.227)` | 0.367 | `(-0.256,-0.229,-0.228)` | 0.367 |

Most decisively, `rho` does not approach the expected nondivergent/vanishing trumpet
behavior: its inner maximum changes from about 2.80 to 3.63 to 4.41 and grows as
approximately `N^1.12`. `L` and `B` also grow under refinement, and multiple evolved
fields have negative local powers. Finite values at these three resolutions are not
evidence of a bounded continuum puncture limit.

### Reused Z4c, horizons, figures, and classification

The reused Z4c histories are the existing 34,901-byte files in
`qualification-runs-20260902/z4c-r{16,20,24}` with SHA-256 values
`d1f48f63883f9083...`, `417c7b4f3dbc22da...`, and `376ed9e74b990dd4...`.
The corresponding `co_0` tracker hashes are `d5fd96081835bcf...`,
`54d6cfc1019d742...`, and `c39e2cdba0c7ac8...`. Full paths, sizes, timestamps, and
hashes are in
`qualification-runs-20260902/regular55-final/verification/z4c-reuse-provenance.log`.
Their endpoint native constraints are unchanged: only Z shows clear convergence
(order 2.22); C/H/M/Theta are flat or nonmonotone. No Z4c evolution was rerun.

Both PC-GH gauges wrote horizon-adapter packages at all scheduled times using regular
conformal interpolation and the even grids 34/42/50. This host still has no actual
AHFinderDirect property output, so no area/mass/shape/residual/iteration plot is
fabricated. The online `ah` label remains a documented spherical proxy. The periodic
boundary can conservatively return to `r=2M` by `4.24M` and the initial horizon radius
by `5.30M`; late-time values are robustness controls, not clean outer-boundary physics.

The final PC-GH, reused Z4c, initial-data, finiteness-window, localization,
boundedness, puncture-power, and Cartesian-slice plots are under
`docs/figures/pc_gh_localization_20260902/regular55/`. Only M/16, M/20, and M/24 are
included. Raw PC-GH runs are under
`qualification-runs-20260902/regular55-final/pcgh-{direct,hyper}/`; the quarantined
`pcgh-hyper-invalid-serial-under-mpirun` directory is execution-provenance evidence
and is excluded from every analysis result.

Final classification:

```text
PARTIAL IMPROVEMENT
```

The formulation removes the immediate nonfinite failure and extends every tested
survival window to `6M` without guard dependence. It is not a `QUALIFIED IMPROVEMENT`
because the negative puncture powers, `rho ~ N^1.12` inner growth, worsening `L/B`,
and nonconvergent full-domain reduction/curl behavior violate the explicit boundedness
gate. Perturbed, boosted, spinning, binary, and production outer-boundary promotion
remain stopped. The next technical question is why evolved `rho=alpha/w`, despite its
stationary target power, develops a resolution-growing wormhole-to-trumpet inner
profile in this first-order system.
