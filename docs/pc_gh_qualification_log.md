# PC-GH qualification log

## Claim policy

This is an append-oriented evidence log.  A build, a short run, or a finite residual
does not by itself qualify the solver.  `PASS` below means only that the named bounded
gate passed.  `OPEN` and `BLOCKED` are not silently replaced by weaker criteria.

Current evidence commit: `2404f5ada107341af8e0a2c6a8651c12a18b4548`.
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
| 2 flat algebra | PARTIAL | exact projection and constrained state-map tests pass; broader randomized SPD sampling remains |
| 3 Minkowski pointwise | PASS | exact state, ADM round trip, RHS, and diagnostics at floating-point zero |
| 4 wave convergence | PARTIAL | nonlinear harmonic gauge-wave implementation and finite `t=1` run; required three-resolution all-sector order ladder remains |
| 5 robust Minkowski | OPEN | random perturbation and resolution-growth search not run |
| 6 stationary trumpet pointwise | PASS | continuum target, decomposed three-precision conditioning audit, maximum locations, and second-order RMS residual ladder pass on their stated punctured domains |
| 7 stationary trumpet evolution | BLOCKED | no derived nonperiodic diagnostic/characteristic outer BC and no frozen-operator clearance |
| 8 perturbed trumpet | OPEN | waits on gates 6 and 7 |
| 9 Bowen-York to trumpet | OPEN | existing ADM initial-data conversion has not been exercised as a qualified transition |
| 10 Gauge A1 | DEFERRED | only if A0 needs bounded feedback after linear analysis |
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

## 2026-09-01 — exact harmonic Minkowski

Input: `tst/inputs/pc_gh_minkowski.athinput`.

Grid: `8 x 4 x 4`, one mesh block, periodic.  CFL `0.25`, `nlim=0`, harmonic gauge.

Result: `PASS: exact PC-GH Minkowski state, ADM round trip, RHS, and diagnostics`.
This is a pointwise construction/RHS gate, not an evolution-stability result.

## 2026-09-01 — nonlinear harmonic gauge wave

Input: `tst/inputs/pc_gh_gauge_wave.athinput`.

Grid: `32 x 1 x 1`, two mesh blocks, periodic.  CFL `0.1`, `t=1`, 321 cycles,
amplitude `0.01`, harmonic gauge.

Result: `L1=1.593071e-04`, `Linf=2.593841e-03`.  This confirms a finite nonlinear
periodic evolution after the Gauge A0 source branch was added.  It does not satisfy the
required three-resolution, all-sector convergence gate.

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

## Current hard stop

Do not start a stationary evolution campaign.  The complete frozen-operator stability
analysis is open, and only periodic physical boundaries exist.  A periodic finite box
is not acceptable as a production single-hole outer boundary.
