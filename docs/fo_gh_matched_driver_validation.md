# Matched-driver FO-GH validation record

## Controlling outcome

**FORMULATION NOT ESTABLISHED.** V0 confirms that matched weighting removes
the previous gauge-driver regularity obstruction, but the exact 58-dimensional
chart fails the required puncture-conditioning gate. No new-system puncture
simulation has been launched.

## Branch provenance

- parent commit: `e0d8c653d30d41a676467c23e02f4969f7629156`;
- branch: `codex/fo-gh-matched-driver-puncture-20260818`;
- scope: vacuum FO-GH only;
- unrelated dirty boundary, Kokkos, and external-directory work was preserved.

## V0 results

Command:

```sh
PYTHONPATH=tst/test_suite python3 \
  tst/test_suite/fo_gh/matched_driver_pullback_audit.py \
  --samples 1000 --maximum-n 64
```

| check | result |
|---|---:|
| independent old-map inverse error | `1.02855344798917668e-15` |
| old normal-mixing power | `-2.182000` |
| matched-map inverse error | `2.45463971295843985e-16` |
| matched-driver dense-oracle error | `2.06084775343976128e-15` |
| matched moving-puncture target error | `3.09222093455266125e-15` |
| maximum normalized-map condition number | `1.20659544338768487e+01` |
| normalized-map inverse residual | `0` |
| minimum scanned production power | `0` |
| high-precision scan maximum relative difference | `1.7061e-14` |

All five direct test functions pass. The scanner uses 110-digit Decimal
references at every \(r=2^{-n}M\) for \(n=1,\ldots,64\).

## Gate ledger

| gate | status | evidence |
|---|---|---|
| independent reproduction of old obstruction | pass | fresh block-matrix inverse and power audit |
| exact matched map and inverse | pass | 1000 random states |
| exact matched driver pullback | pass | component and dense 4x4 oracles |
| regular moving-puncture target | pass | 1000 random states |
| normalized-map conditioning | pass for bounded beta | random condition scan |
| individual driver-intermediate regularity | pass | complete named power inventory and 110-digit scan |
| finite-radius 58D regular/parent map | pass | 1000 round trips and tangent maps |
| finite-radius symmetrizer/hyperbolicity | pass | Minkowski and weak random states |
| puncture-uniform characteristic conditioning | **fail** | \(\kappa(R)\) grows to \(1.18\times10^{10}\) by \(r=1/16\) |
| complete pulled-back Einstein source | not run | stopped after conditioning failure |
| first-bad-state old-control replay | complete | fail-closed capture at cycle 1657, \(t=3.4274776869830532M\) |
| V1--V10 candidate numerical ladder | not run | prohibited after analytic stop |

## Prior control evidence

The inherited corrected-source control at coarse spacing collapsed at
3.431611M. Historical runs at medium and fine collapsed at 3.024995M and
2.658676M, respectively, but those runs predate later diagnostic/source
corrections and do not substitute for the requested telemetry replay. No claim
is made that the matched driver fixes this numerical behavior.

## First-bad-state control replay

The inherited coarse old-formulation control was replayed from \(t=0\) on
Perlmutter allocation `57213683`, using four MPI ranks and four distinct A100
GPU UUIDs. The input reproduced the 232-MeshBlock static-refinement tree and
\(\Delta x_{\min}=1/16M\). The executable was built from the exact parent
commit, not from a candidate matched-driver implementation.

The last periodic history record was at \(t=3.400608334264673M\). Fail-closed
telemetry caught the first invalid state at cycle 1657 and
\(t=3.4274776869830532M\), with proposed `dtnew=0` and maximum characteristic
speed `0.964440912002851`. Four symmetry-related cells failed at coordinate
magnitudes \(0.21875M\), all at radius \(0.378886114155692M\) on physical
refinement level 3. Their coordinate distance to the nearest finest-grid cube
face at \(|x_i|=2M\) is \(1.78125M\); the event is therefore a near-puncture
interior failure, not an SMR-interface or outer-boundary arrival.

For the rank-2 representative cell, telemetry recorded:

| quantity | value |
|---|---:|
| \(\alpha\) | `-1.4805374010543476e-1` |
| \(A=\alpha^2\) | `2.1919909959207621e-2` |
| \(\chi\) | `3.8610807779593842e-1` |
| \(\det\tilde\gamma\) | `1.2097452854436379` |
| eigenvalues of \(\tilde\gamma\) | `0.1668534851, 2.6830400147, 2.7022872130` |
| gauge residual \((h-f)_\perp\) | `-3.0429968689637848e4` |
| largest spatial gauge residual | `-2.8266143330774353e3` |
| largest state | `Atzz = 2.7193503378429919e4` |
| largest last-stage RHS | `Atzz = 2.9642533896420613e7` |

Bad-cell component maxima include \(|a_i|=0.4765721\), \(|Q|=12.58423\),
\(|X|=0.5434604\), \(|B|=3.559459\), \(|\Lambda|=3812.817\),
\(|\pi|=23028.48\), \(|K|=13524.05\), and
\(|\widetilde A|=27193.50\). These are bad-cell values, not global extrema.
The inherited telemetry did not capture the requested global minima or the
largest numerical \(d_i a_j\). Hamiltonian, momentum, GH, reduction, curl,
and conformal-constraint point values were deliberately not evaluated after
the lapse became invalid; the log reports
`constraints=unavailable_invalid_metric_or_speed`. The retained histories
give the last valid region-integrated diagnostic sample at
\(t=3.400608334264673M\).

This observation localizes the old-control failure but does not establish its
cause and does not demonstrate that the matched driver would cure it.

## Numerical ladder disposition

The requested analytic stop was reached before production implementation.
Consequently the 2-by-2 ablation, frozen-coefficient/semidiscrete spectra,
stationary-trumpet runs, three-resolution wormhole-to-trumpet runs, and SMR
candidate runs were not performed. There is no new-system stable-evolution
time to report. This is an intentional fail-closed outcome, not missing
qualification evidence.
