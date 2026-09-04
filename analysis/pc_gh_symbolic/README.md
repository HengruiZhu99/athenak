# PC-GH symbolic audit

These scripts are an independent, from-scratch audit of the conformal identities needed
by PC-GH. They do not import generated equations or any FO-GH/Ref-GH implementation.

The 2026-09-04 `verify_fo_gh_gamma2.py` audit checks the coupled Pi/Phi damping
increment through the existing inverse map, dimensions, characteristic fields,
symmetrizer, variable-rate subsidiary equations, and puncture source powers.
It also reproduces counterexamples to exact off-constraint production equivalence
and to invertibility after the moving-puncture gauge substitution. Its PASS messages
mean that the audit reproduced those results, not that the scheme is qualified.
See `docs/pc_gh_gamma2_audit.md` and the separate compiled production oracle in
`analysis/pc_gh_gamma2/`. The old round-trip and constrained tensor tests alone
cannot establish full off-constraint principal-symbol similarity.

Create an isolated environment and run:

```bash
python3 -m venv /tmp/pc-gh-sympy
/tmp/pc-gh-sympy/bin/python -m pip install -r analysis/pc_gh_symbolic/requirements.txt
/tmp/pc-gh-sympy/bin/python analysis/pc_gh_symbolic/run_all.py
```

Current coverage:

| Script | Exact checks | Classification |
|---|---|---|
| `verify_puncture_regular_55.py` | `w,rho,p,L,Cperp,Z` map; H/E/S/T/scaled-momentum identities; direct and switched gauge gradients including `S'`; wormhole/trumpet powers; absence of puncture-field denominators | exact `PROVED ON w>0,rho>0` equivalence plus denominator-free preferred expressions |
| `verify_regularization.py` | regular lapse Hessian; physical/conformal lapse Hessian; scalar curvature; Hamiltonian; trace-free curvature/lapse tensor; scaled momentum | `PROVED ON r>0` for expressions using positive `chi` and `A`; otherwise `PROVED` |
| `verify_q_projection.py` | product-rule consistency and trace-free property of the simultaneous metric/Q projection | `PROVED` for nonsingular conformal metric |
| `verify_flat_algebra_randomized.py` | 10,000 seeded SPD cofactor-inverse, determinant, Atilde/Q trace projection, and PC-GH/ADM round-trip trials over eight decades in chi and six metric scales | broad binary64 Gate 2 regression, not an exhaustive proof |
| `verify_conformal_ricci.py` | Brown first-order Ricci against coordinate Ricci for a non-diagonal exactly unimodular metric at 18 exact rational component/point pairs | exact regression supporting the written `PROVED` index derivation |
| `verify_primary_projections.py` | normal-normal pi equation; corrected K divergence count; corrected Atilde nonlinear Z term; mixed-projection lapse-acceleration term; exact counterexamples to the supplied K, Atilde, and Lambda regression targets | pi and corrected terms `PROVED`; supplied K/Atilde/Lambda targets `FAILED` |
| `verify_gradient_rhs.py` | exact product-rule expansions of all chi/A/beta/gtilde source gradients and the compatible-versus-standard curl difference | `PROVED` conditional on differentiable metric-only/prescribed Gauge A sources |
| `verify_z4c_mp_gauge.py` | exact cancellation of the regular GH source representation to the direct 1+log/Gamma-driver A, beta, Y, and B equations | `PROVED` for constant `eta` |
| `analyze_z4c_mp_principal.py` | exact 50-field algebraic-tangent principal polynomial and eigenspace ranks plus the regular-state similarity map with determinant `-32 rho^4 w^9` | known direct-gauge defects map to `rho^2 w^4=4/3`, `rho w=2`, and `rho w^3=2/3`; no new positive-rho surface |
| `verify_4d_component_oracle.py` | arbitrary exact rational 3+1 metric point jet; all ten covariant reduced four-tensor components; direct K, Atilde, pi, and Lambda derivatives; Brown-Ricci, Gauss-normal, and Hamiltonian identities | corrected primary equations `PROVED` at the non-diagonal point jet; supplied targets remain excluded |
| `verify_fo_gh_map.py` | exact constrained PC-GH to standard FO-GH to PC-GH round trip for a non-diagonal rational state | algebraic variable-map invertibility `PROVED ON r>0` |
| `generate_gauge_a0_table.py` | independent stationary 1+log implicit solution, isotropic-radius ODE, target-source identities, and inner exponents | Gauge A0 continuum construction `PROVED`; double-precision table generation numerically audited |
| `audit_gauge_a0_cancellation.py` | production table interpolants, all radial/tangential target tensors, 387 named temporaries and additive RHS terms over 73 radii in binary64, long double, and 100-digit arithmetic | fails on any additive RHS term with fitted divergent inner power; logs genuine raw derivative singularities separately |
| `audit_bowen_york_cancellation.py` | time-symmetric, momentum, spin, and combined conformally flat Bowen-York leading fields; 217 named fields, temporaries, and additive RHS terms over 81 radii in binary64, long double, and 100-digit arithmetic | puncture regularity/conditioning audit; the nonzero-momentum/spin cases intentionally omit the regular TwoPunctures correction and are not constraint-satisfying initial data |
| `analyze_frozen_operator.py` | dense 55-field lower-order plus actual-FD Fourier response extracted by `pc_gh_trumpet_a0` with `frozen_operator=true`; raw and 50-dimensional algebraically projected spectra, eigenvector conditioning, Euclidean logarithmic norm, and non-normality | numerical diagnostic only; the formulation-energy symmetrizer analysis remains separate and mandatory |
| `verify_ko_symbol.py` | exact stencil weights and normalized Fourier symbols for every supported KO order | normalized symbols are `-sin(theta/2)^(2p)` and therefore nonpositive |
| `verify_reduction_constraint_growth.py` | exact tangential trace-free Q pointwise-frozen and true reduction-constraint propagation rates on a radial shift-gradient background | pointwise rate `(5t-2u)/3`; reduction rate `2(t-u)/3`, still positive at the Gauge-A0 target and untouched by Gauge A1 |
| `analyze_gauge_wave_convergence.py` | named per-field errors and GH/ADM/reduction/curl residuals over a resolution ladder | fail-closed all-sector second-order shifted harmonic-wave gate |
| `analyze_robust_minkowski.py` | normalized amplification, endpoint growth, and late-time fitted growth for every GH/reduction/curl family | fail-closed resolution-growing-instability search |
| `analyze_bowen_york_residuals.py` | exact time-symmetric isotropic Schwarzschild ADM data through production ADM-to-PC-GH conversion, production RHS, and production constraints; four-resolution RMS ladder plus maximum locations | fail-closed pointwise initial-data/source regression; not a wormhole-to-trumpet evolution gate |
| `verify_source_policy.py` | scans every production PC-GH C++ source/header for `Dxx`/`Dxy`, legacy GH/Z4c includes, and puncture-field quotients in preferred evolution paths | fail-closed source/denominator-policy gate |

Not yet covered, and therefore not established by these scripts:

- Gauge B;
- constraint-satisfying boosted/spinning TwoPunctures data and evolution.

The production implementation must not begin using any formula still classified
`NOT ESTABLISHED`, `FAILED`, or subject to an unmet `CONDITIONAL` hypothesis in
`docs/pc_gh_derivation.md`.
