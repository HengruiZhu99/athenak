# PC-GH symbolic audit

These scripts are an independent, from-scratch audit of the conformal identities needed
by PC-GH. They do not import generated equations or any FO-GH/Ref-GH implementation.

Create an isolated environment and run:

```bash
python3 -m venv /tmp/pc-gh-sympy
/tmp/pc-gh-sympy/bin/python -m pip install -r analysis/pc_gh_symbolic/requirements.txt
/tmp/pc-gh-sympy/bin/python analysis/pc_gh_symbolic/run_all.py
```

Current coverage:

| Script | Exact checks | Classification |
|---|---|---|
| `verify_regularization.py` | regular lapse Hessian; physical/conformal lapse Hessian; scalar curvature; Hamiltonian; trace-free curvature/lapse tensor; scaled momentum | `PROVED ON r>0` for expressions using positive `chi` and `A`; otherwise `PROVED` |
| `verify_q_projection.py` | product-rule consistency and trace-free property of the simultaneous metric/Q projection | `PROVED` for nonsingular conformal metric |
| `verify_conformal_ricci.py` | Brown first-order Ricci against coordinate Ricci for a non-diagonal exactly unimodular metric at 18 exact rational component/point pairs | exact regression supporting the written `PROVED` index derivation |
| `verify_primary_projections.py` | normal-normal pi equation; corrected K divergence count; corrected Atilde nonlinear Z term; mixed-projection lapse-acceleration term; exact counterexamples to the supplied K, Atilde, and Lambda regression targets | pi and corrected terms `PROVED`; supplied K/Atilde/Lambda targets `FAILED` |
| `verify_gradient_rhs.py` | exact product-rule expansions of all chi/A/beta/gtilde source gradients and the compatible-versus-standard curl difference | `PROVED` conditional on differentiable metric-only/prescribed Gauge A sources |
| `verify_4d_component_oracle.py` | arbitrary exact rational 3+1 metric point jet; all ten covariant reduced four-tensor components; direct K, Atilde, pi, and Lambda derivatives; Brown-Ricci, Gauss-normal, and Hamiltonian identities | corrected primary equations `PROVED` at the non-diagonal point jet; supplied targets remain excluded |
| `verify_fo_gh_map.py` | exact constrained PC-GH to standard FO-GH to PC-GH round trip for a non-diagonal rational state | algebraic variable-map invertibility `PROVED ON r>0` |
| `generate_gauge_a0_table.py` | independent stationary 1+log implicit solution, isotropic-radius ODE, target-source identities, and inner exponents | Gauge A0 continuum construction `PROVED`; double-precision table generation numerically audited |
| `audit_gauge_a0_cancellation.py` | production table interpolants, all radial/tangential target tensors, 387 named temporaries and additive RHS terms over 73 radii in binary64, long double, and 100-digit arithmetic | fails on any additive RHS term with fitted divergent inner power; logs genuine raw derivative singularities separately |
| `analyze_frozen_operator.py` | dense 55-field lower-order plus actual-FD Fourier response extracted by `pc_gh_trumpet_a0` with `frozen_operator=true`; raw and 50-dimensional algebraically projected spectra, eigenvector conditioning, Euclidean logarithmic norm, and non-normality | numerical diagnostic only; the formulation-energy symmetrizer analysis remains separate and mandatory |
| `verify_ko_symbol.py` | exact stencil weights and normalized Fourier symbols for every supported KO order | normalized symbols are `-sin(theta/2)^(2p)` and therefore nonpositive |
| `analyze_gauge_wave_convergence.py` | named per-field errors and GH/ADM/reduction/curl residuals over a resolution ladder | fail-closed all-sector second-order shifted harmonic-wave gate |
| `analyze_robust_minkowski.py` | normalized amplification, endpoint growth, and late-time fitted growth for every GH/reduction/curl family | fail-closed resolution-growing-instability search |
| `verify_source_policy.py` | scans every production PC-GH C++ source/header for `Dxx`/`Dxy` calls and legacy GH/Z4c includes | fail-closed source-policy gate |

Not yet covered, and therefore not established by these scripts:

- Gauge B;
- Bowen-York source-cancellation conditioning.

The production implementation must not begin using any formula still classified
`NOT ESTABLISHED`, `FAILED`, or subject to an unmet `CONDITIONAL` hypothesis in
`docs/pc_gh_derivation.md`.
