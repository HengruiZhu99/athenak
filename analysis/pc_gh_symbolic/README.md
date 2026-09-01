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

Not yet covered, and therefore not established by these scripts:

- equivalence to standard FO-GH and the full principal symbol/symmetrizer;
- Gauge A0 or Gauge B;
- puncture asymptotics and source-cancellation conditioning.

The production implementation must not begin using any formula still classified
`NOT ESTABLISHED`, `FAILED`, or subject to an unmet `CONDITIONAL` hypothesis in
`docs/pc_gh_derivation.md`.
