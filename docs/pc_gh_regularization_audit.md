# PC-GH puncture-regularization audit

## Status

This audit records algebraic evidence through
`f2b20729b315fca553449661749e94b010452fe6`.  The theorem domain is `r>0` with
positive `A`, positive `chi`, and nonsingular positive-definite `gtilde`.  It does not
assert a uniformly conditioned continuum extension at an exact `A=chi=0` point.

The source of truth for definitions and proofs is `docs/pc_gh_derivation.md`.  The
scripts under `analysis/pc_gh_symbolic` are independent executable checks, not equation
generators for the production kernel.

## Regular variables and composites

The evolved lapse variable is `A=alpha^2`; no production equation evolves `alpha`
directly.  The first-order variables are

\[
X_i=\partial_i\chi,\quad Y_i=\partial_i A,\quad
Q_{kij}=\partial_k\tilde\gamma_{ij},\quad
B_i{}^j=\partial_i\beta^j.
\]

The puncture monitors are

\[
W_i=X_i/\sqrt\chi,\quad L_i=Y_i/\sqrt A,\quad
r_-=\chi/\sqrt A,\quad r_+=\sqrt{A/\chi}.
\]

The RHS uses the regular conformal identities for the physical scalar curvature,
Hamiltonian and momentum constraints, lapse Hessian, and trace-free curvature/lapse
combination.  It does not construct physical Christoffels or a physical Ricci tensor.

## Exact audit matrix

| Gate | Executable evidence | Classification |
|---|---|---|
| regular scalar/tensor identities | `verify_regularization.py` | exact; `PROVED ON r>0` where division by `A` or `chi` is used |
| metric and `Q` projection | `verify_q_projection.py` | exact for nonsingular `gtilde` |
| Brown conformal Ricci | `verify_conformal_ricci.py` | exact at 18 rational component/point pairs for a non-diagonal unimodular family |
| primary projections | `verify_primary_projections.py` | corrected `K`, `pi`, `Atilde`, and `Lambda` sectors pass |
| gradient product rules | `verify_gradient_rhs.py` | exact for metric-only or prescribed differentiable gauge sources |
| independent four-dimensional reduced equation | `verify_4d_component_oracle.py` | all ten covariant components and corrected primary equations pass at a rational point jet |
| PC-GH / standard FO-GH map | `verify_fo_gh_map.py` | exact constrained round trip; `PROVED ON r>0` |
| production dependency policy | `verify_source_policy.py` | all nine current `src/pc_gh` production files pass |

The controlling regression targets were not assumed correct.  Independent checks found
and retained exact counterexamples to three supplied targets:

- the supplied `K` equation double counted `-chi div Z`;
- the supplied nonlinear `Atilde` term had the wrong lowering/contraction structure;
- the supplied `Lambda` equation omitted `-(chi/2) C_perp L^i`.

The corrected equations, rather than the failed targets, are implemented.

## Gauge A0 regularity evidence

The stationary-table generator solves the implicit stationary 1+log relation and the
isotropic-radius ODE independently.  For the 4097-point, `M=1` table it verifies:

| Quantity | Maximum residual |
|---|---:|
| implicit stationary 1+log relation | `7.458e-15` |
| advective 1+log identity | `6.901e-16` |
| `h_perp` target definition | `1.221e-15` |
| radial `h^i` target definition | `6.939e-18` |

The inner fitted powers are `e_A=2.18259458` and `e_chi=2.00000000`.  The committed
table SHA-256 is
`ea34841f37e6908c3f169b2551492440d699b60f0de0d45fd5e0be704c017dd7`.

These checks establish the continuum Gauge A0 construction and deterministic table.
They do not close the mandatory production source-cancellation audit.

## Runtime diagnostics and hard stops

Every PC-GH constraint pass records `r_-`, `r_+`, `|W|`, `|L|`, primary-RHS and
gradient-RHS maxima in addition to GH, physical, reduction, curl, and algebraic
constraints.  The stationary target problem also writes per-variable maxima and their
locations in serial runs.

Progression must stop on loss of positive `A` or `chi`, loss of conformal-metric SPD,
nonconvergent constraints, a resolution-growing instability, a divergent source
temporary, a defective principal symbol, a nonpositive symmetrizer, or a stationary
residual produced by huge cancellation.  Floors, clipping, increased KO, or weakened
norms are not acceptable substitutes.

## Open audit items

- log every production RHS temporary versus radius for binary64, long double where
  meaningful, and high precision;
- identify and algebraically remove any avoidable divergent cancellation;
- repeat the audit for Bowen-York data once that conversion exists;
- sample the full frozen operator, not just its principal part;
- repeat backend-sensitive checks under MPI and on the intended GPU backends.
