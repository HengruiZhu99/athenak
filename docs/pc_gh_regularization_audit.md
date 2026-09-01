# PC-GH puncture-regularization audit

## Status

This audit records algebraic evidence through
`f74a19ae4d425bccbbd1bff78db72bbe49502f42`.  The theorem domain is `r>0` with
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
| Bowen-York leading-field regularity | `audit_bowen_york_cancellation.py` | three-precision boundedness/conditioning audit on `r>0`; nonzero momentum/spin cases are not complete constraint-satisfying data |
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
| `h_perp` target definition | `1.332e-15` |
| radial `h^i` target definition | `6.939e-18` |

The inner fitted powers are `e_A=2.18259453` and `e_chi=2.00000000`.  The committed
table SHA-256 is
`122d84a52b4f19ea5c7e4c13a4e0bc8a9d488265d5d9df306bdd360978928eb5`.

These checks establish the continuum Gauge A0 construction and deterministic table.

The production cancellation audit logs 387 quantities at 73 radii from
`1.1e-8 M` to `100 M` in binary64, long double, and 100-digit arithmetic.  No additive
RHS term has a fitted inner power below `-0.25`.  The maximum 100-digit total RHS on the
open table domain is `5.118e-5` in the radial `Lambda` sector at
`r=2.0797956529e-8 M`.  The worst binary64 discrepancy from the 100-digit sum is
`2.827e-8` absolute and `3.547e-7` relative to the additive-term scale.

The audit separately reports `partial Atilde` with fitted power `-1.000001`.  This is
the genuine angular derivative of the finite but direction-dependent radial tensor,
not an additive RHS term; every production term multiplying it remains bounded on the
audited punctured domain.

## Bowen-York regularity evidence

`audit_bowen_york_cancellation.py` independently constructs conformally flat
Bowen-York leading fields with `psi=1+M/(2r)`, `A=chi=psi^-4`, and
`Atilde=psi^-6 Abar`.  It evaluates time-symmetric, momentum, spin, and combined
cases at 81 radii from `1e-8 M` through `100 M` in binary64, long double, and
100-digit arithmetic.  Each case logs 217 named state fields, derivatives,
temporaries, additive RHS terms, sums, and term scales.

No stored field, temporary, or additive RHS term has a fitted inner power below
`-0.25`.  The maximum 100-digit residual in the checked conformal Hamiltonian identity
`H + Atilde_ij Atilde^ij = 0` is `9.875e-101`.  Across all cases, the worst RHS-sum
discrepancy normalized by its additive-term scale is `7.612e-14` in binary64 and
`2.403e-17` in long double.

The time-symmetric member is exact Schwarzschild wormhole initial data.  The
nonzero-momentum and spin cases deliberately omit the regular elliptic correction
solved by TwoPunctures, so they establish only near-puncture scaling and source
conditioning.  They are not Hamiltonian-satisfying boosted or spinning initial data,
and they do not close Gates 9, 12, or 13.

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

- repeat the production-path ladder with constraint-satisfying nonzero-momentum and
  spin data from the existing TwoPunctures infrastructure;
- derive and apply the formulation-energy symmetrizer to the extracted full frozen
  operator; the present Euclidean logarithmic norm is not a substitute;
- derive a constraint-energy treatment for the positive tangential trace-free `Q`
  reduction mode; bounded Gauge A1 cannot affect it;
- repeat backend-sensitive checks under MPI and on the intended GPU backends.
