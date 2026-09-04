# PC-GH puncture-regularization audit

The [2026-09-04 gamma2 audit](pc_gh_gamma2_audit.md) finds that a bounded nonzero
coordinate-time rate does not ensure a regular full FO-GH damping pullback.
The coupled primary increments contain inverse lapse/conformal factors. A rate
`lambda=rho^2*w^4*f` with finite `f` regularizes those increments, but does not
repair the separate off-constraint baseline mismatch. The original denominator-free
production kernels remain unchanged; their regular extension is not an exact
off-constraint FO-GH variable transformation.

## Status

This audit now records the puncture-regular 55-field replacement implemented on
2026-09-02.  The theorem domain is `r>0` with positive `w`, positive `rho`, and a
nonsingular positive-definite `gtilde`.  The source contains a denominator-free
polynomial extension of every preferred evolution expression to `w=0` or `rho=0`;
the positive-field assumption is needed only for proving equivalence to the older
variables.  The qualification grids are cell-centered and never place a cell at the
exact puncture.

The source of truth for definitions and proofs is `docs/pc_gh_derivation.md`.  The
scripts under `analysis/pc_gh_symbolic` are independent executable checks, not equation
generators for the production kernel.

## Current regular variables and composites

The production state is

\[
\{w,\tilde\gamma_{ij},K,\tilde A_{ij},Z^i,C_\perp,\rho,
\beta^i,p_i,Q_{kij},L_i,B_i{}^j\},
\]

with

\[
w=\sqrt\chi,\quad \rho=\alpha/w,\quad p_i=\partial_iw,\quad
L_i=2\partial_i\alpha,\quad C_\perp=\pi+K,
\quad Z^i=\tilde\Gamma^i(Q)-\tilde\Lambda^i.
\]

`partial_i rho` is intentionally absent: it diverges as approximately
`r^-0.9087` on a stationary 1+log trumpet, while `rho`, `p`, and `L` have nonnegative
inner powers.  The RHS uses only multiplication/addition of puncture fields plus
conformal-metric inversion.  It does not construct physical Christoffels, a physical
Ricci tensor, or any quotient by `w`, `rho`, `alpha`, `chi`, or `A`.

The only puncture-field quotient is the defining input conversion `rho=alpha/w`.
It is initialization-only, uses the configurable resolution-independent
`initial_data_division_floor`, reports the global minimum unguarded `w` and guard
count, and fails on negative/nonfinite input.  The stationary A0 legacy table needs
the analogous initialization-only conversions.  Floor values `1e-12`, `1e-14`, and
`1e-16` produced byte-identical N=32 residual tables; no qualification cell activated
the default guard.

The remainder of this document below the current audit matrix describes the older
`A,chi,X,Y` investigation and is retained as provenance, not as the current storage
ABI.

## Source division audit

The final `rg -n '/|sqrt|pow('` audit covers the requested RHS, constraints, CFL,
projection, ADM conversion, PC-GH problem generators, and horizon adapter. Every
remaining quotient is classified here:

| path | remaining divisions | reason and safety condition |
|---|---|---|
| `pc_gh_calcrhs.cpp` | inverse grid spacings; constants `/2,/3`; smoothstep width; conformal `1/det(gtilde)`; A0 coordinate `r` and mass | finite-difference definition, fixed coefficients/parameters, conformal-metric inverse, or prescribed-table coordinates whose open `r>0` domain is checked; no puncture-field quotient |
| `pc_gh_constraints.cpp` | inverse grid spacings; constants; conformal `1/det(gtilde)`; analytic symmetric-eigenvalue normalization | diagnostic derivatives, fixed coefficients, conformal inverse, and a scale-invariant SPD eigensolver; no puncture-field quotient |
| `pc_gh_newdt.cpp` | conformal `1/det(gtilde)`; fixed `2/sqrt(3)`; `dx/speed` | conformal inverse, fixed characteristic factor, and the defining CFL quotient after explicit finite/positive speed checks |
| `pc_gh_projection.cpp` | `1/det(gtilde)` and fixed thirds | determinant-one and trace-free conformal projection; SPD is checked before use |
| `pc_gh_adm.cpp` initialization | `1/det(gamma)` and `alpha/guarded_w` | metric determinant extraction and the unavoidable defining map from ADM input to `rho`; initialization-only guard policy described above |
| `pc_gh_adm.cpp` masked output | `1/w^2` | evaluated only outside `physical_output_inner_radius`; masked cells get an explicitly invalid output-only extension |
| Bowen-York initialization/audit | coordinate `1/r`, powers of `r+M/2`, fixed thirds, sample-count RMS | analytic wormhole input on a cell-centered grid whose puncture is required to lie on faces, plus diagnostics; no evolution-field quotient |
| stationary A0 initialization | coordinate `1/r`, mass scaling, `alpha/guarded_w`, `dchi/guarded_w`, `dA/guarded_alpha` | legacy table-to-new-state conversion only; exact-center cells are rejected and the same configurable initialization guard is used |
| PC-GH horizon adapter | Chebyshev normalization and `1/w^2` | regular variables are interpolated first; physical reconstruction occurs only outside the declared inner mask after finite/positive checks |

Integer index arithmetic and path/comment slashes in the raw `rg` listing are not
field divisions. `verify_source_policy.py` independently strips comments and fails if
a quotient by `w`, `rho`, `alpha`, `chi`, or `A` enters a preferred evolution file.

## Exact audit matrix

| Gate | Executable evidence | Classification |
|---|---|---|
| regular 55-field map, equations, gauges, asymptotics | `verify_puncture_regular_55.py` | exact on `w>0,rho>0`; preferred expressions denominator-free |
| regular scalar/tensor identities | `verify_regularization.py` | exact; `PROVED ON r>0` where division by `A` or `chi` is used |
| metric and `Q` projection | `verify_q_projection.py` | exact for nonsingular `gtilde` |
| Brown conformal Ricci | `verify_conformal_ricci.py` | exact at 18 rational component/point pairs for a non-diagonal unimodular family |
| primary projections | `verify_primary_projections.py` | corrected `K`, `pi`, `Atilde`, and `Lambda` sectors pass |
| gradient product rules | `verify_gradient_rhs.py` | exact for metric-only or prescribed differentiable gauge sources |
| direct moving-puncture gauge | `verify_z4c_mp_gauge.py` | exact primary and STANDARD Y/B equations for constant `eta` |
| moving-puncture principal symbol | `analyze_z4c_mp_principal.py` | exact 50-field algebraic-tangent polynomial and eigenspace ranks; direct gauge has three defective surfaces; switched gauge is strongly hyperbolic only on its stated conditional domain |
| independent four-dimensional reduced equation | `verify_4d_component_oracle.py` | all ten covariant components and corrected primary equations pass at a rational point jet |
| PC-GH / standard FO-GH map | `verify_fo_gh_map.py` | exact constrained round trip; `PROVED ON r>0` |
| Bowen-York leading-field regularity | `audit_bowen_york_cancellation.py` | three-precision boundedness/conditioning audit on `r>0`; nonzero momentum/spin cases are not complete constraint-satisfying data |
| production dependency and denominator policy | `verify_source_policy.py` | all nine current `src/pc_gh` production files pass; preferred evolution rejects puncture-field quotients |

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

Every configured boundedness pass records min/max `w,rho,alpha`, maxima of
`p,L,Cperp,Z,K,Atilde,beta,Q,B`, determinant/principal-minor/eigenvalue bounds, every
GH/ADM/reduction/curl/algebraic diagnostic, both RHS-family maxima, and changes in all
four reduction and curl norms across restriction, prolongation, projection, and the
dedicated post-projection restriction/exchange/prolongation. The stationary target
problem also writes per-variable maxima and their locations in serial runs.

The one-puncture moving-gauge diagnostic is MPI-capable and fails closed at the first
nonfinite evolved state, RHS, constraint, or characteristic speed; negative `w/rho`;
nonfinite determinant/eigenvalue; or loss of conformal-metric SPD. The older 20M
uniform-periodic evidence below is superseded by the 2026-09-02 M/16--M/24 SMR study
in the qualification log.

Progression must stop on negative/nonfinite `w` or `rho`, loss of conformal-metric SPD,
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
- determine why the direct and switched moving-puncture wormhole ladders have
  resolution-growing GH/ADM endpoint norms before any perturbed, boosted, spinning,
  or binary promotion;
- establish that the switched gauge trajectory remains inside its conditional
  strong-hyperbolicity domain before considering spectral/SAT evolution;
- repeat backend-sensitive checks under MPI and on the intended GPU backends.
