# Production zero_rate boundary audit

This is an independent read-only audit of the production snapshot in
`athenak-review/review/domain2048-20260919/source`, nominal
`8b694211-plus-static-floor`. No production source, campaign state, checkpoint,
or scheduler job was modified. The rejected `physical_constraint_radiation`
prototype is not the source being audited.

## Source findings

1. **There is no independent-face RHS averaging in production.**
   `src/z4c/z4c_Sbc.cpp:1456` assigns disjoint ownership: x1 owns its incident
   cells, x2 skips those cells, and x3 skips both preceding categories.
   The one owner constructs `side[]` from all incident physical faces and makes
   one characteristic solve in the normalized composite metric normal.
2. All six face signs pass the algebraic orientation audit. The first-order
   principal convention is `u_t=A_n D_n u`; a positive derivative eigenvalue
   propagates into the domain. The code selects that positive incoming branch
   relative to the outward normal. Its one-sided `CoordinateDerivative`
   returns the coordinate derivative before contraction with the normal.
3. With fixed coefficients, zero_rate sets all ten incoming characteristic
   time rates to zero. The ten momentum-like rates
   `(Khat,Theta,A_STF[5],Gamma[3])` are determined by derivatives of the ten
   untouched configuration rates `(chi,g_STF[5],alpha,beta[3])`.
   Sources and dissipation in incoming combinations are removed by this solve.
   It is not a boundary condition directly on the physical constraints
   `(H,M_i,Theta,Q^i)`.
4. The differentiated RHS fields are the configuration fields, which the
   boundary kernel does not write. The momentum fields have unique ownership.
   No in-place stencil race was identified. **However, an independent uncomputed
   RHS ghost read is present:** a non-diagonal metric tilts the contravariant
   normal at a physical coordinate face. At an internal tangential block edge,
   `NormalDerivative` then calls `CoordinateDerivative` with `side=0`, whose
   centered +/-1 stencil reads an RHS ghost. The volume RHS does not populate
   that ghost. Disjoint writes do not resolve this input-validity problem.
   The conformally flat, one-block models below do not exercise this bug.
5. Linear ghost extrapolation is not consistent with a sixth-order centered
   second derivative near the boundary. On `f=x^2`, the first two active values
   of the actual D6 second derivative become `-7/30` and `67/30`, rather than 2,
   independently of grid spacing. Quadratic ghosts restore that polynomial.
   Nevertheless, the tests below show that increasing extrapolation order alone
   does not remove the growing mode.
6. Four-cell linear extrapolation has weights `(5,-4)`, absolute sum 9.
   Sequential tensor-product ghost filling gives absolute sums 81 at an x/y
   ghost corner and 729 at a three-face corner. These are worst-case operator
   amplification factors, not a claim that every solution is amplified by them.
   They explain why a ghost metric can become invalid before an active metric;
   they do not locate the original unstable-mode injection.

## Bounded discrete models

`corner_model.py` implements the full 20-field flat, trace-free linear system,
with production `G=1`, kappa1=.1, kappa2=0, eta=2 and lapse residual damping=.1.
It includes sixth-order bulk first/second/mixed derivatives, KO8=.5, polynomial
ghost continuation and the actual zero_rate characteristic solve. The model is
two-dimensional, with physical x/y boundaries and no z dependence. It omits
background gradients, small background shift, matter, AMR and inter-block
communications. The characteristic rows imported by the helper are byte-for-
byte identical to the production snapshot (SHA256 below).

At 8x8 points with h=32M, all listed eigenpairs except highly ill-conditioned
degree-7 cases have normalized residuals around 1e-15. The original leading
mode is real and contains metric, Gamma and nonzero Theta. It is not primarily
a corner-localized eigenvector. The original 12x12 model grows at .008968/M.

| Boundary/discretization | Largest real eigenvalue /M |
|---|---:|
| Production linear ghosts, D2 boundary derivative | .00827707 |
| Quadratic ghosts, D2 | .00713243 |
| Cubic ghosts, D2 | .00649970 |
| Degree-5 ghosts, D2 | .00567088 |
| Degree-5 ghosts, D6 boundary derivative | .01388273 |
| Independent-face averaged RHS at corners | .00827547 |
| First-face ownership normal at corners | .00827630 |
| Diagnostic kappa=0, other damping retained | <5e-9 numerical neutral-mode noise |
| Diagnostic eta=0 | .00146982 |
| Diagnostic lapse damping=0 | .00743152 |

The diagnostic parameter removals are mechanism tests, not proposed production
changes. Periodic continuum and discrete symbols at the sampled axial/oblique
wavenumbers have no positive real eigenvalues; the modeled positive branch
requires the boundary treatment.

Several seemingly reasonable discrete fixes fail:

* A p-only incoming-amplitude penalty retaining the volume sources, summed over
  incident faces, gives .00171774/M for tau=1 and .00260921/M for tau=2.
* Applying the principal zero_rate projection and then restoring all original
  damping sources gives .01438892/M with linear ghosts, .00447003/M quadratic.
* A compatible low-order SBP reference plus face-summed p penalties has an
  8x8/h32 spectral pass, but fails at h8 (.00704225/M) and 12x12/h32
  (.00063629/M). Its isolated coarse pass is not a stability result.

No candidate in that table is recommended as a cure. In particular, this audit
does not recommend globally reducing the production spatial order, suppressing
kappa1, or resetting residuals. Production dt=.0375M is much smaller than the
timescales above, so RK3 does not suppress these positive semidiscrete modes.

## Independent continuum check

`continuum_controls.py` uses the independent production Schur half-space model
from `../gauge-audit/production_halfspace.py`. It reproduces an original
zero_rate growing root at flat coefficients, G=1, tangential wavenumber .1/M:
lambda=.024565132/M, normalized boundary singular value 5.7e-10.
The gauge audit separately verifies the weak-trumpet-coefficient root and its
bulk/boundary residuals. This is **not** the previously rejected radiation
prototype's mixed gauge/physical root.

At this one tangential frequency, replacing only the four physical constraint
boundary rows removes the known real root; retaining the old gauge/TT rows,
the minimum singular value on lambda=.015..035 is >.032 for outgoing physical
Theta/Q radiation and >.038 with diagonal kappa Robin terms. Replacing all ten
configuration rows by reflecting Dirichlet conditions gives about .995 in the
same interval. These are bounded discriminator tests, not uniform Kreiss
bounds or proofs of full nonlinear stability. The damping-coupled candidate
below is checked further by the gauge-audit agent.

## Constraint-compatible damping and the proposed local candidate

Use frozen conformal-flat coefficients, kappa2=0, no kappa time roll, and the
actual production Gamma damping `-2 alpha kappa1 Q`. Define

```
D0 = partial_t - beta^i partial_i
Q^i = Gamma_evolved^i - Gamma_metric^i
H = physical ADM Hamiltonian constraint
M_i = physical ADM momentum constraint
c = alpha sqrt(chi),  sigma = alpha kappa1
```

The exact frozen constraint propagation equations are

```
D0 H   = -2 alpha chi div M
D0 M_i = -alpha/2 partial_i H
         +alpha chi/2 (Delta Q_i - partial_i div Q)
         +2 sigma partial_i Theta
D0 Theta = alpha H/2 + alpha chi div Q/2 - 2 sigma Theta
D0 Q_i   = 2 alpha M_i + 2 alpha partial_i Theta - 2 sigma Q_i.
```

Here indices on Q in the displayed Euclidean frozen frame are equivalent.
`constraint_identity_check.py` checks these against the independent full
20-field volume symbol, including nonzero constant shift; maximum error is
1.67e-16. Consequently,

```
(D0^2 - c^2 Delta + 2 sigma D0) Q = 0
(D0^2 - c^2 Delta + 2 sigma D0) Theta = -alpha chi sigma div Q.
```

For zero normal shift, the exact exterior Laplace-Fourier decaying solution has
`rho^2=(lambda^2+2 sigma lambda)/c^2 + |k_T|^2`. Its Dirichlet-to-Neumann relations
are

```
Dn Q + rho Q = 0
Dn Theta + rho Theta - kappa1 Q_n/2
    + kappa1 div_T Q_T/(2 rho) = 0.
```

The second relation follows from the resonant particular solution
`x exp(-rho x)` of the forced Theta equation. The normal coupling is not an
optional empirical damping term. Taking the leading local normal-incidence
approximation gives

```
F_Q     = D0 Q + c Dn Q + sigma Q
F_Theta = D0 Theta + c Dn Theta + sigma Theta
          -sigma sqrt(chi) Q_n/2.
```

The normal coupling remains the same at leading order for nonzero normal shift.
The exact oblique relation contains the nonlocal tangential term above; dropping
it must be documented as a first-order absorbing approximation, not an exact
all-angle nonlinear constraint-preserving boundary condition.

The homogeneous damped Q-wave Robin condition has nonincreasing energy
`E_bulk + (c sigma/2) integral_boundary Q^2` in the zero-shift frozen problem:
its derivative is `-2 sigma integral Qt^2 - c integral_boundary Qt^2`.
Theta is triangularly forced by Q. This supplies a meaningful constraint-wave
design, but full metric/gauge boundary compatibility and finite-difference
stability still require testing; the old strong-field radiation prototype
remains rejected.

The old physical-constraint helper uses the residual covector
`z_i = gtilde_ij Q^j/2`. In that notation the new source terms are

```
f_theta += sigma * (theta - sqrt(chi) normal_u[i] z[i]);
f_z[i] += sigma * z[i];
```

before conversion `f_q^i=2 gtilde^ij f_z_j`. Sigma must match the volume source;
unsupported kappa2/time-roll conventions must not silently reuse this formula.
The operator should preserve full-minus-background exact zero, immutable RHS
stencil inputs, all-face ownership and physical response to nonzero residuals.
The original characteristic-rate derivative must also use a valid active-cell
stencil where tangential RHS ghosts have not been computed/exchanged; fixing
the new helper alone does not fix that original path.

## Provenance

```
z4c_Sbc.cpp a06506b322bc44061757203e009f080cacdf210dcd8285ad47a6b26c327e12af
z4c_bcs.cpp 5db0e7a37c835a83ba3c1b2185ab777246fbe70da3e2e755d310cb13d8c7cbc0
z4c_calcrhs.cpp 1284a39e58e8fa21f46b26e38173cfd30c9e3cd43d8810026e01a6d482131d56
check_residual_characteristics_numeric.py
  d3d4318b5c5575c43f3320ab029cf3534cc3b3e1e8f3859e0df872aa454a5a79
```

All numerical scripts/results in this directory are bounded diagnostics.
No Aurora submissions or production modifications were made by this audit.
