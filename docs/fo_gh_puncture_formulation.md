# Vacuum regularized first-order GH formulation

## Scope

The `src/fo_gh/` module evolves the fixed 63-component, everywhere-regularized
vacuum first-order generalized-harmonic (FO-GH) system.  It does not evolve
ordinary physical `g_ab`, `Pi_ab`, or `Phi_iab` as production variables, and it
does not use fluid coupling, evolution excision, floors, clipping, or puncture
resets.  The physical standard-GH fields are reconstructed only by
`RegularToStandardGh` for tests and diagnostics.

Cartesian reference geometry is assumed:

```
bar_gamma_ij = delta_ij,  bar_Gamma^i_jk = 0,
D0 = partial_t - beta^k partial_k,  chi = exp(-4 phi).
```

## State ordering

`src/fo_gh/fo_gh_state.hpp` defines the stored order:

| Range | Variables | Components |
|---|---|---:|
| 0--5 | `gtilde_ij` | 6 |
| 6 | `chi` | 1 |
| 7 | `alpha` | 1 |
| 8--10 | `beta^i` | 3 |
| 11--16 | `Atilde_ij` | 6 |
| 17 | `K` | 1 |
| 18--20 | `Lambda^i` | 3 |
| 21 | `pi` | 1 |
| 22--39 | `Q_kij` | 18 |
| 40--42 | `X_i` | 3 |
| 43--45 | `a_i` | 3 |
| 46--54 | `B_i^j` | 9 |
| 55--58 | `h_perp`, `h^i` | 4 |
| 59--62 | `vartheta_perp`, `vartheta^i` | 4 |

The definitions are

```
Q_kij = partial_k gtilde_ij
X_i   = partial_i chi
a_i   = partial_i alpha
B_i^j = partial_i beta^j
h_perp = alpha H_perp
h^i    = alpha^2 H^i.
```

`gtilde`, `chi`, `alpha`, and `beta` have conformal weight zero.  `Atilde`,
`K`, `Lambda`, and `pi` have conformal weight one.

## Tensor support

`src/athena_tensor.hpp` retains the existing three-dimensional tensor indexing
while supporting `AthenaPointTensor<...,4,...>`.  A symmetric 4 by 4 tensor has
10 stored degrees of freedom.  `MixedTensor<Real,3,4>` stores the 30 independent
components of a spatial index times a symmetric spacetime-index pair and
provides `operator()(i,a,b)` without dynamic allocation.

## Conformal geometry and constraints

`ComputeGeometry` in `src/fo_gh/fo_gh_geometry.hpp` implements

```
gtilde_Gamma^i_jk = 1/2 gtilde^{il}
                    (Q_jlk + Q_klj - Q_ljk)

c^i = -Lambda^i + gtilde^{jk} gtilde_Gamma^i_jk
c_i = gtilde_ij c^j
C_perp = pi + K
C^i = chi c^i.
```

The reduction constraints are

```
Cgamma_kij = Q_kij - D_k gtilde_ij
Cchi_i     = X_i   - D_i chi
Calpha_i   = a_i   - D_i alpha
Cbeta_i^j  = B_i^j - D_i beta^j.
```

The conformal Ricci tensor is evaluated from first-order fields:

```
Rtilde_ij = -1/2 gtilde^{kl} partial_k Q_lij
  + gtilde^{kl} [
      Gamma^m_kl Gamma_(ij)m
    + 2 Gamma^m_k(i Gamma_j)ml
    + Gamma^m_ik Gamma_mjl]
  + gtilde_k(i partial_j) Lambda^k.
```

The Hessians are

```
Dtilde_i Dtilde_j chi   = partial_i X_j - Gamma^k_ij X_k
Dtilde_i Dtilde_j alpha = partial_i a_j - Gamma^k_ij a_k.
```

The vacuum ADM diagnostics are

```
H = 2 K^2/3 - Atilde_ij Atilde^ij + chi Rtilde
    + 2 Dtilde^2 chi - 5 gtilde^{ij} X_i X_j/(2 chi)

M_i = Dtilde_j Atilde^j_i - 2 Dtilde_i K/3
      - 3 Atilde^j_i X_j/(2 chi).
```

## Evolution equations

`ComputePrimaryRhs` in `src/fo_gh/fo_gh_rhs.hpp` first constructs the complete
coordinate-time right-hand sides for the primary fields.

Weight-zero equations:

```
D0 gtilde_ij = -2 alpha Atilde_ij
  + gtilde_ik B_j^k + gtilde_jk B_i^k
  - 2 gtilde_ij B_k^k/3

D0 chi = 2 chi (alpha K - B_k^k)/3
D0 alpha = alpha^2 pi - alpha h_perp

D0 beta^i = h^i + alpha^2 chi Lambda^i
  + alpha^2 gtilde^{ij} X_j/2
  - alpha chi gtilde^{ij} a_j.
```

Weight-one equations use runtime `kappa > 0`:

```
D0 K = alpha Atilde_ij Atilde^ij + alpha K^2/3
  - chi Dtilde^2 alpha + gtilde^{ij} X_i a_j/2
  + alpha [H - K C_perp - chi Dtilde_i c^i + c^i X_i/2]
  - 3 alpha kappa C_perp/2.
```

For `Atilde`, the code forms

```
Rperp_A_ij = [
    alpha chi Rtilde_ij
  + alpha Dtilde_i Dtilde_j chi/2
  - alpha X_i X_j/(4 chi)
  - chi Dtilde_i Dtilde_j alpha
  - Dtilde_(i alpha X_j)]^TF
  - 2 Atilde_ij B_k^k/3
  - 2 alpha Atilde_ik Atilde^k_j
  + alpha K Atilde_ij
  - alpha C_perp Atilde_ij
  + alpha[-c_(i X_j) - chi c_k Gamma^k_(ij)]^TF

D0 Atilde_ij = Rperp_A_ij
  + Atilde_ik B_j^k + Atilde_jk B_i^k.
```

The contracted connection and lapse momentum equations are

```
Rperp_Lambda^i = gtilde^{kl} partial_k B_l^i
  + 2 Lambda^i B_k^k/3 + Dtilde^i(B_k^k)/3
  - 2 Atilde^{ik} a_k
  + 2 alpha Atilde^{kl} Gamma^i_kl
  - 3 alpha Atilde^{ik} X_k/chi
  - 4 alpha Dtilde^i K/3
  + alpha Dtilde^i C_perp
  + 2 alpha K c^i/3 + kappa alpha c^i

D0 Lambda^i = Rperp_Lambda^i - Lambda^k B_k^i

D0 pi = -alpha Atilde_ij Atilde^ij - alpha K^2/3
  + chi Dtilde^2 alpha - gtilde^{ij} X_i a_j/2
  + chi c^i a_i - kappa alpha C_perp/2.
```

## Compatible first-order gradients

`FoGh::CalcRHS` uses two passes.  The first pass computes

```
R_gtilde = beta^k Q_kij + D0 gtilde_ij
R_chi    = beta^k X_k   + D0 chi
R_alpha  = beta^k a_k   + D0 alpha
R_beta   = beta^k B_k^i + D0 beta^i.
```

The second pass uses the same `Dx<FDNG>` operator as the reduction-constraint
diagnostic:

```
partial_t Q_kij = D_k R_gtilde_ij
partial_t X_i   = D_i R_chi
partial_t a_i   = D_i R_alpha
partial_t B_i^j = D_i R_beta^j.
```

The supported centered finite-difference orders are 2, 4, and 6.  Because the
two passes require a doubled stencil radius, the mesh must supply at least
`fd_order` ghost cells.  Sixth order fails closed on multilevel meshes because
the current AthenaK prolongator is not sixth-order compatible.  KO strength is
runtime parameter `fo_gh/diss`.

## Independent moving-puncture gauge driver

The regular targets are

```
f_perp = alpha pi + 2 K

f^i = (3/4 - alpha^2 chi) Lambda^i
      - alpha^2 gtilde^{ij} X_j/2
      + alpha chi gtilde^{ij} a_j
      - eta_beta beta^i.
```

They are targets rather than algebraic substitutions.  The evolved driver is

```
partial_t h_A = beta^k partial_k h_A - mu_H(h_A-f_A) + vartheta_A
partial_t vartheta_A = -eta_H beta^k partial_k h_A - eta_H vartheta_A.
```

When `h=f`, the weight-zero equations reduce to advective 1+log slicing and the
Gamma-driver target.  `mu_H`, `eta_H`, and `eta_beta` are runtime parameters.

## Standard-GH and ADM adapters

`RegularToStandardGh` reconstructs

```
gamma_ij = gtilde_ij/chi
K_ij = (Atilde_ij + gtilde_ij K/3)/chi
rho^i = chi Lambda^i + gtilde^{ij} X_j/2
```

and then the standard spacetime metric, `Phi_iab = partial_i g_ab`, and
`Pi_ab = -D0 g_ab/alpha`.  `FoGhToADM` supplies `alpha`, `beta`, `gamma`, and
`K_ij` to AthenaK diagnostics without depending on the Z4c state vector.

## Identical Z4c one-puncture data

The `fo_gh_puncture` problem generator uses a cell-centered puncture between
cells and initializes exactly

```
psi = 1 + M/(2r)
alpha = psi^-2
chi = psi^-4
beta = 0
gtilde_ij = delta_ij
Atilde_ij = K = pi = Lambda^i = 0
Q_kij = B_i^j = 0
a_i = M psi^-3 x_i/r^3
X_i = 2 M psi^-5 x_i/r^3
h_A = vartheta_A = 0.
```

No analytic expression is evaluated at `r=0`.

## Diagnostic-only puncture mask

FO-GH evolution is unexcised.  Constraint history and checkpoint reductions
exclude cells with `alpha < fo_gh/excise_lapse`; the default puncture input uses
`excise_lapse=0.25`.  The excluded cells contribute neither a numerator nor the
proper-volume normalization.  This is the lapse equivalent of the Z4c history
mask `chi >= 0.0625`, because the identical initial data satisfy
`chi=alpha^2`.  The mask does not modify the state or RHS.
