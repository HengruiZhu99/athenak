# Cartesian physical-constraint radiation data: experimental helper

The implementation is isolated in `athenak-boundary-radiation/src/z4c/z4c_constraint_radiation.hpp`; no main-repository evolution source was changed by this agent. The parent owns the boundary-mode plumbing and evolution tests. The helper is a candidate boundary operator, not a validated stability fix. The first3D v1 pilots failed rapidly, so their outcome does not support a cure.

## Choice of radiation model

Let `g` denote the conformal metric, `chi=psi^-4`, and `Gamma_metric(g)` its contracted Christoffel symbol computed from metric derivatives, including the determinant-gradient term. Define the physical spatial constraint covector

```
Q^i = Gamma_evolved^i - Gamma_metric^i(g)
Z_i = g_ij Q^j / 2
Zres_i = Z_i(full) - Z_i(background)
Thetares = Theta(full) - Theta(background).
```

Identical evaluations of full/background state subtract exactly. This removes the reference discretization defect without resetting small nonzero constraints. On the current conformally flat analytic trumpet, the continuum background Z vanishes.

For the conformal unit outward face normal `n^i`, use

```
c = alpha sqrt(chi)
V^i = -beta^i + c n^i
R = |x| + characteristic_radiation_areal_shift
omega = V^i partial_i(log R).
```

The areal shift is finite and nonnegative, default1M in the parent implementation. For the current trumpet, `R=r+1` is the background areal radius. This is an explicit leading radiation model; it is not an assertion about areal radius for a general spinning background. The chosen ray is face-normal outgoing transport, so it is an approximation for obliquely outgoing waves. The physical-background connection retains the tangential and geometric terms that a universal componentwise `1/r` multiplier omits.

The helper returns

```
FTheta = Thetares_t + V^j partial_j Thetares + omega Thetares
FZi = Zres_i,t + V^j (partial_j Zres_i - Gamma_bar^k_ji Zres_k) + omega Zres_i
FQ^i = 2 g_full^ij FZj.
```

`Gamma_bar` is the spatial Levi-Civita connection of the fixed physical background metric `g_background/chi_background`. Along a radial trumpet ray, its covector transport supplies the derivative of the physical orthonormal basis. Consequently the model reduces to

```
Theta_t + (c-beta_r) [Theta_r + Theta/(r+1)] = 0
q_t + (c-beta_r) [q_r + q/r] = 0
```

for `Q^i=q n^i`. This matches the physically weighted radial model. It is not the old componentwise Sommerfeld overwrite.

## Exact discrete metric time derivative

At a fixed stage, write `S_ij = metric_rhs_ij` and

```
T_jkl = D_j g_kl - D_l g_jk/2
Gamma_metric^i = g^il g^jk T_jkl
(g^ij)_t = -g^ia S_ab g^bj.
```

The helper differentiates this functional exactly:

```
Gamma_metric,t^i = [(g^il)_t g^jk + g^il (g^jk)_t] T_jkl
                  + g^il g^jk [D_j S_kl - D_l S_jk/2].
Z_i,t = S_ij Q_full^j/2
        + g_full,ij [Gamma_rhs^j - Gamma_metric,t^j]/2.
```

The reference state is fixed, so its time derivative is zero. The implementation uses the same linear metric derivative D in the functional and its time derivative. It retains inverse-metric derivatives and the lowering-metric product rule. It does not identify evolved Gamma with Q or neglect metric-defined Gamma_t.

The local volume RHS already contains the selected gauge, geometric, damping and matter terms. Evaluating F from that RHS retains those terms, rather than trying to reconstruct only the principal part. The caller's matter-support restrictions remain necessary.

## Characteristic correction and its scope

For the existing incoming light-speed characteristic amplitudes C1, C2 and C_A, the frozen principal decomposition on a conformal unit face frame gives

```
FTheta = alpha D_n C1 + remaining tangential/coefficient/source terms
FQ_n = -c D_n C2 + remaining terms
FQ_A = -c D_n C_A + remaining terms.
```

The radial coefficient `-chi` assumes `alpha=sqrt(chi)`; the general coefficient is `-alpha sqrt(chi)`.

With the code sign convention `C_t=lambda_in D_n C+...` and `lambda_in=beta_n+c>0`, the differential Bjørhus rate increments are

```
delta C1_t = -lambda_in FTheta / alpha
delta C2_t = +lambda_in FQ_n / c
delta C_A,t = +lambda_in FQ_A / c.
```

These increments must be applied relative to the immutable pre-correction volume characteristic rates. Gauge and two radiation-polarization data remain separate. The parent owns this implementation in `z4c_Sbc.cpp`.

Although the existing p-only solve changes instantaneous outgoing characteristic rates, that fact alone does not make a second-order boundary scheme inconsistent. The independent radial differential-Bjørhus tests show smooth-mode/full-PDE and physical-radiation defects converge. This corrected an overly strong initial concern in my audit. Conversely, a local map's own enforcement residual does not certify continuum compatibility or3D stability. The early v1 failures demonstrate the necessity of full discrete tests.

The physical radiation conditions are not imposed by directly overwriting Theta_t and Gamma_t alongside both incoming gauge rows; that local p-only system is rank deficient. `face_rank_probe.py` separately records that allowing all q_t/p_t degrees of freedom removes this pointwise rank defect, but that exploratory algebra is not the selected implementation and is not a stability result.

## Derivative and ownership rules

All state views are read-only. Only local Gamma_rhs and Theta_rhs plus spatial metric_rhs derivatives are read. The current boundary kernel never changes metric_rhs, so those stencil reads remain immutable without an additional full RHS copy. No neighbor Gamma/Theta RHS is read.

Every derivative uses active cells only, including at internal block edges. RHS ghosts may be uncomputed, and the point tests deliberately poison every ghost input with NaNs. Every nested evaluation selects its own stencil from its own indices; it never reuses the outer point's face mask.

Version1 used D2 for metric-defined Gamma and then D2 for its spatial transport. Its closure truncation error changes between one-sided and centered stencils, so the nested metric term is **only first order** near a boundary. This was caught by the independent radial reviewer and demonstrated in the actual C++ helper.

Version2 uses active-only five-point D4 for metric-defined Gamma and Gamma_t, with consistent biased closures at the first two/last two active points. The outer transport and reference-connection derivative remain D2. Differentiating the inner O(h^4) closure error loses at most one power, so the overall boundary consistency is at least second order. At least5active cells per direction are required. The sixth-order volume operator still uses a different stencil; compatibility of the coupled operator, including block interfaces and corners, remains a test requirement.

## Actual C++ point tests

`point_tests.cpp` includes the actual header and builds against the isolated AthenaK Kokkos configuration. The compiler command is saved in `point-build-command.json`.

- Equal full/background states return zero exactly, including a nonconstant metric.
- Nontrivial metric with evolved Gamma equal to its discrete metric-defined Gamma returns zero Z/Theta radiation data when Z_t/Theta_t are zero. This tests the physical Z functional, not Hamiltonian/momentum constraints or full evolution.
- Manufactured quadratic flat-space constraint data are differentiated to roundoff on faces, edges and corners: maximum error1.25e-16.
- Independent finite differencing of the full nonlinear Z functional, using a nonconstant non-unit-determinant metric and nonzero metric RHS, agrees with its analytic time derivative to1.45e-10.
- All input and RHS ghosts are NaN; all15evaluations return success.

`cubic_convergence.cpp` uses `g_xx=1+0.05x^3`, Gamma_evolved=0, an otherwise flat reference and an analytic physical Z/radiation residual at x=2. Actual C++ errors are:

| h | v1 error | v2 error |
|---:|---:|---:|
|.125|.00589081|.000357807|
|.0625|.00299860|.0000794238|
|.03125|.00150889|.0000185847|
|.015625|.000756366|.00000448687|

Observed orders approach1 for v1 and2 for v2. The v1 source is preserved as `constraint-radiation-v1.hpp`; raw results are `cubic-convergence-v1.json`, `cubic-convergence-v2.json` and `point-tests-v2.json`.

These tests establish the helper's algebra, differential accuracy and input ownership. They do not validate the full characteristic closure, ghost extrapolation,3D spectrum, MPI partition independence, GPU execution or nonlinear/matter stability. No Aurora jobs or production changes were made by this agent.
