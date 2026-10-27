# Regular source combinations without a fixed puncture exponent

The evolved fields and the Minkowski-reference gauge are unchanged. This
arithmetic rewrite applies at positive chi and lapse; it supplies neither a
puncture endpoint theorem nor a value for an undefined unit normal at alpha=0.
It assumes no wormhole/trumpet power and never floors chi, alpha or Omega.

Write g for the evolved conformal metric, D for its Levi-Civita derivative,
and bar gamma=g/chi. The actual tensor equations need the following combinations,
which can be evaluated without constructing derivatives of 1/chi:

```
chi Rbar_ij = chi R_ij + (D_i D_j chi + g_ij Delta_g chi)/2
             - chi_i (chi_j/chi)/4
             - 3 g_ij g^ab chi_a (chi_b/chi)/4
Rbar = chi R_g + 2 Delta_g chi - (5/2) g^ab chi_a (chi_b/chi)
chi Dbar_i Dbar_j f = chi D_i D_j f
    + (chi_i f_j + chi_j f_i - g_ij g^ab chi_a f_b)/2
Delta_bar f = chi Delta_g f - g^ab chi_a f_b/2
```

`regular_penrose.hpp` also retains the complete covector derivative/divergence
for independent Z. It does not replace evolved Lambda by the metric connection,
assume det(g)=1, discard trace(A), or impose conformal flatness. The old
`PenroseMetric` path is retained for comparison and other callers. The full RHS
and evolved-constraint diagnostic now use these scaled combinations.

For stationary compactification, store the product

```
N = alpha w = -beta^i Omega_i
partial_j N = -(partial_j beta^i) Omega_i - beta^i Omega_ij.
```

The general helper also accepts explicit coordinate Omega_t and its spatial
derivatives. The raw unit-normal value remains w=N/alpha wherever defined;
cached N and its derivative avoid dividing then multiplying by a tiny lapse
inside linear pole numerators. Nonstationary callers without cached data retain
their explicit alpha*w product. An unrepresentable raw null diagnostic remains
invalid, rather than being fabricated from a finite cached product.

For the already-selected Appendix-C Theta source, `ConformalRHS(...,true)`
omits the analytically cancelled `-3 N Theta` directly. Selected callers no
longer add `+3 N Theta` after rounding the other terms away. The default false
preserves the original full tensor Theta equation. This selects the same
disclosed spherical equation; it does not subtract a background physical RHS.

The frozen legacy function in the test tree is used only for independent
ordinary-range comparisons. Across 384 off-constraint nonflat cases with three
normal-data modes, full RHS/constraints agree below9e-16. Three extreme full-RHS
fixtures and four scaled-geometry fixtures now remain finite where the old
arithmetic failed. A selected-Theta fixture retains -3.07427e-200; the old
post-add arithmetic leaves an erroneous3.47e-18. Sixteen independent radial
scri limits agree to2.31e-15. See `evidence/regular-kernel-20261027/` for actual
commands, failures in the test driver, corrected invocations and raw results.

## Physical volume and stopping tests

The spatial measure is the same Penrose volume quadrature used previously:
`dV_bar=dV_coordinate sqrt(det(g)) chi^(-3/2)`. It is not silently replaced by
coordinate volume or a finite inner cutoff. `log_volume_norm.hpp` retains
the measure in logarithms and forms signed log-sum products for Krylov norms.
A cell whose normalized weight would underflow is still retained when its
residual makes its product significant. Absolute volume/L2 values outside
the floating-point range are disclosed along with their finite logarithms.
Zero chi still requires an independent endpoint measure and is rejected.

Physical-volume RMS alone is insufficient near an asymptotically flat end:
one near-node experiment stopped after3 iterations with RMS8.7e-13 despite
pointwise residual40.56. The default balanced physical/conformal norm plus
an independent maximum residual guard rejects that result. The corrected solve
takes32 iterations with maximum residual1.48e-9. Its unnormalized physical
L2 residual is about713, which remains recorded and precludes a continuum
endpoint-accuracy claim from the tiny RMS. Every active cell remains in the
diagnostics. Bounded column scaling reconstructs physical beta before applying
the original operator and showed no iteration-count benefit in these tests.
