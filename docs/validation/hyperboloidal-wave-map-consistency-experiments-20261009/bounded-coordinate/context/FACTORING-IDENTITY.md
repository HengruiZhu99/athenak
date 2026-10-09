The second Cartesian implementation evaluates the same stationary active-Lie lift in conformal variables. The original physical-variable source, compile receipt, executable and all outputs remain unchanged in `coordinate-lift`; the new source is in `coordinate-lift-factored-002`.

Write `o=Omega`, `c=chi`, `h0=Lie_X(bar_gamma)+bar_beta_i T_j+bar_beta_j T_i`, `psi=X(o)/o`, so `hbar=h0-2 psi bar_gamma`. Define

```
Q0_ij = Lie_X(A)_ij - A_ij X(log(o*c))
       + A_ki beta^k T_j + A_kj beta^k T_i
       - alpha*c Dbar_i Dbar_j T
       - c(alpha_i T_j + alpha_j T_i).
q = (c X(P)/3 + alpha*c bar_gamma^{ij} o_i T_j)/o.
```

The conformal connection transformation gives
`o*c [delta Kcov - P*hphys/3] = Q0 + q*bar_gamma`.
At the analytic reference `bar_gamma^{-1}:A=0`; hence the contraction of the `-2 psi bar_gamma` part of `hbar` with raised `A` vanishes. The equivalent cancellation-safe expressions are

```
delta P = X(P) + 3 alpha bar_gamma^{ij} o_i T_j
          + (o/c) [bar_gamma^{-1}:Q0 - A_raised_bar:h0],
delta A = Q0 - bar_gamma*(tr_bar(Q0)-A_raised_bar:h0)/3
          + (delta chi/c)*A,
delta chi = c*(2 psi - tr_bar(h0)/3),
delta gtilde = c*(h0-bar_gamma*tr_bar(h0)/3).
```

No gauge or geometric equation is changed. The original physical formula is included under a different function name as an independent numerical identity comparator. The predeclared 629-case run compares its complete consumed lift jets with the new formula at all originally declared radii through .95, with a separate 2e-7 identity gate. All cases/radii through .995 are still used for the original source/geometry gates; the identity-comparison range does not remove outer source failures.

The source extracts only complete available orders: configuration through second, P/A/Lambda through first; input background configuration through third, physical curvature through second. Neither implementation differentiates actual momentum-source rows or claims an independent C_ref[F_actual] constraint-rate oracle.
