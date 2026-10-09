# Saved-mode discrete constraint closure

The growing C0 approximate direction is not described by a separately
discretized coefficient-aware continuum constraint subsidiary, even on
points whose complete nested native stencils avoid ghost reads. The defect
persists at N20. This measures a substantial finite-grid closure defect on the
sampled global directions; it does not identify a unique bulk or boundary
cause, an autonomous interior instability or a continuum unstable mode.

## Operators and norms

Both grids use S=1,a=.5, reference transition(.05,.95), κinput=10,κ2=0,
the physical-P/spatial-norm gauge, native centered/mixed/upwind derivatives,
KO=.1 and symmetric quadratic primitive continuation. J_h is the continuous
projected generator `P_ref J22 Lift`, distinct from the native final-only
SSPRK3 step. Let

```
c = C_h v,
d = C_h(J_h v),
E_c = d−K_c c.
```

C_h is the directional derivative of the actual native constraint functional.
K_c independently discretizes the frozen continuum Subsidiary() using all
analytic reference coefficient gradients and native centered derivatives.
It acts on signed physical H, M covector, Z covector and physical Theta, in
that order. Theta is appended through the pinned primitive lift and is not
divided by Ω. M/Z group RMS contracts with the reference Penrose inverse;
H/Theta use unweighted nodal complex RMS. These are component norms, not
physical energies. K_c is a comparator, not the constraint closure induced
by the native primitive equations or ghosts.

The calculation uses actual J_hv rather than λv. The N16 direction has
λ≈1.916338997+6.790995421i and generator residual 5.887e−7. The fresh N20
export exactly reproduces the previously checked rank32 result
λ≈1.918280286+6.805508834i and residual 3.824e−7. A rank16 control agrees
after phase alignment to 6.46e−8. Only those small, previously specified
reduced eigendecompositions are replayed, with warning-free contractions;
there is no new global eigensolve, matrix, propagation or native evolution.
The vectors remain approximate/pseudospectral directions without eigenvalue
error bounds or rightmost-spectrum certification.

## Exact support and observed defect

The full nested mask explicitly enumerates every centered axis/mixed stencil
offset composed with the primitive centered/upwind/KO support. A radial
collar estimate would miss mixed compositions and is not used.

| Grid | Full nested active points | Radius range | Mode H²/M²/Z²/Theta² fractions |
| --- | ---: | --- | --- |
| N16, span2.2 | 32/1640 | .119078–.228018 | .163710 / .007960 / .000136 / .079199 |
| N20, span2.2 | 184/3112 | .095263–.392779 | .313221 / .030432 / .001437 / .104521 |

Even the N20 mask covers only 3.04% of M² and .144% of Z². These clean
interior tests therefore cannot clear the outer closure. The mode itself
is a global vector whose interior shape can depend on the outer treatment.

On each grid's complete nested mask, centered defect/actual ratios are:

| Grid | H | M | Z | Theta |
| --- | ---: | ---: | ---: | ---: |
| N16 | .624812 | .853364 | 1.607605 | 1.445123 |
| N20 | .385943 | 1.974894 | .990937 | 2.300188 |

For N16 the actual d RMS is .390295/.0979774/.00477680/.00725605, and
the centered defect RMS is .243861/.0836104/.00767921/.0104859. For N20
these are .247266/.0709466/.00487520/.00343412 and
.0954306/.140112/.00483102/.00789912. Relative mismatch is substantial;
the absolute norms and limited mode fractions remain essential context.

A common physical ball r≤.228017954 contains 32 admitted points in each
grid. It samples different radii: N16 .119078–.228018 and N20
.095263–.182414. Its centered defect/actual ratios are:

| Grid in common ball | H | M | Z | Theta |
| --- | ---: | ---: | ---: | ---: |
| N16 | .624812 | .853364 | 1.607605 | 1.445123 |
| N20 | .158906 | 1.829937 | 2.643117 | 1.519239 |

H mismatch becomes smaller while M/Z do not. The global directions differ,
as do the sampled points and closest-shell phase: Ωmin is .0026953125
at N16 and .022925 at default N20. No order or uniform refinement trend
is inferred. Separately discretizing a continuum identity need not give
exact finite-h closure. These two nonmatched modes and grids establish
neither an anomalous truncation rate nor that the defect causes growth.
The existing controlled-phase N20 history is not propagated further in
this audit.

## Transport, KO and outer continuation

Native upwind and KO terms are separated rather than assumed to commute
through C_h. Write U for the projected primitive `Lx−βDx` correction and Q
for projected primitive KO, with corresponding componentwise constraint
operators U_c,Q_c. The receipts separately measure

```
C_h(J_h−U−Q)v−K_c c,
C_h Uv−U_c c,
C_h Qv−Q_c c.
```

Matching the constraint-side transport and KO leaves nested-mask
defect/actual ratios .670352/.870750/1.602048/1.450405 at N16 and
.498149/2.046781/1.082135/2.360349 at N20. The mismatch therefore remains
after this explicit correction. This vector does not separate all Hessian,
product-rule, diagnostic and algebraic-projection effects uniquely.

A separately named outer comparator applies the primitive same-ray plan
componentwise to the eight constraints. Its donors are strictly active and
nonrecursive, and constant/quadratic bridge oracles pass. This chosen
constraint extension is not the extension induced by continuing primitives
and then constructing C_h. It yields large all-grid centered M ratios,
1093.72 at N16 and 97.52 at N20. Those numbers describe an extension-dependent
comparator mismatch; they are not evidence that native primitive ghosts
are unstable. Strict and extended rows agree bitwise on shared support.

## Verification and remaining work

Actual native RHS checks agree with cached J_hv to 6.75e−8 and 3.59e−8
in state-generator units. Constraint callbacks, their amplitude sensitivity
and derivative bridges are checked independently. The measured
`C_h(J_hv−λv)/C_hJ_hv` is at most 1.52e−7 at N16 and 8.07e−7 at N20,
far below the strict mismatch. Analytical constant/quadratic subsidiary
oracles agree within 5.82e−16 after scale normalization. These are bridge
and finite-difference checks, not exact discrete Bianchi closure.

The N20 callback differs from the scrutinized N16 source only in grid
construction. All 1082 compiler dependencies per callback, libraries,
executables, original histories and prior sources are pinned and rechecked.
Production headers match implementation
`27c19d20696ea6dd4704032c51dfd026218f64f2`. Scratch preparation/compile
failures and the first completed sensitivity experiment remain preserved.

N16 frozen index:
`0ccc0eb70207cfdc7ba14d7da156063902c0850119690215f65bc0c8b3c3321b`.
N20 frozen index:
`278f2f444719828ae81ef481ce60b6b6c0f7780bc4b79a3cd49d7f9d163514bd`.

The result motivates a separate spherical-angular tensor control of the
actual continuum kernel. Such a control must retain nonradial modes,
Cartesian tensor components and regular-origin conditions; the existing
eight-field symmetry-reduced driver is insufficient. No new solver,
nonlinear scri closure, stable finite pulse or black-hole transition is
accepted here. The later black-hole target remains the inner
wormhole-to-trumpet transition with a Minkowski hyperboloidal reference.

The byte-preserved [compact archive](validation/hyperboloidal-mode-subsidiary-closure-experiments-20261009/README.md)
contains 71 cataloged blobs totaling 1,778,500 bytes, including 28 finite JSON
files. Its catalog SHA256 is
`6501b7fe994bc22e1ab1eff3cbe3525fce5a1474fded411cf5097d93cb84223c`.
Binaries, objects and large arrays remain metadata-only in the original
frozen indices. No production source is changed by this evidence stage.
