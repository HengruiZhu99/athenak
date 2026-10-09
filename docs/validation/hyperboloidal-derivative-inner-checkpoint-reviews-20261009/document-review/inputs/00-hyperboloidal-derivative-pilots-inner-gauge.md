# Derivative pilots and the coupled inner gauge

This checkpoint records two completed flat-space derivative pilots and a
candidate inner gauge. It does not repair the failed native wave-map runs in
[the final matrix report](hyperboloidal-reference-wave-map-final-matrix.md).
The later black-hole acceptance target is explicitly a resolved inner
wormhole-to-trumpet transition with the **Minkowski hyperboloidal reference
retained throughout**. A black-hole reference or RHS subtraction is not part
of that target.

## Completed derivative pilots

The fixed-Lorentz-frame Kirchhoff calculation evolves the actual native
lapse/shift pulse as four scalar coordinate displacements on physical
Minkowski spacetime. Analytic differentiation supplies four values, sixteen
first derivatives and forty symmetric second derivatives at each ray. The
ray direction remains fixed during differentiation. These are scalar wave
checks, not an independent evolution of the nonlinear Z4c equations.

The old pilot and compact-radius comparison used the same 480 rays in 24
groups, at 60 and 80 decimal digits, with unchanged pulse, quadrature and
derivative formulas. The sample includes an initial origin point, a short-time
transition point and an outer-collar point, with two fixed boosts.

| Completed check | Result | Child runtime |
| --- | --- | --- |
| Original ray solver | 4,744 checks passed | 193.8207 s |
| Compact-radius solver and saved-jet comparison | 38,344 checks passed | 18.9809 s |
| Independent saved-data readback | All 480 ray keys and 28,800 jet components matched within the declared bounds | No oracle rerun |

The independent readback used 120-digit `Decimal` arithmetic. The largest
scaled old/new jet difference was `3.8369807166e-50`; the largest scaled metric
difference was `1.065114715834e-50`. The largest original ray-equation residual
was `5.6496846024350477e-52`. The compact pilot used 160 exact initial roots,
160 safeguarded Newton roots and 160 analytic outer roots; no fallback was
needed on that sample. Original source and saved-output hashes were unchanged.

For the new root variable, let `q=r/Omega`,
`lambda=(T-H(r))/k0`, `y=X-lambda*k_spatial` and `F=q-|y|`. Then

```
F_r = (L/Omega^2) K/k0 > 0,
K   = k0 - (b/A) nu.k_spatial.
```

The solver retains endpoint signs, future-ray checks, the actual compact-radius
and transformed-lambda widths, and the original ray residual. It uses a fixed
16-probe safeguarded Newton stage followed, if needed, by the unchanged fixed
512-step bisection cap. Height quadrature and endpoint signs are numerical;
they are not interval enclosures. Zero width in an analytic branch records an
exact formula branch, not a validated interval certificate.

These finite pilots neither pass the much larger 493,568-root derivative gate
nor establish angular quadrature convergence, global absence of caustics, or
coverage at native coordinate time. The larger gate was still running when
this checkpoint was prepared. A later four-dimensional inverse must solve
`X+u(X)=Y_reference(t_native,x_native)` and separately check the Jacobian,
future time orientation, coverage and injectivity. Reference time is not
automatically native target time.

The completed evidence is in
[the compact capsule](validation/hyperboloidal-derivative-pilots-inner-pencil-20261009/README.md).
Its catalog SHA256 is
`79b850b690bb21b17898506b932370e1f48fa630b0023e0760de64243c4a1b5f`.
It contains 287 files, 6,100,522 bytes and 152 finite JSON files. All seven
NPZ/NPY/JSONL or oversized payload omissions and 315 external dependencies
remain identified by exact hashes. Source and log whitespace is preserved.

## Coupled inner principal family

The earlier weighted core connection driver weakens as `alpha^2 chi` collapses.
Increasing that coefficient alone crosses defective scalar speed coincidences.
The new candidate changes its metric-gradient coupling jointly. In the actual
constrained 20-field symbol, use

```
epsilon_alpha = 1,
epsilon_chi   = 2 mu^2/(1+mu)^2,
q            = (4 mu-2 epsilon_chi)/3,
C            = 2(1+mu)^2/(4 mu^2+5 mu+3).
```

For finite `f>0, mu>0`, define `H=h+2cchi`, `V=Lambda+2cchi` and
`X=cchi-C V`. The scalar block becomes four independent wave pairs with
speed squares `f,q,1,1`. The finite identity
`C(q-1)=2(mu-1)/3` cancels the scalar light-speed coincidence without dividing
by `q-1`; `epsilon_alpha=1` removes the lapse forcing at `q=f`. The two vector
blocks have speed squares `1,mu`; the two tensor blocks have speed square `1`.
The explicit transform and inverse are preserved in the capsule's corrected
inner assessment. The scalar storage is `pi=P/Omega`, with
`P=K_phys-2Theta_phys`, not `K_phys/Omega`.

The original displayed `A_t` row accidentally contained `2 beta/3`; its
correct term is `2 Lambda/3`. The erroneous note, one-line erratum, corrected
note and independent review are all retained. Only the corrected note is
authoritative.

Starting from the complete physical-reference wave-map gauge, let
`cW=1-W`, `A0=alpha^2 chi`, `B=cW G0+W A0` and `kappa=B/(A0+B)`. The proposed
reference-deviation additions are

```
Delta alpha_t = -2 cW alpha (P-P_hat)/Omega,
Delta beta_t^i = (B-A0)(Lambda^i-Lambda_hat^i)
  + alpha^2 (2 kappa^2-1/2) gtildeInv^{ij}
      (chi_j-chi chi_hat_j/chi_hat)
  - cW eta_I (beta^i-beta_hat^i).
```

The resulting normalized coefficients are `f=1+2cW/alpha` and `mu=B/A0`.
An implementation can evaluate the bounded `kappa` without forming `mu`.
At `W=1`, every addition vanishes; an exact branch must retain the unchanged
outer wave-map rows. At the Minkowski reference every deviation vanishes,
including the nonzero reference connection in the geometric transition.
That algebraic statement still requires nonlinear implementation and arithmetic
tests on the nonflat reference.

The subsequent private compiled principal gate passed 118 actual-kernel cases
in Release and Address/UndefinedBehavior-sanitized Debug, plus 18 exact rational
scalar cases. Printed matrices were byte-identical between builds. The largest
matrix error was `5.346834086594754e-13`, the largest explicit basis-inverse
error `4.440892098500626e-16`, and the largest sampled infinity-norm basis
condition `113.32241771251452`. The sample includes oblique propagation,
positive-definite nontrivial metrics and `mu=1`, `q=1`, `f=1`, `q=f`
coincidences. This compiled result is separate from the pencil-only capsule;
its exact source/run evidence will be archived separately.

These results establish neither a uniform diagonalizer at a puncture nor
nonlinear regularity or evolution stability. Nonflat reference identities,
collapsed-lapse/high-contrast arithmetic, variable-coefficient sources and the
actual puncture treatment remain separate gates. In a putative radial trumpet,
the constant connection response additionally needs `Lambda=O(r)` or derived
cancellation of stronger connection residues. The unchanged outer wave-map
condition also retains its conditional stationary mass-log obstruction.

Production `src/` and root `CMakeLists.txt` remain byte-identical to
`27c19d20696ea6dd4704032c51dfd026218f64f2`; no candidate is adopted by this
checkpoint. Existing CPU Serial/double, vacuum, uniform single-MeshBlock
restrictions remain in force.
