# Manufactured angular time maps and slicing admissibility

The manufactured Gaussian map supplies an independent flat-spacetime gauge
benchmark. Its finite checks found a genuine future slicing failure for one
strong disturbance with the original `a=1/2` height. Changing the private
control to `a=2` removes all failures in the physical-event sample. Analytic
estimates also prove its exterior and late-time tails. The remaining compact
region is unproved, and no evolution stability follows from these results.

The later black-hole target remains a resolved inner wormhole-to-trumpet
transition with the **Minkowski hyperboloidal reference retained throughout**.
These flat manufactured fields neither solve the proposed inner modified
Bona–Massó gauge nor validate that black-hole transition.

## Exact map and two separate admissibility conditions

In physical Minkowski coordinates `(T,X)`, let `R=|X|` and

```
f(T) = sigma^4 exp[-T^2/(2 sigma^2)],
G(T,R) = [f(T-R)-f(T+R)]/R,
F(T,X) = partial_X1 partial_X2 G,
Y0 = T + epsilon F,   Yi = Xi.
```

The origin is defined by the smooth limit. Each `Y` satisfies the physical
Minkowski wave equation. With `p=n1*n2`, write `F=R^2 p C`, where

```
C = -(1/8) integral_-1^1 (1-z^2)^2 f^(5)(T+Rz) dz.
```

This integral and its regular series avoid subtraction of nearly equal
retarded and advanced radial expressions near the origin. The sphere
representation gives a global Jacobian bound
`J=1+epsilon F_T >= 1-4|epsilon|/pi >0` for `|epsilon|<=3/4`.
The bounded displacement then gives a unique global time inverse. This
Jacobian statement does not guarantee that the resulting slices are spacelike.

For the prescribed reference height `H`, the native target obeys
`Y0=t_native+H(R)`. Define

```
w = grad H - epsilon grad F,
D = J^2 - |w|^2.
```

A spacelike native slice requires `D>0`. Its physical lapse is `1/sqrt(D)`.
The code reports negative `D` directly; it does not floor it or take its square
root. Because `F` is odd in `T`, the global monotone time map preserves its
sign. With `H(0)=0` and nonnegative height slope, future native times therefore
have `T>=0`.

## Completed finite checks

All screens use `sigma in {7/20,1/2}` and
`epsilon in {0,1/4,1/2,3/4}`. Independent readbacks inspect saved values;
they do not rerun a root solver or manufacture new oracle targets.

| Check | Completed result | Runtime |
| --- | --- | --- |
| Physical-event screen, `a=1/2`, 80/110 digits | 56,064 records; identity and precision gates pass; some `D<0` | 13.4028 s |
| Native-time inverse/value screen, `a=1/2`, four precision/height levels | 54,432 records; inverse and value consistency gates pass; one profile has `D<0` | 125.7124 s |
| Physical-event control, `a=2`, unchanged screen source and grids | 56,064 records; identity/precision gates pass and all sampled `D>0` | 13.4411 s |

The original physical-event failures at `T=0` correspond to negative native
times when `H>0`; they alone were not future-native counterexamples. The
subsequent native inverse screen resolves that distinction. For
`sigma=1/2, epsilon=3/4`, it finds five negative events at each of four
precision/height levels. The worst sampled event is

```
r_native = 0.75, t_native = 0.1, p = 1/2,
R = 1.63316748444..., T = 1.0108913263...,
D/D_reference = -0.25785359294198...,
J = 0.94964250243058... >0.
```

Thus a smooth invertible wave map can still fail the spacelike-slice condition
at a future native time. The other sampled profiles remain positive. For
`sigma=7/20, epsilon=3/4`, the minimum sampled native ratio is
`0.16161994896724...`; for `sigma=1/2, epsilon=1/2`, it is
`0.15328765981288...`. Neither finite minimum proves positivity between samples.

The native screen uses 21 compact radii, nine native times and nine angular
parameters. Its four levels combine 80/110 decimal digits with 128/256-point
height quadrature. It solves a bounded retarded variable near scri and checks
the original map equation independently. The largest saved original-map
residual is `1.9306e-56`; the largest direct/factored identity difference is
`6.0958e-45`. The bracket signs and quadrature are high-precision numerical
checks, not formal interval enclosures. It supplies ADM values, not the
curvature and connection jets needed for a full 22-field RHS test.

The `a=2` physical screen changes only the recipe's height parameter; its
scientific source is byte-identical to the accepted general-`a` screen. Its
smallest sampled `D/D_reference` is `0.09256287306414186...`, at
`sigma=1/2, epsilon=3/4, R/sigma=1.5, T=0, p=-1/2`, with
`D=0.08115101200143944...` and `J=0.45214896133278...`.
It is not an `a=2` native inverse screen.

## Analytic regional bounds for the private `a=2` control

For the pure CMC endpoint, `h=H_R=R/sqrt(R^2+4)`. Write `z=1/R`,
`tau=|grad_S p|^2`, and use the exact bounded radial profiles

```
F = p z Phi,
F_T+F_R = p z^2 Aplus,
F_T-F_R = p z Aminus,
eta = (1-h)/z^2,
R^2 D = (eta+epsilon p Aplus)(1+h+epsilon p z Aminus)
        -epsilon^2 z^2 Phi^2 tau.
```

For `R>=4, T>=0, 0<sigma<=1/2, |epsilon|<=3/4`, explicit Gaussian tail
bounds retain the advanced `R f'''(T+R)` contribution and give

```
|Phi|<3/8, |Aplus|<1/2, |Aminus|<37/8,
R^2 D_CMC > 2+5965/49152 >2.
```

On `R<=4, T>=4+8sigma`, the sphere formula gives
`J-|w|>1/10-(3/4)2^-19>0`. The earlier core proof for `a=1/2` gives
`D>126/616225` on `R<=1/50`. Positivity transfers to the smaller `a=2`
height slope by the height concavity argument below. The remaining region is

```
1/50 <= R <= 4,   0 <= T <= 4+8sigma.
```

That compact box has not been certified. A separate future interval algorithm
must prove enclosure and complete coverage; agreement of sampled positive
minima is insufficient.

For the actual layer, its slope is
`h_layer=w_cut R/sqrt(a^2+w_cut^2 R^2)`, in `[0,h_CMC]`.
The global sphere bound gives Cauchy-endpoint positivity at `h=0`.
At fixed event, `D(h)=J^2-|h n-epsilon grad F|^2` is concave, so positive
Cauchy and CMC endpoints imply positive actual-layer `D`. This transfers
positivity, not the quantitative CMC exterior bound unchanged.

## Reproducible evidence and remaining work

[The frozen capsule](validation/hyperboloidal-manufactured-Gaussian-slicing-20261009/README.md)
contains 200 files, 1,694,736 bytes and 120 finite JSON files. Catalog SHA256:
`1d7bdd231b1c696a5c535aec9b7a313fd48ffdd942047120b79a88443b8a0031`.
All three large sample payloads are metadata only, with exact sizes and streamed
SHA256 hashes. Sources and native logs retain their exact bytes. The capsule
also preserves the initial Python import-guard rejection, the first saved
readback's overly strict printed-decimal comparison, and their fresh corrected
attempts. The scientific tolerances were not relaxed.

A third-derivative embedding and full 22-field consistency plan is held in
the capsule. It has not been implemented or executed. Near-scri scaled gates,
binary64 target-coordinate representability and independently derived
curvature/connection jets remain required. The original native angular pulse's
larger derivative gate is separate work and was still running at this checkpoint.

Production `src/` and root `CMakeLists.txt` remain unchanged from
`27c19d20696ea6dd4704032c51dfd026218f64f2`. These private controls add no
production gauge and establish no stable finite disturbance, black-hole run,
puncture regularity, or discrete boundary closure.

## Subsequent oracle and interval checkpoint

The [2026-10-10 handoff](hyperboloidal-assessment-handoff-20261010.md) supersedes
the earlier implementation status of the held third-derivative plan. The v3
oracle is implemented privately, with 318 units and a 20-record timing stage
passing. Its full 5,010-record geometric gate and native RHS binding never ran.
The v2 timing's six connection failures remain preserved. The interval v6
producer reached its 600-second cap; v7 has source review only, with proposed
237 units, producer and replay unexecuted. The remaining compact-region
positivity certificate and evolution stability are still open.
