# Factored Q lapse and null-residue feedback

This private gauge candidate fails the Cartesian stability screen. Its local
source, principal-symbol and scri-pole checks pass, and a short native reference
run stays stationary to roundoff. The angular pulse nevertheless has worse
short-time constraints, followed by much larger growth in the projected global
linear system. The candidate is rejected. Production remains implementation
`27c19d20696ea6dd4704032c51dfd026218f64f2`.

## Equations and experimental scope

The candidate preserves physical `P=Kphys−2Θphys` storage and the complete C0
geometric evolution. Only gauge sources change. The experiment uses `S=1`,
`a=.5`, `κinput=10`, `κ2=0`, geometry radii `(.05,.95)`, gauge
`W=SmoothCutoff(.45,.85)`, `q0=.5` and `ξ=2`. The original finite ν/η functions
and geometric damping `κ1=κinput/α` remain in place.

Put `h=αref`, `B=β·dΩ`, `Bh=βref·dΩ`, `D=α Bh/h−B` and
`F=α²+2(1−W)α`. Define two lapse sources:

```
Rα,Q = β·[dα−(α/h)dh]−αν log(α/h),
Sα,Q = −F(P−Pref)+3[α+2(1−W)]D;
Rα,P = β·dα−βref·dh−αν log(α/h),
Sα,P = −F(P−Pref)−Wξ(α+h)(α−h)
       −W[α(β−βref)+βref(α−h)]·dΩ.
```

The tested `physical_inner=true` variant takes
`Rα=(1−W)Rα,P+W Rα,Q`, `Sα=(1−W)Sα,P+W Sα,Q`, then
`αt=Rα+Sα/Ω`. It recovers the physical-P lapse exactly at `W=0` and the
harmonic Q source exactly at `W=1`. The unused Q branch is skipped at `W=0`.
A separately named global-Q variant is also checked locally; it is not
interchanged with this blend.

The regular shift uses the original preferred-source gauge extension. The
additional beta pole is

```
v = SmoothCutoff(r;.85,.95),  σ=5,
V^i = Ω_i/(δjk Ω_j Ω_k),
G = χ gtildeinv(dΩ,dΩ),
Nraw = G−(β·dΩ/α)²,
ΔSβ^i = v σ V^i α²(Nraw−Nraw_ref).
```

Feedback starts at the harmonic collar and is fully on by `.95`. V uses the
nonzero Euclidean gradient norm. The cancellation-free difference is

```
δG = [(χ−χref)ginvref+χ(ginv−ginvref)](dΩ,dΩ),
α²δNraw = α²δG+D(B+α Bh/h).
```

D uses deviations near reference and its direct form for collapsed lapse.
The implementation does not form `Q=(P−3wn)/Ω` or a live-lapse quotient in
the Q numerator. It also factors the preferred projection and
`α²dlog(α/h)=αdα−α²dlog(h)`. This relative-log identity corrects a prose typo
in the frozen derivation; the frozen code already implements it correctly.
The original gauge assembler ignores beta poles, so the private wrapper adds
`Sβ/Ω` exactly once.

At `W=1` the actual off-constraint four-dimensional source satisfies

```
Γ4^0+2Z4^0 = F0 = [β·dlog(h)+νlog(α/h)]/α²−Kbar_ref/α,
Ω_i F^i = g4ij Ω_ij−Ω What−vσ δNraw/Ω,
Box(Ω) = Ω What+2Z4^iΩ_i+vσ δNraw/Ω.
```

F0 is bounded in a smooth Ω limit with positive nonzero lapse. This supplies
no uniform collapsed-lapse bound. The feedback contribution is O(Ω) if
`δNraw=O(Ω²)`, a falloff not proved preserved here. The lapse blend changes
the actual temporal and spatial sources through the nonharmonic transition;
the displayed preferred Box identity is restricted to `W=1`.

## Local checks and the compatible pole

The core contains 14 successful commands and 382 unchanged inputs: 365
production files and 17 scratch files. Checks cover 640 reference/blend rows
for both `ξ=1.5` and `ξ=1/a`, 160 independent four-dimensional source rows,
48 tiny-positive core cases, 320 noncore cases, 504 complete principal matrices
and 16 matched actual full20 reference poles. Release, Debug and sanitizer
JSON outputs agree byte-for-byte. The principal system retains the canceled
scalar/vector/tensor basis of the original blend. This does not establish
uniform hyperbolicity at a puncture.

Let `z=a²λ`, `K=κinput a²` and use normal coordinates
`C=a²δNraw`, `T=aδ(P−3wn)`, `E=aδΘ`. The pole block has polynomial

```
z³+2(K+σ)z²+[4σ(K+1)−9]z+(8K−6)σ−12K.
```

For `K>0, σ≥0`, it is Hurwitz precisely when `K>3/4` and
`σ>6K/(4K−3)`. At `σ=5`, this requires `K>15/14`. The four sampled values
`a=.5,.75,1,2` and `κinput=5,10` satisfy it. Exact reconstruction of these
sampled sigma-five reference matrices has `rank M=rank M²=11`: nine
semisimple zeros and eleven modes with negative real part. The compatible
zero directions obey linked null/Q/Θ conditions. They do not supply
independent boundary lapse and shift values or full first-jet closure.

The initially Einstein-constraint-free linear witness `δα=Ω, δβ=−Ωn` now
has `αt0=1/a²`, `βn,t0=0`, `Pt0=3/a²` and
`Nraw_t0=(P−3wn)_t0=0`. The earlier physical-P lapse with its fully rederived
preferred projection and null feedback also cancels this witness corner,
while failing its native screen. The witness is a prerequisite rather than
a unique stabilization result. The Q candidate changes gauge dynamics and
adds one compatible zero direction.

The core index is
`dac759becbe666c71443328b61324a50aeaf0f72e76b4e751a9cabfa3a87cd96`;
the independent math/source review is
`a456408b97c35ddf34ffe0af9c52411a13d990c9fa95f83a987234de24a3b650`.
All original sources and failures remain byte-preserved.

## Native integrity and finite-radius growth

A separate private build recompiles six Cartesian-dependent translation
units in 9.7001 seconds. It preserves 182 original objects, four libraries,
369 base catalog entries and four old overlays. The base entries comprise
365 production files plus four auxiliary files. The build replaces the old
spatial-norm forced include, sets explicit gauge flags and `ξ=2`, and adds
the beta pole once. Its 268 dependencies and three private headers are
pinned. The loader, ng3 ghosts, spatial stencils, KO and final-only projection
remain unchanged.

| Native N24, span 2.1 check | Result |
| --- | --- |
| Reference to t=.05 | Exit 0, 16.5406 s; maximum 25-field binary64 drift 1.30128e−14 |
| Angular pulse to t=.02 | Exit 0, 6.93474 s; H/M/Z = .00322665038/.00690835275/.00131840943 |
| Pulse versus original spatial-norm gauge | H/M/Z ratios 1.03730/1.41158/1.10051; Θ ratio 1.39099 |
| Six saved states | All 25 RST fields and BIN casts finite; initial fields, geometry, determinant/trace and spatial SPD checks pass |

A values-only utility verifies manual/factored null differences to 3.19e−15,
beta pole increments to 2.33e−14 and single assembly to 2.78e−17. It holds
reference derivative jets equal for sigma-five and sigma-zero to isolate
this algebraic increment; it does not validate full live derivative sources.
At t=.02 sampled outer maxima are `δNraw=.00102106`, Q-deviation numerator
`.00207898` and feedback beta RHS `.174425`. These are not future bounds.
The first utility compile failure, repaired only by namespace qualification,
is preserved.

The executable SHA is
`5d7ce0c747ef0a8519deed61d1d2cbdbba71d8924728127bbbd4f8b1aa39e43e`;
build receipt
`8f7e60ef5d5df9dad31f8feba359b244b36859aa115d525d2f52b057a8813508`;
native compact index
`f314a5ded8966fca7f0ec6f9b8c330a2d280bcf126ffcb9800afe3ac88a059ba`.
Independent native wiring/utility review:
`7217f64a021420608213b49b83d6799e24c07aed22672569a95fe549cebd6f9d`.
Historical `physical_metric` eigenvalue keys measure Penrose `gtilde/χ`;
physical SPD is equivalent for Ω>0.

The separate 1120-matrix finite-frequency screen uses four a values, seven
radii and five wave numbers in radial and oblique directions. Here k is the
unscaled Cartesian phase wave number in `δu=v exp(ik n·x)`, with constant
stored primitive amplitudes. Both Q variants at `a=.5` have worst sampled
primitive `Re λ=22.4676` at `r=.85, k=4` radially, versus a spatial-norm
maximum of `1.532` over the matched sample grid (at `r=.45, k=4`). At the
Q maximum, `W=1` but `v=0` exactly. The original-Q
sigma-zero control reaches `251.97` near `r=.98, k=0`. Stable scri poles
therefore do not remove this sampled finite-radius growth. These are frozen
primitive matrices; they do not establish constraint-subsidiary growth,
transported global instability or eigenvalue convergence.

The negative-screen index is
`adaa2b1437054f4ba4cf6471eafb6f83af96a31e9ea376d723a2666abcf5e72a`.
Independent review verifies all 1120 roots and the lapse/feedback Jacobians;
its index is
`62d9e68a59858026e09a750f981ddd820ace5d2395208ac769ce7818e8541cc4`.

## Cartesian rejection

Actual Cartesian N16, span 2.2 source and projection checks pass: full22
CSR/cache discrepancy 1.106e−15, native centered-RHS discrepancy 8.952e−10,
final-only RK derivative discrepancy 1.903e−10 and projector discrepancy
3.0e−16. Only local chi/metric/alpha/beta value columns change; geometric,
P, Θ, A and Λ rows remain bitwise equal. An independent short canonical
action agrees with exploratory Arnoldi to 2.874e−14.

The guarded t=2 projected-continuous action is decisively negative:

| Seed | Candidate H/M/Z | Original spatial-norm H/M/Z |
| --- | --- | --- |
| Angular gauge | 10866.019 / 31087.375 / 5713.473 | 2.19266 / 1.22040 / .369710 |
| Shell/random | 1580.822 / 4535.785 / 841.236 | .00250440 / .00228468 / .000996051 |

Gauge constraint ratios are approximately 4956/25473/15454. Component
amplifications are 423893 and 18730; these are not physical energies. The
states stay finite and do not hit the guard, which does not qualify as
stability. Maximum local pair error is 9.695e−11, curve defect divided by
state norm is 1.326e−10 per unit time, and shared short-prefix discrepancy
is 1.101e−14. Retained floating-status warnings have no assigned cause;
finite-state, direct-residual and independent short-action checks pass.
No nonnormal forward-error bound follows from those local checks.

The global index is
`d12e8e4da86f0918cfc7ad741c61a4cff3214494dc808f90df9eb061f0918214`.
No t=6 or longer nonlinear native run is justified for this candidate. This
is a coarse fixed-grid rejection, not a continuum instability theorem.

Separately named earlier-feedback alternatives remain under cheap
screening. No production promotion, closed scri formulation, stable pulse
or black-hole admission follows. Later black-hole acceptance must include
the inner wormhole-to-trumpet transition with the Minkowski hyperboloidal
reference retained throughout.

The [compact archive](validation/hyperboloidal-conformal-Q-null-feedback-experiments-20261009/README.md)
contains 307 cataloged blobs totaling 6,080,735 bytes: 97 parseable finite
JSON files and one explicitly classified historical invalid-JSON artifact.
Catalog SHA:
`c093e98a4e6cef14ee7cca8950c1cad930eb6245cf716fcd379d571e69029156`.
Large arrays, binaries and numeric payloads over 1 MiB remain metadata-only.
The malformed historical FD-reader output retains its exact leading-dot
decimals. The first collector stopped on it; its collector, observed-error
receipt and partial copied archive remain preserved. A fresh v2 collection
hash-checks this specific opaque failure without treating it as valid JSON.

The [earlier-feedback and first-jet follow-up](hyperboloidal-Q-early-and-jet-audit.md)
also rejects moving the onset earlier. Its actual complete residue map
identifies omitted Theta/shear conditions, and an initially Einstein-compatible
quadratic shift shows that sigma-five fails null-jet time tangency even when
the next leading RHS pole vanishes. A separate initial gauge-only ADM/Box
identity motivates further analysis of sigma-three; it admits no evolution
candidate or nonlinear closure.

The [complete angular and geometric jet audit](hyperboloidal-Q-angular-geometric-jet-audit.md)
confirms sigma-three's initial gauge-only cancellation and exposes an
independent Einstein geometric null time-jet defect. It adopts no new
production gauge or boundary prescription.
