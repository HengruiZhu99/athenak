# Earlier null feedback and the complete Q first-jet map

Moving the Q/null feedback onset earlier fails to rescue the Cartesian
linear evolution. This is a separate weight-only control of the rejected
[Q/null candidate](hyperboloidal-conformal-Q-null-feedback-audit.md), with
the same physical-P storage and geometric equations. A separate actual-kernel
Taylor audit identifies the additional Theta and shear conditions needed to
cancel every first-jet pole. Neither result supplies a closed nonlinear
scri prescription or stable finite pulse.

## Earlier feedback: one controlled change

The late candidate uses `v=SmoothCutoff(.85,.95)`. The new local controls use
`v=W=SmoothCutoff(.45,.85)` or `v=SmoothCutoff(.65,.85)`, retaining sigma=5.
The lapse remains the physical-P/Q blend, and all reference, characteristic,
geometric, damping and discretization choices are retained. Both weights are
one by .85; the complete outer equations and reference poles are identical
to the late candidate at r≥.95. Only the first weight is propagated globally.

Support admission applies to the tested S=1, geometry(.05,.95),
gauge(.45,.85), a=.5,.75,1,2 configurations. The feedback denominator is the
positive Euclidean gradient norm on its support. This does not admit arbitrary
transition radii. The added feedback changes the off-constraint spatial-source
extension in 0<W<1; the full preferred Box identity is restricted to W=1.
The generic delta identity `Delta Box(Omega)=v sigma deltaNraw/Omega` passes
independently at finite Ω.

The local gate has 504 complete principal matrices, 192 source/reference
rows, 160 leading gauge columns and 560 new primitive Fourier matrices. The
symbol error is 3.55e−15, canceled left-eigenfield error 8.88e−16 and maximum
normalized basis condition 11.530. Reference residual is below 7.55e−15;
the feedback Box identity agrees to 2.67e−15. Outer/leading-source comparisons
are exactly equal. Independent reconstruction of the rank-one difference
from the late feedback agrees across all 560 matrices to 7.11e−14.

For a=.5, maxima over the sampled radii and radial/oblique directions are:

| Cartesian phase k | Spatial-norm C0 | Late feedback | Earlier v=W | Earlier v(.65,.85) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.518016 | 20.399344 | 1.518016 | 1.518016 |
| 4 | 1.531952 | 22.467616 | 3.343106 | 3.169341 |
| 16 | 1.269652 | 21.298014 | 8.745175 | 8.745175 |
| 64 | −.259155 | 20.168941 | 20.168941 | 20.168941 |
| 256 | −.253218 | 15.290290 | 15.137307 | 15.137307 |

These are Reλ of frozen primitive matrices with constant stored amplitudes
in `delta u=v exp(ik n·x)`. They are not coefficient-aware subsidiary roots
or discrete global eigenvalues. The k=16 positive root lies below the
coordinate Nyquist of both coarse grids; this alone does not establish
accurate finite-difference representation or discrete growth. The k=64
warning is retained even though it is above those coarse Nyquists.

The local index is
`902d875e6da80bf3b10e7103aa59cf123c4d8a034862f0a5d9be6f9e220c56c0`;
independent review
`273fdab75ae847d4a54a6cd8b50334a215a839969de39c5bb7378b50fa24b19a`.

## Actual Cartesian weight-only rejection

The N16/span2.2 source change affects only beta value rows. Its analytic
early-minus-late expression agrees with the raw22 and projected20 matrices
to 2.001e−10 and 2.031e−10; every non-beta row is bitwise unchanged. Actual
native RHS and final-only step derivatives agree to 8.986e−10 and 1.894e−10.
Full22 CSR/cache discrepancy is 1.106e−15. Independent short canonical action
at .025 and .05 agrees to 4.240e−14. The initial discrete constraint source
is identical to the late and original spatial-norm controls for all four
checked seeds.

| Seed at t=2 | Earlier v=W H/M/Z | Earlier / late H/M/Z |
| --- | --- | --- |
| Angular gauge | 11471.899 / 37398.250 / 8004.284 | 1.05576 / 1.20300 / 1.40095 |
| Shell/random | 3176.536 / 10248.979 / 2303.739 | 2.00942 / 2.25958 / 2.73852 |

The earlier gauge H/M/Z ratios to the original spatial-norm control are
approximately 5232/30644/21650. Component amplifications are 647824 and
56729.9, versus late 423893 and 18730; none is a physical energy. All states
remain finite and the guard is not hit. Shared short-prefix discrepancy is
2.191e−14, maximum local pair error 9.956e−11 and curve defect divided by
state norm 2.074e−10 per unit time. Thirty retained floating-status warnings
have no assigned cause; finite-state, direct-residual and independent short
action checks pass. These local checks do not give a nonnormal
forward-error bound. The onset gap is not a successful repair, and no longer
native or t=6 run follows.

Global frozen index:
`296c1f8dc21915d71331bbe3436ac1f0f797afa66614a17aba109a0df1052432`.

## What cancels the full first-jet pole

An independent Taylor gate evaluates the actual C0 geometry with the Q/sigma
gauge at the outer S=1 Minkowski CMC reference, κinput=10 and sigma=5. Both
early weights have identical boundary jets. Each first jet has 80 coefficients:
the 20 algebraically reduced primitives multiplying {1,Ω,n_y,n_z} at the north
point. Cartesian derivatives of the radial unit vector and reference lapse
are retained. The 20×80 residue map has rank 11 and nullity 69 for each
a=.5,.75,1,2. The actual dual basis agrees with the reconstructed formula to
3.55e−15 across 320 columns. Release and sanitizer Debug JSON agree exactly;
an independent finite-difference check decreases from 4.0e−5 to 4.04e−9.

Write `N0=(delta Nraw)_0` and `Q0=(delta(P−3wn))_0`. The scalar/gauge rows
include

```
R0_alpha = −Q0/a²,
R0_beta_n = −5N0/a,
R0_chi = 2(Q0+2Theta0)/(3a).
```

Together with `N0=Q0=Theta0=0`, the remaining scalar rows vanish. This allows
coupled finite lapse, normal shift and geometry deviations. It does not
allow independent boundary lapse and shift values. Quadratic null regularity
also requires N1=0, which is not implied by the value pole map.

The actual momentum identity is

```
M0_i − (10a−2/a)Z0_i + partial_i Theta0 = (a/2)R0_Lambda_i.
```

If Theta0 vanishes as a boundary field and Theta1=0, M0=Z0=0 therefore
cancels the connection pole. The two tangential tracefree shear residues
must also vanish. In this reference linear first-jet space, the leading
constraint/null conditions have rank 10; adjoining R0 raises the rank to 18.
Adding full tangential identities and H1 gives rank 17. Adding just Theta1
and the two tangential shear residues raises it to 20 and contains every
row of R0. This exact rowspace statement is limited to this linear jet
space; it is not a nonlinear boundary theorem or an invariant ideal proof.

The missing conditions have explicit witnesses. `Theta=Ω, delta P=−2Ω`
keeps physical K and the ADM constraints unchanged and satisfies the leading
null/Q conditions, but leaves `R0_Lambda_n=−2/a²`. It is off Einstein because
Theta is nonzero inside. A local tangential tracefree A shear satisfies the
listed leading constraints, H1 and Theta1 but leaves `R0_Ayy=−2/a²`; it is
not a construction of full Einstein initial data on the sphere.

The separate `delta P=Ω²` control satisfies the entire first-jet condition
set but fails higher Einstein constraints. Its exact first RHS, with spatially
varying reference lapse retained, produces next residues
`R0_A_nn=−8/(3a⁴)` and `R0_Lambda_n=(12−80a²)/(3a⁴)`. This establishes
failure of invariance on arbitrary off-constraint higher jets. It does not
settle exact Einstein-compatible higher jets or finite-Q amplitude behavior.

The first-jet gate index is
`5490057dfc04e060bca65ec7ee1a3bb363a34cab0fe5c890f23d477a7bda7ec1`;
independent reconstruction
`32a4e257413ee50a8b9d09c84baa2898032d6d866a8c0247bd9967f3780578e9`.
No production falloff, ghost projection or boundary value is introduced.

## An Einstein-compatible null-jet obstruction

A separate actual-kernel probe leaves geometry, P and Theta exactly at the
reference and perturbs only `delta beta=Ω² n`. Initial physical H/M/Z/Theta
vanish identically, `delta Nraw=2Ω²/a+O(Ω³)` and the Q numerator is O(Ω²).
All first-jet R0, shear and Theta1 conditions vanish. Initial four-dimensional
Hessian/Box checks also pass: the tracefree Hessian divided by Ω is bounded,
and the scalar curvature inferred from the conformal trace identity is
finite. This does not establish all higher spacetime compatibility conditions.

Nevertheless, sigma-five gives `(delta Nraw_t)_1=−8/a³`. The exact next R0
still vanishes to roundoff; preserving R0 alone misses loss of N1=0.
Thus quadratic-null first-jet compatibility is not invariant even on this
initial Einstein gauge subspace. This is an initial smooth-in-time Taylor
obstruction, not a theorem of null-residue amplitude blowup or an exclusion
of a nonuniform stiff temporal layer.

The general radial control `delta beta=Ω^m n` has

```
delta Nraw_t = 4(m+1−sigma)Ω^(m−1)/a³+O(Ω^m).
```

Actual checks include m=2,3,4, sigma=0,3,5 and all four a values. The
complete analytic first-RHS field matches actual full20 values to 3.55e−15;
its analytic Cartesian jets are then fed into the next actual kernel.
Release and sanitizer Debug outputs agree exactly. The sigma-five negative gate is
`4cfbc8ed743c46787f617c33fdbe9875ec092fd117e82f1daddf5e165f2ef29a`.

An independent ADM/Box derivation avoids the Z4c kernel. At the initial
reference spatial/extrinsic geometry and arbitrary radial lapse/shift gauge
perturbations, it proves

```
delta Nraw_t = T(r) partial_r(delta Nraw)+C(r)delta Nraw,
T = −(r⁴+6r²+1)/(4ar),
C = (r⁶+16r⁴sigma−29r⁴+15r²−3)/(4ar²(r²−1)).
```

For `n=delta Nraw/Ω²`, the reaction coefficient is `C+2TΩ'/Ω`. At sigma=3
it reduces to `−(r²+3)(3r²−1)/(4ar²)`, with boundary value `−2/a`.
Tangential-shift terms cancel between the metric trace and Box divergence
in this initial calculation. The formulas apply in the outer collar, not
at the origin. They describe the initial gauge-only tangent subspace rather
than a closed evolution equation on general perturbed geometry. The independent
symbolic index is
`78c6870117cf1fb74e1175cda6abc38b08b5ac9092f3fce7d904bf8dd958e585`.

At sigma=3 the separate reference-pole Hurwitz condition is
`K=κinput a²>3/2`. The target κinput=10,a=.5 passes; the older κinput=5,a=.5
control fails. Removing this initial weighted-null pole does not admit a
sigma-three evolution candidate. Full angular kernel checks and the compatible
ideal on spatially perturbed Einstein geometry remain separate.

Independent review of these invariant and ADM/Box gates:
`aafc5a2aa870d3185f3b474f123cae16cb5240ddbd384dc28ffdd8c8d534c259`.
The review pins SymPy 1.14.0 and mpmath 1.3.0, independently reconstructs
the transport/weighted reaction and angular cancellation, and confirms the
actual first fields and next N1 coefficient. Exact common lapse/shift
rescaling has zero symbolic rate; the saved binary64 actual-kernel residual
reaches 4.021e−9 at small Ω. That control uses a separate 1e−8 tolerance,
so the frozen description of roundoff must not imply 1e−14 accuracy.
Original checker failures and all scientific source bytes are preserved.

Production remains implementation `27c19d20696ea6dd4704032c51dfd026218f64f2`.
Stable finite Minkowski pulses and the later inner wormhole-to-trumpet transition
with a Minkowski hyperboloidal reference remain required and unvalidated.

The [compact archive](validation/hyperboloidal-Q-early-and-jet-experiments-20261009/README.md)
contains 255 cataloged blobs totaling 5,876,027 bytes, including 71 parseable
finite JSON files. Catalog SHA:
`8da0e1eae87e5ed54a76d5d04a90038139ec145b458cca55af964e5bcdc07f43`.
All original source/index bytes and failed independent checkers are preserved.
Large arrays and numeric payloads over 1 MiB remain metadata-only.

The [angular and Einstein-geometric follow-up](hyperboloidal-Q-angular-geometric-jet-audit.md)
proves completeness of the compatible local gauge second jets and verifies
sigma-three tangency on that reference-geometry subset. A larger genuine
Einstein spatial-pullback family fails null-jet time tangency, so no constant
sigma closes the smooth ideal containing both tested families. The conformal
frame distinction and the limits of this time-corner result remain explicit.
