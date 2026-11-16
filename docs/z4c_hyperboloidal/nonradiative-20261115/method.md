# Nonradiative validation of unchanged A

This directory separates analytic geometry/extraction checks from native A
evolution. The complete request is preserved in `request.txt`. Baseline source
was `a0346b767a7f2a03ac6d6e2974af09f5b050581a`. No numerical evolution object,
gauge parameter, penalty, projection, shift compatibility solve, or lift changes.
The recorder's optional `boundary_diagnostics=true` evaluates new passive
observables from accepted states. Only full overlap (`agreement_full`) is used.

## Reference definitions and units

[Moxon et al., V.A (HTML V.1), Eqs. 49–51, Table I, Fig. 10](https://arxiv.org/html/2110.08635v1)
use rotation `phi_old=phi+omega*u`, `omega*M=0.1…0.8`; the bounce is
`x_old=x+2M sin^4(2 pi t/(40M))`; and the gauge transformation is
`v=t+R+F(t-R)/R`, with `F=A sin(0.5 u) exp[-(u-25)^2/100]` in M=1 units,
A=0.01…10. The rotation is right-handed around z. Table I covers lmax=8…24;
the bounce radii in Fig. 10 are 15,20,25,30 M. Expected news is zero.

[Barkett et al., VI.3–VI.5, Eqs. 149–153](https://arxiv.org/html/1910.09677)
use the same rotation/bounce transformations. Their gauge profile instead has
phase 0.01, center 40 M, width 10 M, and amplitude 1, with an additional
z translation `2[1-exp(-(t/40)^4)]`. These are distinct tests; the primary fixture
uses Moxon's parameters. The sine-fourth function actually repeats after b/2;
we retain the published b=40 M rather than alter its argument.

The Moxon PDF pp.16–18 was checked. Eq.51 omits a printed `dR^2` after its
radial coefficient; we derive the metric from the stated coordinate pullback.
The extra `sin^2(theta)` on the printed angular line of Eq.49 is not applied
to the entire sphere metric: the pullback of Schwarzschild gives
`R^2(dtheta^2+sin^2(theta)dphi^2)`. Both choices are independently checked by
vacuum Ricci and finite-radius pullback comparisons.

Here `M_code=1/3`, lengths and times are M_code times the M=1 paper values,
angular frequencies are divided by M_code, and F has units of length squared.
The physical radial coordinate is R, compact radius is r, and the common
conformal factor near scri is `Omega=(1-r^2)/2`. News normalization is
`d[R(h_plus-i h_cross)]/du_B`; the polarization axis of the extractor is x,
while the paper rotation is about z. Zero news is insensitive to the overall
news-convention factor; nonzero component magnitudes must use our convention.

## Regular analytic chart and an important limitation

For the analytic fixtures set `R=r/Omega`, `u=t+(1-r)/(1+r)`. This uses the
native conformal factor and surface topology. Its interior radial coordinate
is not the native isotropic-CMC reference map; native evolution uses its own
unchanged reference. The exact outgoing Schwarzschild line element pulls back
to a regular metric, with mass term `2M Omega^3/r du^2`. Its endpoint second
jets equal Minkowski's. The rotating fixture replaces the angular differential
by `dn+omega (ez cross n)du`; all coefficients remain regular.

For the literal gauge-wave metric, unperturbed ingoing Kerr–Schild time is
`T=u+R+4M log(R/(2M)-1)`. Consequently the physical outgoing coordinate is
`U=u+F(u+4M log(R/(2M)-1))/R`. We use this exact expression in the interior.
At scri the sine-Gaussian correction is flat: every finite derivative vanishes,
since exp[-c(log R)^2] dominates all powers of R. Assigning these derived zero
limits is not an evaluation of an infinite physical radius. The endpoint
alone is therefore a weak test of this family; finite-radius conversion and
native radial support remain necessary.

A literal bouncing Kerr–Schild chart does not have a smooth endpoint under a
naive radial hyperboloidal height. Already in flat spacetime, writing
`X=x+a(T)ex`, `T=t+h(R)`, `h'->1`, gives a radial physical coefficient tending
to `2 n_x a'(T)+a'(T)^2`. Multiplication by
`Omega^2(dR/dr)^2` produces an oscillating inverse-square divergence. It must
not be assigned a numerical scri limit. An exact regular conversion uses the
inverse translation `x=X-a(T)ex` and inertial outgoing U; this cancels the bounce
and yields the common Schwarzschild control. Independent finite-radius matrix
pullbacks test that cancellation. It does **not** reproduce the paper's moving
worldtube characteristic evolution.

The additional `retarded_bounce` fixture uses `X=x/Omega+a(U)ex`. It is an exact
Schwarzschild spacetime with smooth, time-dependent coordinate derivatives,
but is explicitly a counterpart, not Eq.50 in unchanged Kerr–Schild time.
With `Z=x+Omega*a(U)ex`, `L=|Z|`, its regular formula is

```
g = -Omega^2(1-2M Omega/L)dU^2
    -2 dU (Omega dL-L dOmega) + dZ.dZ-dL^2.
```

All initial cuts used by the frame test are analytically round. A supplementary
angle-dependent clock `U=u+epsilon M sin(u/M) P2(n_x)` also starts on a round
cut, but exercises nonzero conformal/frame terms. No arbitrary uniformization
or fitted background subtraction is performed.

## What each path tests

`nonradiative_analytic.py` supplies second spacetime jets via forward AD,
independent of the production reconstruction and recorded-time differences.
Literal finite-radius paper metrics satisfy vacuum Ricci; an independent
finite-difference derivative check guards Jacobian wiring. Exact endpoint jets
go through the existing Ricci and full transported-frame extractor.

`nonradiative_sampling.py` evaluates metric-derived native chi, determinant-one
metric, lapse and shift at the *actual* archived donor coordinates, applies the
saved native derivative rows (including angular seams), restores references
and algebraic components, then reconstructs the four-metric. This isolates
sampling/derivative errors with an exact analytic frame. It does not apply A's
normal compatibility solve to prescribed gauges that do not satisfy A.
In particular, rotating tangential shift conflicts with A's projected radial
endpoint shift. This row replay must not be called a complete native evolution
or a full recorder/normal-lift replay. Actual native A runs use all projections
and normal lifts unchanged.

The gauge-wave radial layer explains why these donor tests need not yet be
asymptotic. At t/M=.6, ignoring the small interior height offset,
`argument/M=.6+4 log(R/(2M)-1)`. The envelope center 25 M lies near r=.99665;
its +1-width point lies near r=.999724, and its +3-width point near r=.99999814.
The outermost N=12/16/20 donors are .995/.99625/.997. Refining through this range
moves the stencil into the profile peak, while most of the decay to the flat
endpoint remains unresolved. This is a feature of this declared compactified
coordinate profile, not evidence of radiation or a derivative-oracle failure.
No prescribed-geometry result is substituted for a native evolution result.

## Pointwise accepted-state diagnostics

The recorder adds four arrays, all indexed identically to `surface_ids`:

- `projected_metric_rates[q,11]`: chi, six conformal metric components,
  lapse, three shift components. The gzz derivative is obtained by differentiating
  the implemented five-component determinant-one chart. Lapse rate is zero.
  Shift rate is `-s_t n`, where `s=sqrt(chi g^nn)` uses that same chart tangent.
- `unprojected_boundary_rates[q,31]`: freshly evaluated boundary equation rates,
  including geometric penalties; never a stale RK stage buffer. Here
  "unprojected" refers to the metric-chart and shift tangents: A already returns
  a zero lapse array entry, and the discarded lapse equation is recorded
  separately as clock.
- `boundary_penalty_rates[q,31]`: pointwise penalized-minus-unpenalized rates.
- `lapse_identity_terms[q,12]`: clock, `2 alpha_n(P+3)`, `P_n`,
  `2 kappa(alpha_n-1)`, `-beta_nn`, s, -2, alpha_n,
  raw s_t, projected s_t, identity residual, raw-minus-chart gzz rate.

At alpha=1, beta=-s n, |n|=1, unchanged outer A gives exactly

```
clock = 2 alpha_n(P+3) + P_n + 2 kappa(alpha_n-1) - beta_nn + s - 2.
```

This follows by differentiating the implemented full lapse-pole numerator and
adding its regular advection term. It retains the finite-resolution P+3,
s-1, normal lapse, and normal shift defects. Projected alpha=1 alone says
nothing about this identity being zero. The per-point equality residual
checks its assumptions and implementation. For alpha_n=1, P=-3, s=1 the
fixed-point condition reduces to `P_n-beta_nn-1=0`.

The analysis forms all 16 components of
`T_ab=nabla_a nabla_b Omega-g_ab Box(Omega)/4` twice: with native spatial jets
and projected metric rates, and with the same spatial jets and seven accepted
records. It preserves both pointwise tensors, signed lapse terms and penalty
rates before maxima. D's extrapolated endpoint rate is not claimed by the
metric-rate diagnostic. It is unnecessary for this tensor. Finite RK/cadence
error must still be assessed in the comparison.

## Native initial data

`nonradiative_seed.cpp` changes only lapse in the native Schwarzschild seed.
The final compact support is (.9465,.9645), outside the configured telegrapher
blend and separated from scri by .0355. The complete N12 donor set reaches
r=.955, so the bump is not outside every archived donor. Admission is instead
checked on the reconstructed endpoint: sampled jets differ from control by
at most 2.38e-15, with identical normal lifts and projected metric rates. Its angular factor is
`1+.5(nx^2-ny^2)` and amplitude is .001. Physical spatial geometry, extrinsic
curvature, Theta, Lambda, B, D and the endpoint are unchanged. The physical
constraint evaluator has no lapse dependence; bitwise equality of its entire
initial data proves equal discrete constraints, not exact zero truncation error.
B is a dynamical auxiliary and is decoupled in this outer region.

The first trial support (.953,.989) was rejected locally: its coarse boundary
stencil produced clock=0.0677408 at the initial cut. The rejected gauge seed was never evolved on
Aurora. This is a boundary-support admission failure, not evidence against A.

The additional radial conformal-chart control sets
`Z=x/exp[epsilon sin(u/M)P2(n_x)]` in the regular metric formula, with
`epsilon=.1`. Omega itself is still the native fixed function. The induced
cut scale varies, so the preferred conformal factor, its Hessian contribution,
and the Ricci contribution are individually nonzero. The initial cut is round
because the exponent vanishes at t=0 on scri. This exact Schwarzschild
coordinate control tests cancellation of appreciable contributions without
altering the extractor. Its timestep study uses dt=.01 and .005 and its angular
study compares lmax=4 and 8.


## Strong gauge-wave chart admission

The original analytic height `u=t+(1-r)/(1+r)` gives a nonspacelike spatial
metric for the paper's largest A=10 on some native donors. Those attempted
replays are recorded as rejected admissions, not zero-news successes. A separate
analytic chart uses `u=t+H(1-r)/(1+r)`, H=16, with the same native Omega.
Its asymptotic radial spatial coefficient is `H+2F-8M Fprime` before the
flat endpoint limit. For A<=10, M=1/3, the envelope bounds give
`2|F|+8M|Fprime| <= A M^2[2+8(.5+sqrt(2/e)/10)] < 7.43`.
H=16 provides a positive asymptotic margin; the full spatial metric is also
checked positive at every sampled donor and time. This choice is declared
in every H16 result, and untransformed H16 controls are included.

At scri this chart has radial gamma=H, alpha=1/sqrt(H), beta^r=-1/H.
It is a valid analytic chart with round initial angular cuts, but is not an
exact solution of A's fixed endpoint lapse/shift prescription. No native input,
gauge parameter, or normal solve is altered by this analytic chart study.

## Duration and sampling

The implemented frozen-background outgoing lapse speed is
`-beta^r+sqrt(chi*F*g^rr)` in this outer region. Integrating its reciprocal from
the inner edge, center, and outer edge of the bump gives travel times
.0823968, .0682245, and .0541894 M. This is a principal-speed estimate, not a
full hyperbolicity certificate or an exact disturbed characteristic solution.
The N12 pilot reaches .140833 M and already shows an early discrete response,
including before the continuum estimate; the finite stencil and coarse bump
resolution matter. The principal extension reaches .718246 M, over eight times
the longest outgoing transit, and its observed late behavior determines whether
that interval supplies a useful post-outgoing-pulse comparison.

The native radial face spacings are .005, .00375, .003 at N=12,16,20, with
864,1536,2400 surface nodes. The derivative support itself is refined. All
principal runs record every accepted step; the N12 half-step run records every
two steps so that it shares the full-step run's physical recorder cadence.
Further cadence tests decimate these same records. Existing signed pole/h,
penalty and normal-lift histories are enabled in the principal runs.

## Gauge-equation admission

`nonradiative_gauge_admission.py` independently constructs lapse, shift and the
physical trace of extrinsic curvature from the manufactured four-metric:

```
K_phys = Omega/(2 alpha_bar) gamma_bar^ij
         (-dt gamma_bar_ij + L_beta gamma_bar_ij)
         -3 beta.grad(Omega)/alpha_bar.
```

It checks K=-3 for the Minkowski control, then passes these values and lapse
first derivatives to the compiled, unchanged `HyperbolicGauge` and
`ApplyStationaryGaugeSource` routines. The actual native stationary source is
retained, including its transition near r=.94; it is not assumed zero in the
outer collar. No choice of B can repair these outer lapse residuals because B
is decoupled from this equation there.

The simple analytic radial/height chart already differs from the native CMC
reference, so even its untransformed Schwarzschild case is not an A coordinate
fixed point. Rotation has an additional endpoint shift defect .15 (omega M=.1)
or 1.2 (omega M=.8); retarded translation has a .164 defect at t=5 M. The H1
literal gauge-wave endpoints satisfy the lapse/shift values, but its prescribed
interior history fails the native lapse equation; the A=1 residual reaches
4.25 at the tested r=.94,.96,.98. H16 is explicitly incompatible with A's
endpoint values. These are admission checks on prescribed coordinate histories,
not failures of A's evolution. The native pure-lapse run is the compatible
initial-value counterpart, not an exact Moxon coordinate history.

Pointwise chi, P, Theta pole/h values are also recovered from the recorded
penalty rates by inverting `penalty/h*(4 pole-8 defect)`; the lapse pole/h uses
`[-(P+3)+1-s]/h` at the projected endpoint. Full penalty arrays and all normal
lift jets remain in the native records. The remaining pole/regularity components
are retained in the existing signed/extremum history groups. The pointwise
T tensors and lapse summands are not inferred from separate extrema.


The endpoint-rate audit follows `SurfaceExchange::RHS` through the RK update:
`EvaluateScriPoint` supplies the entire surface rate. KO and overlap forcing
are bulk terms and are not separate omitted additions to `s_rhs`. The private
recorder recomputes the same endpoint equations on accepted-state jets; the
subsequent determinant-one chart and fixed lapse/shift projections are
explicitly differentiated. This is an accepted-state vector-field evaluation,
not a claim that the final RK stage buffer is the accepted time derivative.

## Interpreting observed orders

`SurfaceExchange::BuildRow` uses seven radial nodes (six centers plus scri),
five centered nodes per angular direction, and six-node peer interpolation.
For smooth resolved data, the radial first/second derivatives have nominal
orders six/five; the centered angular derivatives are fourth order. Sixth-order
interpolated values can contribute fourth-order errors after two derivatives,
so fourth-order news convergence in the smooth donor replays is consistent with
the stencil structure. This is not a promised global order for the evolved A
system with its compatibility lifts and penalties. The logarithmic analytic
layer and narrow native pulse are not demonstrably in that asymptotic regime.

Seven accepted times determine the recorded-time polynomial: nominal order six
for the first derivative and five for the second on arbitrary distinct times
(order six for the centered second derivative on a uniform stencil). Native
RK4 and frame RK4 add their own fourth-order errors. The measured timestep,
cadence, and spatial comparisons are therefore kept separate; a differentiated
quantity near its floating-point floor is not assigned an artificial order.
