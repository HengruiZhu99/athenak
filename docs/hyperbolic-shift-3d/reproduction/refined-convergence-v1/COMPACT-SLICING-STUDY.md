# Spreading the slicing transition at fixed central scale

This is a separate, analysis-only initial-data experiment. It changes no
production source, frozen configuration, checkpoint, or submitted job. The
large-block comparison 9040972 retains its original slicing. The previous goal
turn was a verified wait on that live allocation; this study supplies new
evidence toward reducing the radial resolution requirement.

## Construction

For the stationary spherical slice write `f=1-2M/R` and
`a_s=sqrt(f+R^2 S^2)`, in the existing code's curvature/time normalization.
The physical spatial metric and its isotropic compact coordinate satisfy

\[
 dl^2=\frac{dR^2}{a_s^2}+R^2d\omega^2
     =\left(\frac{R}{r}\right)^2(dr^2+r^2d\omega^2),\qquad
 \frac{dr}{dR}=\frac{r}{R a_s}.
\]

Instead of specifying the slicing switch in areal radius, prescribe it in
compact radius and solve the map. With `y=1/R`,

\[
 \frac{dy}{dr}=-\frac{\sqrt{y^2-2My^3+S(r)^2}}{r},\qquad y(1)=0,
 \quad S(r)=I_z(p,7),\quad z=\frac{r-a}{b-a},
\]

with exact plateaus zero below `a` and one above `b`. All tested `p>=7`, so
the switch is at least C6 at both joins. The inner join is fixed to the
native value `a=0.6402906611936061`. For each chosen `b`, solve for `p` so
`y(a)=1/(20M)`. Consequently the inner isotropic scale stays fixed at
`k=0.10116863541977153`; this does not obtain a smoother profile by changing
the central puncture scale or shrinking the central areal 20M region.

The physical extrinsic-curvature eigenvalues and compatible stationary gauge
are

\[
 K^r{}_r=-S-\frac{rS'}{a_s},\qquad K^\theta{}_\theta=K^\phi{}_\phi=-S,
 \qquad K=-3S-\frac{rS'}{a_s},
\]
\[
 \bar\alpha=\Omega a_s,\quad \beta^r=-Sr,\quad
 \chi=\left(\frac{r}{\Omega R}\right)^2.
\]

These expressions obey the stationary metric evolution equation. They use
the **stationary-reference lapse**, not the separate free-initial-lapse blend.
The exterior has the same Omega=(1-r²)/2 as the production configuration;
its flat-inner conformal frame is unchanged. For `r>=b`, both map ODE and its
scri boundary condition coincide exactly with the original outer solution.
Thus the limits alpha=chi=1, beta^r=-1 and K=-3 at scri are retained. This is
an exact spherical geometric construction; the native boosted initializer,
reference-gauge sources and their device representation have not yet been
adapted or tested for this profile.

## Independent checks and measured initial effects

`study_compact_slicing.py` independently integrates the original native S(R)
baseline and checks it against the preserved C++ production reference samples.
Maximum scaled discrepancies in R,S,chi,alpha,K are below7.51e-13; differentiated
profiles agree within3.10e-9. Tightening the ODE tolerance for b=.97 changes
the tested fields/gradient by at most9.67e-13 scaled. Analytic Hamiltonian
residuals for all profiles are below3e-14. The sampled map derivative and
Schwarzschild-chart spacelike margin remain positive; the latter tends to
zero at null infinity as expected, so this is not a uniform-positive-margin
claim or an evolved regularity proof.

| Profile | p | Outer end R/M | Peak abs(alpha') | Gradient halfmax width | Peak abs(K) |
|---|---:|---:|---:|---:|---:|
| Native S(R),20–40M | — |40.00|29.48|.02137|6.212|
| Compact b=.93 |51.2720|41.30|29.00|.02728|9.003|
| Compact b=.95 |31.8238|58.46|18.30|.04348|6.100|
| Compact b=.97 |23.0688|98.48|13.53|.05920|4.794|
| Compact b=.99 |18.0904|298.49|10.86|.07409|4.052|

Simply expressing the switch in r with nearly the same outer endpoint is not
enough: b=.93 worsens the peak K. Broader controls b=.95/.97 reduce both the
lapse and shift gradients; b=.97 also lowers the peak chi gradient10.126→8.183.
The broadest b=.99 pushes completion to nearly300M and leaves little compact
radial space before scri; it is not automatically the preferred production
choice. These controls change the physical slicing as well as the radial map.

The derivative experiment uses independent fourth-order centered differences
on `w=r/R`, avoiding logarithmic variables singular at scri:

\[
 H=4w(w''+2w'/r)-6(w')^2+K^2-(K^r{}_r)^2-2(K^\theta{}_\theta)^2,
\]
\[
 M_{\hat r}=w\left[(K^r{}_r)' +2(1/r-w'/w)
                    (K^r{}_r-K^\theta{}_\theta)-K'\right].
\]

It measures coordinate-r²-weighted RMS on the same r=.65–.995 interval,
not the campaign's conformal/native constraint operator or physical-volume
norm. Exact extrinsic curvature is used, spatial derivatives are differenced.
At h=1/960, b=.95 lowers H/M by46.75×/59.40×; b=.97 by172.03×/277.99×.
The four tested spacings give residual orders approximately4 for these new
profiles. This shows improved *initial radial differentiability in practice*,
not evolved convergence, multidimensional stencil accuracy, admissible larger
timesteps, or measured GPU-hour savings.

The inspected figure and full machine-readable samples/results are in
`compact-slicing-study/`. Reproduce into a fresh directory with Python,
numpy, scipy and matplotlib:

```
python study_compact_slicing.py fresh-output-directory
```

Next: qualify a compact-profile native initializer/reference gauge with the
same conformal frame and independent stationary/boosted geometric tests, then
test a coarser radial mesh and its actual characteristic speeds. Preserve the
existing larger-block benchmark unchanged. A coordinate stretch alone rescales
speeds as well as spacing; no CFL or fourfold performance gain is established
by this study. The full boosted10M convergence and performance gates remain open.

Implementation audit before any production candidate: `NativeBoostedWormhole`
uses the slicing switch in several independent places. `Tail` integrates it,
`Fields` differentiates the isotropic map, and `Radial` currently reconstructs
the areal polynomial and its first/second derivatives explicitly. Replacing
only `NativeWormhole::Switch` would therefore leave inconsistent boosted
geometry. The new profile must supply consistent S, dS/dR and d²S/dR² to all
three paths, along with the height normalization at the unchanged inner join.
`StationaryGaugeReference::At` must use the same map/profile when constructing
the lapse and shift sources. The present Python control does not modify these
paths and must not be used as evidence that the boosted implementation is ready.
