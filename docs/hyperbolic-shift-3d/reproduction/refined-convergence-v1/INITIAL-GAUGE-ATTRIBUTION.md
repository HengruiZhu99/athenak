# Initial gauge and enlarged-slice compatibility

This is a continuum initial-rate attribution, not a stability proof or admission
of failed data. It follows the radial field analysis in
`stationary20-separated-failure/profiles/`. The frozen matched constant-driver
job9038285 has now also failed at the same step/time as the separated control.
Both failed runs are preserved; no alternative below has yet been evolved.

## What the failed fields show

At the final nominal radial cell r=.9984375, the failed separated-driver stage
has chi16.6774–16.6808 across all angular cells. Its relative angular range is
0.02036%; the largest such range on any nominal shell row is0.03735%. The
face-center chi changes25.23% between the last two radial cells. This is a
large, nearly spherical radial deformation with steep final-cell variation,
not the strongly corner-localized pattern of the older stationary6 endpoint.
The physical conformal radial metric is1.1072 and the mean tangential metric
about.01395 at that row; all saved active metrics remain positive. Neither
these finite values nor null projection establish a regular continuum limit.
The failed surface was not saved, so no value is inferred at r=1.

## Exact initial shear cancellation

Write the native initial slicing profile as S(R), with physical areal radius R,
and define a(R)=sqrt(1-2M/R+R^2 S^2). The native isotropic compact radial map
satisfies dR/dr=R a/r. The initial determinant-one conformal spatial metric is
the Cartesian identity, and

    Abar_rr = -2 R S_R/(3 Omega),
    K = -3 S - R S_R,
    alphabar = Omega alpha_phys.

For radial shift beta^i=b(r)x^i, its actual metric evolution equation reduces to

    d_t gtilde_rr = -2 alphabar Abar_rr + (4/3) r b_r.

The current free shift is beta=-x outside the gauge transition, so b_r=0.
It removes the shear cancellation and therefore gives the nonzero continuum
initial rate (4/3) alpha_phys R S_R. This is present with exact initial data
before any RK step; refining the stencil alone cannot remove it.

The native initial shift has b=I(R)-1 with

    I(R) = integral_R^R_end alpha_phys S_R/a dR.

Then r b_r=-alpha_phys R S_R, cancelling the metric shear. Blending to this
shift only outside the central gauge plateau preserves the free zero shift in
the central region, but introduces a smaller blend-region shear. It does not
make the entire initial solution stationary or prove stability.

The separate Killing-gauge control uses

    alphabar_K = Omega a(R), beta_K^i=-S(R)x^i.

It cancels both the metric and chi initial rates on the stationary outer
geometry. The pointwise control is only sampled outside r=.4, where the
physical Schwarzschild lapse is positive. It is not a proposal to impose a
sign-changing Killing lapse through the puncture or horizon.

## Measured pointwise controls

`initial_shear_probe.inc` calls the actual conformal metric/chi and gauge RHS
on identical initial spatial geometry and extrinsic curvature. Only initial
lapse/shift change. First derivatives use independent centered four-point
samples with spacings1e-4,5e-5,2.5e-5. Second-derivative rates are not tested.
At the finest derivative spacing, outside the large gauge end r=.78:

| initial gauge | max metric rr rate | max chi rate | max original shift RHS | max original lapse RHS |
|---|---:|---:|---:|---:|
| current free gauge |41.8879|16.9111|0|4.6010|
| native outer shift, original lapse |1.57e-9|.52537|27.1413|11.4252|
| stationary outer lapse/shift |1.59e-9|7.67e-10|24.7804|9.5563|

The metric-rate algebraic identity agrees with the assembled point equations
within7.11e-15. Native-shift cancellation residuals fall approximately16-fold
when the differentiation step halves, consistent with the derivative stencil.
The peak current metric rate occurs at r=.90875, R/M=31.7337. The old moderate
6M–18M slice has a smaller but still nonzero maximum12.4762 outside its gauge
end r=.65. Moving the slice farther out therefore worsens this particular
initial forcing under the unchanged free-gauge choice.

The extra restoring source alone reaches26.7459 for the native outer shift,
and24.7705 for the stationary outer shift. Thus merely correcting the initial
shift, while keeping the current evolution targets, immediately drives the
gauge away from that cancellation. It is not yet a coherent repair.

Reproduce with a fresh analysis build using `build_endpoint_diagnostic.py
--body .../initial_shear_probe.inc --output UNIQUE_DIR`, then run its reader
with each configuration and a fresh CSV. `analyze_initial_shear.py` preserves
the measured data, source/build hashes and inspected plot under
`initial-gauge-shear/`. The raw compressed CSV includes all four modes and
all three derivative-step controls.

## Next candidate, pending matched evolution evidence

A compatible stationary outer reference gauge is a justified next local
control. It would retain the actual inner telegrapher equations and precollapsed
lapse, blend to stationary outer lapse/shift beyond the constant-Omega core,
and drive deviations from that same reference through the late slicing
transition. Geometry/constraints must never be reset to analytic data.

A fixed reference subtraction alone is insufficiently justified. In the
current lapse restoring factor -2 xi alpha(alpha-A)/Omega, linearization at a
reference alpha_ref<A/2 has the wrong damping sign if one only subtracts its
constant background value. A consistent reference-lapse target must be
derived as well; do not implement naive whole-RHS subtraction and label it a
stability repair. Any reference correction must vanish in the inner plateau,
preserve the scri limiting equations/normal compatibility, and have nontrivial
perturbation tests, not just a zero-background test.

For the stationary CMC outer region S=1, beta_ref=-x exactly and
alpha_ref-A=O(Omega^3); the assembled lapse reference residual is at most
O(Omega^2). This suggests preservation of the current scri value and first
normal limiting condition, but an implemented correction must verify its
actual asymptotics and CPU/device paths. Boosted data must use the same map
and conformal frame, retain the true boosted geometry, and receive an explicit
fresh identity. No such new campaign has been submitted yet.

## Full geometric RHS check

`initial_stationary_geometry_probe.inc` extends the initial pointwise control to
all18 geometric RHS fields, including independently differenced second
and mixed derivatives. Twelve radial samples cover.4–.95. No production field,
gauge equation, boundary value or evolution is changed by this audit.
For compatible stationary lapse/shift, the maximum geometric RHS residual at
derivative spacings.0008,.0004,.0002 is.00209652,.000130388,.00000812470,
corresponding to orders4.007 and4.004. At.0001 it reaches1.74e-6; at.00005
cancellation/roundoff raises it to5.60e-6. The current free initial gauge instead
retains a maximum rate about1753.43 on these samples, dominated by Lambda.
This is not a convergence result for an evolved solution or an admission of the
original gauge equations as stationary. It verifies that the initial geometry
can satisfy the actual geometric RHS in a compatible outer gauge, to the
independent derivative accuracy. Evidence/reproducer and inspected figure are
in `initial-stationary-geometry/` and `analyze_stationary_geometry.py`.

The paired failed evolutions provide complementary evidence: changing only the
inner policy leaves the accepted failure step unchanged, and nominal r>.875
failed active fields differ by at most9.03e-4 scaled. The original inner policy
is therefore not necessary for this particular early outer failure. The missing
failed surface still prevents exact attribution of the rejected interpolation.

### Candidate source algebra (not implemented)

Let G_alpha(U) and G_beta(U) denote the original outer gauge RHS, A=(1+r²)/2,
and U_* the stationary outer reference geometry with alpha_*=Omega a,
beta_*=-S x, B_*=D_*=0. A source-only candidate is

    G_beta_new(U) = G_beta(U) - G_beta(U_*),
    G_alpha_new(U) = G_alpha(U) - G_alpha(U_*)
                    + 2 xi (alpha-alpha_*) (alpha_*-A)/Omega.

The extra lapse term is necessary: the resulting restoring contribution is
exactly -2 xi alpha(alpha-alpha_*)/Omega, whose derivative at positive alpha_*
is -2 xi alpha_*/Omega. Subtraction alone would instead give
-2 xi(2 alpha_*-A)/Omega. This statement concerns only that damping term;
it is not a stability claim for the complete coupled system.

Apply changes to the *outer* equation before the existing complete-RHS gauge
blend. The original inner plateau remains exactly unchanged. Initial lapse
must blend from the retained precollapse to alpha_* outside the flat core;
the same spatial geometry/K and configured conformal frame are retained.
Reference source values are prescribed functions of position, so their
subtraction does not change the principal symbol. The lapse correction is
also lower order. In the S=1 region beta_*=-x and alpha_*-A=O(Omega³), so the
added lapse term is O(Omega²) for bounded lapse perturbations. Its value and
first normal derivative vanish at scri; this must be checked in the actual
closure implementation, including dependent D and nontrivial geometry.

Implementation should precompute reference values/source arrays once and
upload them with mesh geometry. Recomputing native quadrature per cell per RK
stage would be avoidable performance overhead. A bounded local test must
verify reference balance, nonzero perturbation response, inner exact identity,
asymptotic limits and CPU/device agreement before any separately bound fresh
stationary evolution. Merely zeroing the background RHS is not sufficient.


### Follow-up: consistent curvature target required

The v1 source algebra above cancels the reference and repairs the explicit
lapse-restoring sign, but a further local perturbation audit found a remaining
positive lapse response up to27.049 from the unmodified K+3 slicing factor.
V1 was never submitted for evolution. V2 adds
[F(alpha)-F(alpha_*)](K_*+3)/Omega to the outer lapse RHS, converting that
factor to K-K_*. Its sampled held-geometry lapse response is strictly negative;
it still does not establish coupled stability. The actual geometry/K is never
reset. The correction vanishes identically in the CMC S=1 neighborhood of
scri. See STATIONARY-REFERENCE-GAUGE.md, stationary-reference-response/ and
the versioned v1/v2 local verification records. Original failed campaigns and
the unused v1 scientific snapshot remain immutable.
