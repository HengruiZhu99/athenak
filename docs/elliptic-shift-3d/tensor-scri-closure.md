# Evolved tensor scri closure, continuation after 0149acb26

The old active-only outflow extrapolation remains selectable and its evidence
is unchanged. The continuation adds actual evolved geometric/flux values on the
Omega=0 surface and the full tensor limiting equation. This is a numerical
closure with independently checked algebra and finite-resolution diagnostics;
it is not a nonlinear well-posedness theorem.

## Analytic limiting sources

For each geometric field the native tensor system supplies `R+S/Omega`. On a
smooth compatible solution, `S|scri=0` and

```
U_t|scri = R|scri + (N^i partial_i S)/(N^i partial_i Omega)|scri.
```

The combined poles in `ConformalRHS` use metric/chi first derivatives, curvature
values, trace/Theta first derivatives, Lambda values and shift/lapse values.
Their normal derivatives consequently require only the existing second metric,
chi, trace and Theta jets, curvature first derivatives, and lapse/shift/Lambda
first derivatives. No third metric derivatives or tiny Omega denominator are
needed. `tensor_scri.hpp` evaluates these complete poles with directional
automatic differentiation. The stationary unit-sphere Omega is quadratic, so its
third derivatives vanish; this adapter explicitly excludes other Omega profiles
unless their missing derivatives are supplied.

The pole evaluator is checked against the original tensor and lapse kernels on
32 nondiagonal, off-constraint metric jets, for both Theta-source choices.
Values agree to 1.22e-15. Independent physical-coordinate finite differences of
the original kernels give normal-derivative discrepancy 1.28e-9 at step2e-5.
Sixteen independent radial reductions compare the **full finite limiting rate**
with the existing `scri_limits.py`/`g1_rhs.py` formulas at 2.55e-15. The native
curvature-gradient term in the Lambda pole is minus `4 alpha A^ij Omega_j`;
the audit caught a sign transcription error in the new evaluator before any
coupled closure run. The original tensor kernel/equations were not changed.

Off the compatible manifold, l'Hopital is not a continuum limit. The code uses
the same finite-part extension as the spherical raw-l'Hopital implementation,
adds disclosed weak pole penalties, and always records the unmodified pole
numerator. It never sets a field or differential constraint to zero to hide an
incompatible state.

## Necessary gauge traces without conformal flatness

Set alpha_scri=1, `a_n=sqrt(chi g^ij n_i n_j)` and select future nullness plus
fixed angular coordinates, so beta=-a_n n and w=-a_n. With the selected
Appendix-C Theta source and positive `kappa1(2+kappa2)`, cancellation of the
complete poles implies:

```
S_chi=0 -> K_physical=-3 a_n
S_Theta=0 -> Theta=0
S_alpha=0 -> 2(a_n-1)=0
therefore chi g^ij n_i n_j=1, Khat=-3, Theta=0.
```

Thus the general metric/chi regularity trace is
`chi-1/(g^ij n_i n_j)`. It is **not** `chi-g_nn` for an arbitrary nondiagonal
metric and does not impose conformal flatness. The continuation weakly damps
these three derived traces in the chi/trace/Theta boundary rates, using the
declared spherical strengths8/h and4/h for regularity and complete poles. No
additional assumption `Lambda=V` is imposed: Lambda remains evolved, and its
complete pole is retained and weakly damped. The full curvature pole, including
reference-connection error and metric-dependent Hessian terms, is used instead
of transplanting a radial A_rr identity.

The unmodified raw lapse-clock limiting rate is recorded separately before
applying the prescribed clock datum. It measures the higher corner hierarchy;
small nodal clock error alone cannot prove that hierarchy converged. The
no-penalty and refinement controls retain any resulting defects.

## Surface/volume coupling

`ScriSurface` stores all22 geometric/gauge-flux fields at the outer physical
face, with 64-angle diagnostics remaining extraction only. The boundary fields
are advanced with the **same RK stage states and weights** as the volume. The
elliptic solve sees their live metric/chi and prescribed clock at every stage.
The geometric boundary rates use the complete tensor limit above; flux uses
its own transported/relaxed equation, since the taper and its derivatives
vanish at scri. The two angular shift conditions and future normal branch are
unchanged.

Each surface jet uses five interior radial values and the endpoint, plus
six-point angular stencils. Cartesian first/second derivatives include the
inverse Hessian. Surface/volume ghost values use physical patch ownership and
active donors; no permanent six-face field layout is assumed. The volume
MeshBlock active shape remains N cubed. Surface values and weak-boundary
controls are included in the version3 research checkpoint; version2 is readable
as the old scheme, not silently reinterpreted as an evolved boundary.

## Independent boundary residual vs active-only trace error

The old nine failures compared a six-active-point extrapolation with a nodal
Dirichlet target. A cell-centered boundary condition includes its ghost closure;
the active-only polynomial can miss an exactly supplied boundary value even at
t=0. That error remains a useful spatial/extraction diagnostic, and is retained
under its original names and in the old manifest.

The new independent boundary check reconstructs the face from **six active
values plus the first ghost**, without reading the supplied surface value. It
then evaluates the original clock, future-null and tangential equations. A
degree-six radial manufactured test distinguishes the two: active-only
extrapolation has a nonzero known truncation error, while the actual lifted
boundary equation is recovered to roundoff. This does not weaken the original
1e-13 clock/1e-12 null targets; it measures the discrete boundary equation rather
than a different polynomial extrapolation. Raw pole values, raw clock-rate
corners, full-domain constraints and active-only errors remain independently
reported, with no excluded cells or smoothed data.

The first six coupled radial/nonspherical cases at N6/8/12 reach t=.002 with
ghost-based clock/null/tangent residuals around2e-16--4e-16 and raw tensor poles
at most1.32e-9. The nonzero-pulse version3 restart is bitwise reproducible,
including evolved boundary values. Longer, temporal/tolerance and zero-penalty
controls are retained separately; these short results do not establish late
front or selected-wormhole convergence.
