# Scratch conformally flat spatial layer height

This is an ignored research prototype, not a native evolution acceptance result.
No tracked files were edited. `flat_height.hpp` is frozen for the boundary
agent's actual Cartesian tangent comparison. Its SHA256 is
`2ca1f35470be9a140b6e025827c460ea6a831154d85fcd6379216acaed3475a9`.

## Geometry and consumed jets

Keep the current monotone compactification Omega and let

```
z = -r Omega'
L = Omega + z
b = sqrt(z (2 Omega + z))
h_R = b/L, h_r = b/Omega^2
alpha = L
beta^i = -b n^i
chi = 1, gtilde_ij = delta_ij, Lambda^i = 0
```

The conformal four-metric is
`-Omega^2 dt^2 - 2b dt dr + dr^2 + r^2 dOmega_sphere^2`.
Its determinant factor is L, and `g4^rr=Omega^2/L^2`.
The physical spatial metric is `delta_ij/Omega^2`.
The existing `0<r0<r1<S`, `a>=S/2` conditions imply Omega>0, Omega'<0,
L>0 and b/L<1 in the transition. No additional mathematical admissibility
condition is required.

Define

```
Kbar = (-b' - 2b/r)/L
D = (-b' + b/r)/L
D' = (-b'' + b'/r - b/r^2)/L - D L'/L
A_ij = D (n_i n_j - delta_ij/3)
P = [-Omega (b' + 2b/r) + 3b Omega']/L
  = (-Omega b' + b Omega')/L - 2b/r
P' = [-Omega (b'' + 2b'/r - 2b/r^2)
       + 2Omega' (b' - b/r) + 3b Omega'']/L - P L'/L
L' = -r Omega''
L'' = -Omega'' - r Omega'''
```

Supply alpha through second derivatives, beta through second derivatives,
P through first derivatives and A through first derivatives. Scalar/tensor
Cartesianization uses the existing `CartesianRadialJet`; metric, chi and
Lambda jets are exactly constant/zero. P, A and Lambda second derivatives
are not consumed by the current tensor kernel. No division by b or by r is
performed in the exact core branch.

The inherited compactification quantities remain correct:

```
Nref = Omega'^2/L^2
BoxOmega/Omega = 2Nref + Omega/L^2
                  (Omega'' + 2Omega'/r - Omega' L'/L)
c_out = L+b
c_in = -Omega^2/(L+b)
```

At and outside r1 the prototype returns the original CMC branch verbatim;
at and inside r0 it returns the original exact Cauchy branch verbatim.

## Endpoint evaluation

Use `s=(r-r0)/width`, `t=(r1-r)/width`, `g=-1/s+1/t` and `e=exp(g)` in the
inner half. Let `D0=1-Omega_out`, `E=-Omega_out'`, and

```
F = D0 g'/(1+e)^2 + E/(1+e)
z = r exp(g) F
log(b) = [log(r)+g+log(F)+log(2Omega+z)]/2
```

Analytically differentiate log(b), then evaluate b, b' and b'' with their
own scaled exponentials, `sign(c)*exp(log(b)+log(abs(c)))`. This retains
representable derivatives after e, Omega', or b itself underflows. The
prototype obtains F's first two derivatives using ordinary Radial2 algebra;
it only needs Omega through order three and g through order three.

Near the inner endpoint,

```
b ~ sqrt(2 r0 (1-Omega_out(r0))/width) exp(g/2)/s,
g = -1/s + 1 + O(s).
```

Every derivative is flat at r0. At r1 the square-root argument tends to
positive r1^2/a^2, and the exponential cutoff supplies C-infinity CMC matching.
The 100-digit oracle includes a double-precision case at
r=0.3502578980229621 where b=0 and b'=0 but b''=3.65e-321; the prototype
retains that derivative. Relative boost-oracle error in the inner half is
at most 6.51e-14; maximum absolute error is 1.48e-12. Outer b'' checks use
absolute tolerance because it is an exponentially tiny correction between
finite terms.

The prototype is robust for the project scales and nextafter endpoints.
Arbitrary floating-point extremes can still overflow g1^2/g3 or form
0*infinity in an exponential jet. Production work should either enforce
representable dimensionless scales/widths or evaluate all factors with scaled
logs. This is a numerical-range limitation, not a loss of smoothness.

## Independent constraints and tests

An independent warped-metric calculation gives

```
kr = (-Omega b' + b Omega')/L
kt = -b/r
R3 = 4Omega (Omega''+2Omega'/r) - 6Omega'^2
Mr = -2kt' + 2(rho'/rho)(kr-kt) = 0, rho=r/Omega
Fdefect = b^2+Omega^2-L^2
H = 2Omega Fdefect'/(rL)
    + 2(Omega-3rOmega')Fdefect/(r^2 L)
m_MisnerSharp = r Fdefect/(2Omega^3)
```

Thus H, M and Minkowski mass vanish identically. This proof does not rely
on reference subtraction. `independent_constraints.py` also verifies the
inherited Box identity.

`run_audit.py` reproduces six compilations and nine successful run stages
(three Release tests, three ASAN/UBSan Debug tests, boost export, mpmath,
symbolic proof) in 9.32 seconds. macOS lacks LeakSanitizer, so detect_leaks=0;
AddressSanitizer and UndefinedBehaviorSanitizer are active. Logs, binary
SHA256 values, commands and durations are in `receipt.json`.

Default sparse point audit:

- H <= 1.06e-13, M <= 4.18e-14.
- Geometric fixedpoint <= 5.69e-13 for Omega>=1e-4.
- Physical-P gauge fixedpoint <= 1.78e-15.

Extended audit: 1,498 axis/oblique points across seven configurations,
including a=S/2, S=2/a=3.5, broad layers and width .02; 85 exact core/outer
objects are byte-identical to the original. Worst absolute H/M
6.63e-10/3.78e-10 and regular geometry/gauge residuals
9.78e-9/2.92e-11 come from the intentionally narrow, stiff layer.
Independent Cartesian finite differences of all consumed jets give
normalized first/second errors <=5.81e-10/2.26e-5.

The 800 signed actual-kernel probes perturb every one of the 20 fields on
axis and oblique outer points, both gauge choices, and exact scri. Every
regular and pole block is byte-identical, so outer principal and linear pole
matrices are unchanged. This does not claim that transition lower-order
stiffness or nonlinear regularity closure improves.

At Omega~1e-15 the existing unfactored geometric assembly amplifies
roundoff in `(P-3wn)/Omega` and related terms, reaching 0.33 in the default
case and 1.33 in the a=.5 sweep. The new and old outer results are
byte-identical. These points are tested for parity and finiteness, separately
from the regular-region stationary residual gate.

## First native comparison and production path

A dense 20,000-radius scan gives:

| a,r0,r1 | max alpha | max abs(P) | max abs(Axx) | max outgoing | gauge bound |
|---|---:|---:|---:|---:|---:|
| 1,.35,.75 | 2.67481 | 22.70542 | 14.80396 | 5.29084 | 6.02632 |
| .5,.2,.8 | 2 | 9.55368 | 2.88793 | 4 | 4.30577 |
| .5,.15,.85 | 2 | 9.08043 | 2.30887 | 4 | 4.20543 |

Use a=.5,r0=.2,r1=.8 for the first native comparison: mild distortion,
larger exact Cauchy core than .15→.85, and the current gauge collar remains
admissible. The old height on this broad configuration has max abs(P)
6.35941 and max abs(Axx)2.59710; flatness increases some curvature even
though it eliminates the metric/connection transition derivatives.
The live MaxGaugeSpeed already scans
`abs(beta_i)+sqrt(alpha2f*chi*gtilde^ii)` and bounds the light speed because
alpha2f>=alpha^2. A narrow width .02 can have outgoing speed64.12 and
should not be used as the initial stability experiment.

If the actual Cartesian tangent/evolution evidence supports adoption:

1. Add an optional boolean after existing r0/r1 members in LayerParameters,
   default false, preserving current aggregate callers and legacy path.
2. Parse the explicit native opt-in parameter, e.g.
   `layer_flat_spatial_reference=false`.
3. Compute common Omega/L jets once in LayerReference::At; branch into this
   flat-height state construction before the old metric/connection algebra.
   Keep exact core/outer returns and their small-Omega continuation.
4. Keep the reference Minkowski. No black-hole RHS subtraction is introduced.
5. Explicitly reject mass>0 with the new flag until LayerWormhole's hard-coded
   `b=r*w/a` curvature formulas are generalized. Existing mass>0 formulas
   must not silently run against the new reference.

For the later wormhole generalization, with m=M Omega/(2r), psi=1+m and
N=(1-m)/(1+m), the mass-corrected-height curvature is

```
kr_BH = psi^-2 [kr_ref - 2(b/r)m/(1-m^2)]
kt_BH = -psi^-2 (b/r)N
(kr_BH-kt_BH)/Omega = psi^-2
       [D_ref - b M(2-m)/(r^2(1-m^2))].
```

These factored formulas preserve outer regularity and the exact Cauchy
throat branch. Only first P/A jets are consumed: do not add an unnecessary
b'''/Omega'''' requirement to populate unused trace Hessians. Recover b's
second jet from `-beta_ref*L/alpha_ref` and D_ref from the reference A/metric
if a generic reference interface is preferred. Lapse/shift mass factors and
the positive pre-collapsed core can remain as in the derived wormhole class.
The user's Minkowski gate still precedes any black-hole evolution acceptance.

## Actual Cartesian tangent comparison: adverse initial evidence

Boundary agent's unchanged native tangent harness, compiled through an ignored
header overlay, reports that this specific height worsens the finite difference
transition defect. Broad a=.5 degree2 Hdot RMS at N24/36/48 is
5.8861/5.3953/5.0517 versus original .79715/.37021/.21625. The gentle case
gives 19.9397/25.3065/28.5217 versus original1.9382/1.6038/1.2818. Degree3
has the same bulk maxima. At N48 flat peak Hdot is89.843 at r=.2426 broad
and366.139 at r=.37316 gentle; nearly all Hdot squared lies below r1.
Exact-jet continuum checks pass at appropriately resolved difference steps.
N64/72 and dt consistency supplements remain with that agent.

This evidence does not support production adoption of the current flat height.
Eliminating metric/connection gradients is insufficient when the square root
creates sharper boost/curvature derivatives near the start of the transition.
The broad geometry above is the appropriate comparison configuration, not a
stability recommendation after these adverse native results.

A possible later compactification experiment is
`Omega=(1-w^2)+w^2 Omega_out`. It retains monotonicity and exact endpoints,
but gives inner `b~exp(g)/s`, instead of `exp(g/2)/s`. It may shift the peak
distortion outward and must be tested. A derivative-controlled alternative is
`-Omega'=(r/aS)w^2[1+c(1-w)^2]`, with c chosen by the outer endpoint integral;
then the inner boost scales as w. That requires accurate cumulative quadrature
for Omega. Neither candidate has been implemented, and no evolution claim is
made for either.
