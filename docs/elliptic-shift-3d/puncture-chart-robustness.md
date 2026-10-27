# Puncture charts across changing radial regularity

This audit compares equation-preserving variable transformations before any
new nonlinear run. It does not assume that `chi~r^4`, `alpha~r²`, a fixed
trumpet exponent, or a regular-center Taylor series holds universally. It does
not modify the selected Minkowski-reference gauge, initial slice, differential
constraints, or shift boundary data. The earlier fixed-power angular AF audit
is a conditional early-time reduction, not a proposed universal atlas.

Reproduce the few-second, CPU-only calculation with

```sh
python3 tst/elliptic_shift_3d/analyze_puncture_charts.py \
  --output docs/elliptic-shift-3d/evidence/puncture-chart-robustness-20261027.json
```

The recorded run takes under a tenth of a second. Its 80 analytic
transformation cases include time-dependent powers and moving centers, with
maximum normalized chain-rule/reconstruction error1.40e-14. A separate Cartesian
vector-Lamé product-rule test has maximum error4.89e-16. The report retains81
conditioning comparisons and9 expected extreme-scaling floating-point
failures. These checks establish algebraic equivalence and exhibit failures;
they are not Einstein evolutions or nonlinear-stability results.

## What changes as the puncture regimes change

Let `rho=|x-X(t)|`. Representative manufactured fields are

    U = rho^p(t) exp(H(x,t)),    U>0 for rho>0.

The test covers chi powers4,3.5,3,2.5,2 and lapse powers2,1.7,1.3,1.09. These
are examples spanning the requested comparison, not a claim that every
wormhole/trumpet solution has those exponents. The positive smooth multiplier
can vary in all Cartesian directions. A finite-time AF endpoint and a
trumpet-like exterior annulus can also coexist: a nonuniform limit need not be
described by a single `p(t)` at all radii.

For the centered manufactured field without H,

    partial_i U = p rho^(p-1) n_i,
    partial_ij U = p rho^(p-2)[delta_ij+(p-2)n_i n_j],
    U_t/U = p_dot log(rho) - p n_i Xdot^i/rho + H_t.

Thus a noninteger positive power can be bounded and positive but have limited
Cartesian differentiability. For unknown smooth angular coefficients,
`rho^p F(n)` is safely `C^k` only for integers `k<p`; special radial polynomial
cases can be smoother. A logarithm does not make its spatial derivatives
regular: `grad(log U)~p n/rho` and its Hessian is `O(rho^-2)`. A uniformly
changing exponent introduces `p_dot log(rho)` even after subtracting any *fixed*
power. These terms must be retained, not discarded as a reference update.

The powers should therefore be diagnosed from resolved physical fields and
their derivatives, for example effective radial slopes `rho partial_r log U`
with angular scatter and resolution controls. A tiny unresolved inner region
cannot be declared regular because it survives. Nor does a fitted exterior
power prove the power of the actual second-infinity endpoint.

## Fixed division versus fixed overregularization

Take a known, fixed positive factor `f(rho)` and write either

    v=U/f,   U=f v,                 (divide the unknown)
    w=f U,   U=w/f.                 (multiply/overregularize the unknown)

On the punctured domain these are invertible in exact arithmetic. At the
missing endpoint, a factor tending to zero is not invertible, and the allowed
function space/endpoint information must be specified independently.

For `f=rho^s`, division gives a stored power `p-s`; multiplication gives `p+s`.
Consequently:

| Physical example | Fixed division | Stored behavior |
|---|---|---|
|chi~rho^4|chi/rho^4|finite leading coefficient|
|chi~rho²|chi/rho^4|rho^-2 divergence|
|alpha~rho²|alpha/rho²|finite leading coefficient|
|alpha~rho^1.09|alpha/rho²|rho^-0.91 divergence|

A chart based on the first line is not a universal regular chart for the
second. The current generic rational factors have these same limiting powers.
Their successful initial reconstruction does not establish robustness after
the physical regularity changes.

The conditioning table tests both the pure powers and the actual centered
generic factors `C=rho^4/(rho^4+Omega^4)` and
`L=Aref*rho²/(rho²+Omega^4)`. Their finite prefactors change the stored range,
but not these limiting divergent powers when used to divide the physical
unknown in the later illustrative regimes.

Multiplying by a *fixed* sufficiently large positive power does make the
stored examples smoother. For instance `w=rho^6 alpha` has illustrative
power7.09 when alpha has power1.09; generic smooth angular coefficients then
give at least `C^7` stored data. This is a reason to investigate fixed
overregularization, but it is not a proof that the transformed equation or its
physical reconstruction is better conditioned.

For linear rescalings, the **relative** condition number with respect to a
relative perturbation of the stored value is1 in exact arithmetic. It would be
incorrect to call every such transformation intrinsically relatively ill
conditioned. The **absolute** reconstruction Jacobians are different:

    dU/dv=f,       dU/dw=1/f.

Uniform absolute interpolation/solver errors in w can therefore be amplified
by `rho^-s`. Conversely, division can suppress an absolute coefficient error
in physical reconstruction while making the coefficient itself unbounded and
nonregular. Norms and truncation errors, not a roundtrip alone, decide the
usefulness of either representation.

The tests retain both effects. At rho=1e-24, `alpha~rho^1.09` divided by rho²
has coefficient about6.92e21, despite a machine-precision pointwise roundtrip.
Multiplying it by rho^6 remains representable, but its physical reconstruction
Jacobian is1e144. More extreme rho^16 scaling causes recorded storage/factor
underflows or nonfinite reconstruction while the original positive physical
field remains representable. Those are expected failed stress controls, not
clipped values or accepted states. The common-absolute-error stress in the
manifest is a norm-conditioning test, not an assertion that every floating
operation incurs that same absolute error.

For `ell=log U`, an absolute error delta ell gives relative physical error
approximately delta ell, independently of the magnitude of U. This is useful
for positivity/range, but does not remove singular spatial jets, large additive
sources divided by U, RK stiffness, or exponential underflow. Direct positive
fields avoid those divisions but require a tested positivity-preserving time
method or a failed-step response. Neither representation licenses a floor,
an upper cap, constraint projection, or relabeling a failed run.

## A fixed bounded positive normalization

The tested alternative

    f(rho)=1+3 exp[-(rho/.25)^2],    1<=f<=4,

does not assume any physical power. Its inverse and its first/second logarithmic
derivatives remain bounded with a fixed length scale. Scaling by this factor
preserves equivalence of absolute norms up to fixed constants. It also
preserves the original cusp: this is a conditioning normalization, not a
claim of smoothing the physical endpoint.

There is a material distinction between “positive for every rho>0”, “bounded
above”, and “bounded above **and below away from zero** uniformly in
resolution”. `rho^6` satisfies the first two on a bounded punctured domain,
but not the third. A softened factor involving a positive length ell may have
bounded inverse at fixed ell, but its derivative bounds can grow as ell^-1
and ell^-2. Taking ell proportional to grid spacing does not establish a
uniformly conditioned continuum transformation. The length and its sensitivity
must be recorded rather than hidden in a shrinking excluded core.

## All transformed terms, including time and center transport

Let sigma=`log f`, epsilon=`+1` for `w=f U` and `-1` for `w=U/f`. For the
manufactured scalar test vehicle

    U_t = beta.grad U + kappa Delta U + S,

the exactly equivalent equation is

    w_t = beta.grad w + kappa Delta w - 2 kappa epsilon grad(sigma).grad w
        +[epsilon sigma_t - epsilon beta.grad(sigma)
          +kappa(|grad sigma|²-epsilon Delta sigma)] w + f^epsilon S.

The diffusion equation is only an algebraic chain-rule test; it is not being
substituted for an Einstein equation. For `f=rho^s`,

    grad sigma = s n/rho,  Delta sigma=s/rho²,
    sigma_t=-s n.Xdot/rho.

Every inverse, gradient, Hessian and time term matters. The included tests use
both fixed centers and `Xdot=-beta0` with `beta=beta0+M(x-X)`. Because native
transport has the sign `+beta.grad`, this is the sign that cancels the leading
translation in the moving coefficient frame. Changing the center without the
factor's time derivative changes the represented equation. Fixed factors also
cannot remove the manufactured `p_dot log rho` term of a changing exponent.

This is an algebraic identity at positive rho; it is not an endpoint boundary
theorem. The original equation and boundary data must still be checked after
reconstructing physical fields. At an actual endpoint the original physical
fields may vanish and the transformed equation may require a different
limiting function space.

## Why a smoother stored field can give a worse physical discretization

Continuum product rules are not identities between independently differenced
discrete operators. A deliberately simple radial manufactured control exposes
the problem. For `U=rho^p`, `W=rho^m U`,

    Delta U = rho^-m[W''+(2-2m)W'/rho+m(m-1)W/rho²]

in three spatial dimensions. The script applies fourth-order centered first
and second derivatives separately to W, then reconstructs this expression.
All donor values come from the prescribed analytic scalar on the punctured
Cartesian line; no unknown is reflected and no endpoint parity BC is imposed.

For the entirely smooth physical `U=rho²`, the exact Laplacian is6. Direct
physical differencing gives6 exactly at the first positive point rho=h/2.
With fixed multiplication by rho^6, the stored W=rho^8 is also polynomial
smooth, but the separately differenced product rule gives132102 at that point
for **every** N=32,64,128,256. Its relative error22016 does not improve under
refinement there. At a fixed bulk radius the same control converges normally;
that bulk result cannot override the first-point failure.

The physical residual132096 stays unchanged while the residual multiplied by
rho^6 decreases from1.92e-6 at N32 to7.33e-12 at N256. A weighted stored-equation
residual alone could therefore claim apparent convergence while hiding the
failed physical equation. Full-domain physical residuals and point maxima are
essential. No core cell may be removed from acceptance because its weight is
small.

In contrast, **exact discrete conjugation** reconstructs U at every donor
before applying the original physical stencil. It gives6 exactly for this
control, independently of the stored power. This does not magically improve
the physical operator: for illustrative `U=rho^1.09`, direct physical
differencing and exact discrete conjugation both have first-point relative
Laplacian error about0.1403 at all these resolutions, while their fixed-bulk
errors improve. The original radial function has limited endpoint regularity.
In an actual Einstein equation its singular derivative can be multiplied by
vanishing physical coefficients, so the full combined physical RHS, not this
isolated Laplacian, is the relevant acceptance quantity.

This counterexample rejects the *naive separately differenced product-rule*
implementation, not every weighted-function-space method. A weighted basis,
consistent summation-by-parts construction, exact discrete conjugation, or
analytic cancellation of complete combined terms needs its own proof/tests.
It cannot be inferred from the smoothness of W or from a tighter linear solve.

## Shift scaling and uniform metric ellipticity

The current shift gauge has the flat principal reduction

    L0 beta = Delta beta + (1/3) grad div beta.

For a frozen positive factor f and `beta=f v`, exact substitution gives

    f^-1 L0(f v) = L0 v + 2 a^j partial_j v + (Delta f/f)v
       +(1/3)[(Hess f/f)v + (partial_i v^j)a_j + a^i div v],
    a=grad(log f).

For `f=rho^s`,

    a_i=s n_i/rho,
    Delta f/f=s(s+1)/rho²,
    (Hess f/f)_ij=[s delta_ij+s(s-2)n_i n_j]/rho².

Dividing by f preserves the flat principal symbol at every positive rho, but
introduces unbounded drift and inverse-square coupling. Keeping the undivided
equation instead multiplies its principal coefficient by f: it degenerates
as rho tends to zero for s>0 and becomes unbounded for s<0. Multiplying the
unknown by rho^s gives the same formulas with the opposite exponent in the
physical reconstruction. None of these choices supplies a missing endpoint
condition automatically. Beta is an algebraic stage solve here; no new beta
time-evolution law should be invented to account for interstage rescaling.

For the full variable conformal metric G, the original principal symbol is

    P^i_j(k)=a(k) delta^i_j+(1/3)G^ik k_k k_j,
    a(k)=G^ij k_i k_j.

It is symmetrized by the covariant metric G. Its eigenvalues are
`a,a,4a/3`; the script checks this on three nondiagonal/anisotropic SPD metric
examples. Uniform eigenvalue bounds on the **evolved conformal metric** give
uniform principal ellipticity. Neither chi nor alpha occurs in this symbol.
The Penrose/physical metric can be singular at the AF end without proving
degeneracy of this particular elliptic shift principal operator.

This does not prove full coefficient/source regularity. For bounded metric
jets, a source term differentiating `alpha*Atilde` has illustrative behavior
`rho^(q+b-1)` when alpha~rho^q and Atilde~rho^b. For q=1.09,b=0 it is only a
small positive fractional power; for q=2,b=2 it is power3. Those examples are
not universal puncture laws. Lower-order metric-connection derivatives can
also lose regularity even while G remains uniformly SPD. Report source jets,
coefficient regularity, original residuals and convergence separately from
principal ellipticity. Apply a variable-metric conjugation through the full
tensor jets; transplanting only the displayed flat corrections would omit
metric-dependent terms.

A vanishing-power beta scale also obscures translation. Physical
`beta=beta0+M y+...` may have a nonzero beta0. Dividing beta by rho forces its
stored value to diverge in that case; multiplying it by rho^s makes its stored
endpoint value zero and loses beta0 unless the leading weighted coefficient
is retained. Setting the zero stored value is not permission to pin physical
beta0. At scri, transformed vector data must be the actual required physical
shift divided/multiplied by the chosen factor, including its angular
dependence for a translated center, rather than independently homogeneous
potential data.

## Moving-center gauge incompatibility already identified

Moving-reference chain rules alone do not repair the current selected lapse
source at a translated puncture. `WormholeLapseRate` uses global z=|x|². Sending
the collapsed lapse and Khat to zero at X!=0 leaves

    z_X [1-(1-z_X)^4]/Omega_X

in its nonadvective lapse rate. It is nonzero for an interior translated
puncture. A rho²-vanishing lapse centered at X is therefore not preserved by
that source as written, independently of the chart used for storing it. This
is a gauge-compatibility issue for later moving/binary work, not a reason to
change the selected stationary gauge silently. The existing fixed-center
pilot and its small odd perturbation should retain their beta0 diagnostics;
the measured current dipole response is far too small to explain the dominant
transition error. No beta=0 pin or regular-center parity should be added to
hide it.

## Recommended fixed strategy and next gates

1. Keep the physical vector shift unscaled in the global coupled operator and
   in its acceptance residual. Preserve the evolved conformal metric's full
   coefficient corrections and diagnose its SPD bounds independently of chi.
2. Use exponent-independent positive/direct or logarithmic scalar variables
   with a fixed uniformly positive bounded normalization where useful for
   range. Keep the actual reconstruction/time/center derivatives. Compare
   physical-update and log-update time controls; this audit chooses no lapse
   floor or new gauge and proves no nonlinear positivity method.
3. Do not promote the current divide-by-rho^4/rho² chart or a universal angular
   atlas based on those powers. Preserve it as an early-regime control. Infer
   resolved power behavior and angular scatter from the physical fields, not
   from the assumed chart denominator.
4. Fixed rho^6 multiplication is a reasonable *smoothing hypothesis* for the
   listed scalar examples, but the naive operator test fails decisively. If
   pursued, first implement it as a reversible auxiliary representation with
   exact discrete reconstruction/conjugation and original combined-RHS
   residuals, then reproduce radial wormhole and trumpet-like regimes without
   excluding cells. Do not infer success from the weighted residual.
5. Only after those radial gates pass should a 3D angular treatment be selected.
   It must preserve independent Lambda/Z and all tensor degrees of freedom,
   accept changing/noninteger resolved regularity, retain the actual second
   infinity rather than a regular-center closure, and demonstrate convergence
   of physical quantities across interfaces and the puncture region.

This recommends a conservative fixed formulation with no exponent selected by
the expected answer. It leaves fixed overregularization available for a
properly discretized weighted method, while recording a concrete failure of
the tempting smooth-unknown/product-rule shortcut.

## Prepared exponent-independent scalar API (not yet built)

`src/z4c/hyperboloidal/positive_field_chart.hpp` now supplies a pure scalar
representation API, without changing a current gauge, stage solver or pilot.
Its three fixed functional choices are direct `F`, `log(F/R)` for a supplied
known positive reference jet R, and `F²` as an explicit overregularization
control. No radial exponent is a constructor parameter. Beta is untouched.

The API encodes/reconstructs full scalar value/gradient/Hessian jets and
converts physical/stored time rates. The log formulas include the supplied
fixed-x reference rate:

    ell_t=F_t/F-R_t/R,       F_t=F(ell_t+R_t/R).

This includes a known moving-reference contribution if present. The square
form uses `q_t=2 F F_t`; inverse derivative reconstruction uses `F=sqrt(q)`.
Its Hessian is evaluated without explicitly forming F³, which could underflow
before the stored F² becomes unrepresentable. `ReconstructionJacobian` reports
the absolute physical-value sensitivity:1, F, or `1/(2F)` respectively.

There are no floors, clamps, positive-radius excisions, or regular-center
rules. Operations return explicit statuses and leave their output unchanged
on failure. Zero-value inverse log/square derivative or rate reconstruction
returns `endpoint_data_required`: neither a logarithm nor a square root can
infer missing limiting spatial/time data from the zero value. Direct storage
can carry an explicitly supplied zero-endpoint jet. Negative physical values,
invalid references, nonfinite arithmetic, underflow/overflow and subnormal
scaled values are reported. The normal representable arithmetic range is the
supported contract; subnormal storage is not silently called accurate.

`tst/elliptic_shift_3d/test_positive_field_chart.cpp` is prepared for the root
agent to build/integrate. It has not been compiled or run by this audit agent.
It uses independent scalar AD in either spatial or time directions to check
F with powers2,4,1.09,1.5, angular anisotropy, translated moving centers and a
changing exponent. The same physical chi/lapse jets are reconstructed into
off-constraint, nondiagonal-metric tensor jets and compared through every
regular/pole `ConformalRHS` component, the assembled physical RHS, and the
unchanged selected `WormholeLapseRate`. It checks beta value/gradient/Hessian
identity and retains a deliberately omitted-reference-time control. Small
positive fields report reconstruction sensitivity; extreme square-storage
underflow/subnormal/overflow and extreme exponential failures must be explicit
expected failures rather than silently passing equivalence cases.

The fixed functional `F²` control is not an accepted regularization merely
because it is positive or smoother: a physical illustrative power1.09 becomes
2.18, which still does not provide the derivatives required for a universal
fourth-order endpoint stencil. Its inverse loses conditioning as F tends to
zero, and the endpoint limiting-data status must not be converted into a
regular-center closure. The tests establish representation equivalence only
after they run; radial evolution and combined physical-RHS/interface controls
remain separate gates.
