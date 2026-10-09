# Held finite native-time Gaussian inverse/value screen

This is source-only preparation. No source in this folder has been imported or
scientifically executed. Exact root review and authorization are required before
mpmath import, height quadrature, inverse solve or ADM value evaluation. The
source/index/recipe must remain fixed through any released single invocation.

The manufactured map is the independently reviewed pure-time physical map
`Y0=T+epsilon F(T,X), Yi=Xi`, with
`F=partial_X1 partial_X2 [(f(T-R)-f(T+R))/R]` and
`f(T)=sigma^4 exp(-T^2/(2 sigma^2))`. The retained Minkowski native reference
is exactly S1,a.5,geometry .05/.95,H(0)=0. This is not the original native lapse/
shift pulse, not an inner modified-BM solution, and not a PDE/evolution test.

## Fixed source and grid

`values_context.py` is byte-exact source89c96ee729b8cc741757740534566c9557dd6fdce2e66213c2fe8ff1c7e8ffe7.
Only its Layer/reference/defect/height and Gauss routines are consumed. The
Gaussian `derivatives` and `radial` definitions are extracted byte-exact from
accepted v2 screen6c9cae79. Their unchanged nested definitions are wrapped in a
function that supplies the already admitted mpmath context. No old scalar pulse,
Kirchhoff integral, kernel API, source equation, matrix or evolution is called.

The exact root-requested native radii are 0,.025,.05,.1,.2,.3,.45,.5,.65,.75,
.84,.85,.9,.95,.98,.995,1-1e-3,1-1e-6,1-1e-9,1-1e-12,1-1e-18.
Native times are 0,.05,.1,.2,.5,1,2,4,6. Polarization p=n1*n2 is
-.5,-.375,-.25,-.125,0,.125,.25,.375,.5. Profiles are sigma7/20,1/2 and
epsilon0,1/4,1/2,3/4. This gives 13,608 events per level. All four combinations
(80digits,height128),(80digits,height256),(110digits,height128),
(110digits,height256) are retained, with 40/60 origin-series terms respectively:
54,432 total records. No adaptive domain extension or sampled minimization is
performed. Root's fixed grid is not narrowed.

At a fixed native time/radius/p, F depends on angle only through p, so the
inverse T also depends only on p. For that fixed p, the exact nonpositive
`-epsilon^2 R^2 C^2 s` contribution is minimized at s=n1^2+n2^2=1, which is
attainable for every declared p. A real n with s=1 and product p is used.
This is an exact reduction in s, not a continuous minimum over p or native time.
The prior fixed-physical-event sphere minimum does not give this inverse event
domain; in particular its reported T0 negative cases have t_native=-H(R).

## Inverse equations and gates

Write R=r/Omega and h=H_R=b/alpha_hat. The accepted height helper computes
`H=R+defect`, with fixed Gauss panels .05,.15,.3,.5,.7,.85,.95 and the analytic
outer defect. Full inverse data use Y0=t_native+H. Since the even Gaussian has
F(0,X)=0 and `J=1+epsilon F_T>=1-4epsilon/pi>0`, Y0>=0 implies T>=0.
Core/transition roots use the explicit bracket
`max(0,Y0-16epsilon sigma/(3pi)) <= T <= Y0+16epsilon sigma/(3pi)`.
It follows from the global |F| bound using M3<=4sigma. Structurally zero F
(R0,epsilon0,p0 or zero target Y0) admits the collapsed analytic inverse branch.
Both endpoint signs and residual are still saved.

For r>=.95 use z=Omega/r and the compact pencil:

```
s=t_native+C_H, u=s+z c_ret, v=u+2/z,
Phi=f2(u)-f2(v)+3z(f1(u)+f1(v))+3z^2(f0(u)-f0(v)),
c_ret+epsilon p Phi=a^2/(sqrt(1+a^2 z^2)+1).
```

The bracket center is its right side and its radius is
`|epsilon p| [2sigma^2+6|z|sigma^3+6z^2 sigma^4]`, from exact bounds on f0,f1,f2.
The derivative in c_ret is the same J. This solves a bounded unknown instead of
subtracting T-R or H-R in the primary root. Advanced terms are retained.

The fixed solver permits 16 safeguarded Newton iterations then at most 512
bisections. Initial and final endpoint signs, absolute residual<=1e-50 and
width<=1e-55 are required and saved. Once residual<=J_lower*width_tolerance/8,
it may test a local quarter-width bracket around the Newton point; it accepts
that bracket only after explicitly evaluating both signs. This avoids many
unnecessary bisections without changing root thresholds. Numerical sign brackets
and quadrature are not formal interval enclosures. Collapsed analytic branch
width0 is bookkeeping, not a new interval certificate. At most 528 iterations
are allowed per nonanalytic root; exact cost is unmeasured and every root saves
its actual evaluation count. No D/Delta/derivative floor or clamp is present.

The original time-map residual is separately evaluated: direct core/transition
`T+epsilon R^2 p C-Y0`; at outer points the independent radial C and direct
high-precision T-R are used against the accepted stable height defect. Its
absolute maximum also must stay<=1e-50. Primary inversion remains factored.

## Compact determinant and ADM values

The compact outer expressions retain all advanced and retarded contributions:

```
Aplus= -f2(u)-6z f1(u)-9z^2 f0(u)
       -2 f3(v)/z+7 f2(v)-12z f1(v)+9z^2 f0(v),
Aminus=2 f3(u)+7z f2(u)+12z^2 f1(u)+9z^3 f0(u)
       -z f2(v)+6z^2 f1(v)-9z^3 f0(v),
J=1+epsilon p z(Aminus+z Aplus)/2,
wr=h+epsilon p z(Aminus-z Aplus)/2,
wT=-epsilon z^2 Phi grad_S p,
eta=a^2/[sqrt(1+a^2z^2)(sqrt(1+a^2z^2)+1)],
kminus=eta+epsilon p Aplus, jplus=1+h+epsilon p z Aminus,
Delta=kminus*jplus-epsilon^2 z^2 Phi^2(1-4p^2), D=z^2 Delta.
```

`D/referenceD=Delta/(a^2 h^2)` is the primary outer value. An independent
original-vector gradient construction using the extracted radial C/CT/CR and
direct high-precision T±R must agree with this normalized value at scaled1e-30.
At the most compact radius the 80-digit direct subtraction still serves only
as a diagnostic cross-check; it is not the compact primary algorithm.

When D>0 only, source computes conformal alpha, beta, bar-gamma, its determinant,
chi and gtilde using the compact pencil at outer points and the native radial
coordinate transform in the core/transition. Conformal alpha/alpha_hat is saved.
For D<=0 these ADM fields are null with explicit reason; negative D events stay
in the output and do not fail the independent consistency check. No complex
square root, sign replacement or positive floor is used. No K/P/A/Lambda jets
or time rates are constructed. The original reference is retained throughout.

## Comparisons, outputs and admission

Precision is compared at each fixed height rule; height128/256 is compared at
each precision. Fixed source metrics are D/referenceD,J,T,u,root unknown and
alpha/alpha_hat when defined. All scaled differences and the height-constant
comparison require<=1e-30. A disagreement in whether the positive-D branch is
defined fails consistency. Identity/precision/root PASS remains separate from
all_sampled_D_positive. Full records are append-only samples.jsonl, always a
large payload for future compact collection; profile minima and worst event
metadata are retained. All negative rows are saved. Partial files and true
failure/progress/pin receipts remain on any stop. No successful run exists yet.

Before scientific imports the inner guard requires exact authorization/source/
local recipe/index hashes, actual resolved interpreter SHA, isolated unoptimized
-B state, fixed environment, and all before/after pins. It inserts the exact
pinned mpmath parent and verifies imported origin, then adds the fixed local
helper directory and verifies both helper origins. The stdlib outer forces
the exact runtime/-I/-B/environment, refuses existing child/outer paths, binds
its own authorization digest, and preserves pre-child and child failures.
No prior frozen source, failed process, archive, or production file is edited.

Positive finite samples are not a proof over the full native domain, continuum
admissibility, inverse uniqueness beyond the analytic J result, radiation,
Scri regularity, PDE stability, caustic avoidance for another profile, or BH
evolution. A later single BH still must survive wormhole-to-trumpet transition
with the Minkowski hyperboloidal reference retained; this manufactured scalar
value screen does not supply that gate.
