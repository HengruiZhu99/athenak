# Independent source review of Gaussian physical-event screen v1

Disposition: the mathematical formulas and bounded grid are consistent with the
reviewed pencil. Execution admission fails because the consumed interpreter and
mpmath import are not bound to the expected hashed paths. No reviewed source was
imported or executed. This is a source/math/metadata review, not a numerical
validation of its outputs.

The reviewed source index is
`b7cfe471f620e0b9b261fea8ad7b8be98d6f9437921c28ce5157217b1915cd3a`;
screen source is
`070c0f296102b8f07af09f2b6a3533b05100e759ee6c26719d501740edd1ebcb`;
recipe is
`1cf3e91ddb86db89a794c5693af1c78b20fe8b57aa3391e2558d1f31b2286da5`.
All five original preparation/source/index files were copied before review.
The standard-library metadata readback rehashed 100 unique original paths,
including the source index and 95 runtime/context pins, with no differences.
Only AST parsing and exact rational grid counting were executed.

## Formula and coverage review

For `f(T)=sigma^4 exp(-T^2/(2 sigma^2))`, the code uses the probabilists'
Hermite recurrence `H_(k+1)=x H_k-k H_(k-1)` and
`f^(k)=(-1)^k sigma^(4-k) H_k(T/sigma) exp(-T^2/(2 sigma^2))`.
Its derivative-list indices, powers, and signs agree with this identity.

The regular origin series coefficient is
`-8(j+2)(j+1)/(2j+5)!`; its first terms are
`C=-2 f5/15-R^2 f7/105-...`. The C_T and C_R series differentiate this same
expression in T and R. The exact-origin branch retains C and C_T and sets C_R=0.
For R>sigma/8, direct differentiation of the advanced/retarded C gives

```
C_R=-(f3(u)+f3(v))/R^3-6(f2(u)-f2(v))/R^4
    -15(f1(u)+f1(v))/R^5-15(f0(u)-f0(v))/R^6,
u=T-R, v=T+R.
```

These are the source's coefficients and signs. The declared 40-term/80-digit
and 60-term/110-digit comparison is a numerical consistency test, not a
rigorous origin-series remainder enclosure.

At fixed physical T,R, let p=n1*n2 and s=n1^2+n2^2. The code's scalar quadratic
is the exact reduction of

```
D=d0+2 epsilon R^2 p [C_T+h(C_R+2C/R)]
  +epsilon^2 {R^4 p^2 [C_T^2-C_R^2-4C C_R/R]-R^2 C^2 s},
d0=a^2/(R^2+a^2), h=R/sqrt(R^2+a^2).
```

The s coefficient is nonpositive. For every p in [-1/2,1/2], a point with s=1
exists, so minimizing over the sphere reduces exactly to both endpoints and a
convex interior vertex, when present. Concave and linear degeneracies require
only endpoints; the epsilon=0 case also follows. The chosen Cartesian point
`n1=sqrt((1+sqrt(1-4p^2))/2), n2=p/n1, n3=0` has norm one and product p.
The separately written gradient at that point is
`R C (n2,n1,0)+R^2 p C_R n`; substituting it into
`J^2-|h n-epsilon grad F|^2` independently checks the angular algebra. It is
not an independent oracle for C or its derivatives, because both calculations
consume the same radial values.

The fixed exact-rational union of ordinary and retarded times has 3,504
radius/time events per profile. Two sigmas and four epsilons give exactly
28,032 records at each precision, 56,064 total. The two runs iterate the same
sorted exact Fraction times. Pairing checks profile, radius, branch and time;
it compares D, D/referenceD and J with the unchanged 1e-60 scaled threshold.
The angular-identity threshold remains 1e-65. Positivity is reported separately
and is not part of the identity/precision pass, so negative D samples are
retained. No parameter, grid, tolerance or scientific formula change is
recommended by this review.

## Blocking execution binding

The authorization guard correctly binds the screen, local recipe and source
index, rejects optimized Python, requires bytecode-off environment, checks
inputs before/after, and requires a fresh direct attempt directory. However,
`python_runtime_path` is hashed as an ordinary input without comparison to
`sys.executable`; `import mpmath` is not checked against the pinned module
origin. A different interpreter or package can therefore consume unchanged
expected files and still reach the numerical body. Root additionally reports
that the pinned CLT Python3.9 under `-I` does not find the pinned venv mpmath
without an explicit package-parent path. Hashing that venv's files cannot fix
Python's import search path. This is an execution prerequisite defect, not a
Gaussian formula defect.

Preserve this v1 source/index/recipe unexecuted. A separately pinned guard-only
sibling should verify the resolved interpreter's exact digest, add only the
known pinned mpmath package parent before import, verify the imported origin,
and enforce the reviewed isolation/environment settings. Its scientific body,
grid, thresholds, and negative-result preservation should stay unchanged.
The exact external launcher must also retain failures before the inner attempt
wrapper is entered.

## Scope

Even a positive finite screen is not a proof for all R,T>=0. These are physical
events on the pure-CMC endpoint. At fixed native time the inverse T depends on
angle; the screen does not solve it, establish the native event domain, or
construct complete ADM jets. Neither a positive screen nor a negative sample
alone admits a native slicing conclusion. No PDE, source-kernel, operator,
growth, caustic, Scri, inner-trumpet or black-hole statement is certified here.
All existing frozen evidence and source bytes remain unchanged.
