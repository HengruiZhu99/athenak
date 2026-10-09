# Held scalar-value prototype release recipe

This is a narrower first gate than PLAN.md. The implementation is source-only
and has not even been imported or syntax-executed. Numerical map derivatives,
the full inverse map, native target-time coverage, spectra, PDE evolution and
all native/kernel calls remain absent/held. The accepted/failed historical
files are untouched. Root must review exact source and recipe hashes before
an execution authorization exists.

`flat_ivp_values.py` imports only Python standard-library modules and mpmath.
It reimplements the fixed S1/a.5/.05-.95 reference formulas and the exact
native .2/.1 angular pulse; it imports no production or archived source and
invokes no executable. Historical files are hash dependencies only. The
80/110-digit arithmetic is not outward interval arithmetic and does not make
the convergence checks a rigorous global caustic exclusion.

The initial version computes only the four scalar values u=deltaY and
phi=u/Omega_event at fixed *reference* events. Reference time tau is explicit
in every event. The two native failure positions and displayed times occur as
reference-coordinate query labels, not as the full native inverse target
events. No plot/receipt may relabel them as a reconstructed native metric.

## Fixed mathematical implementation

* The complete logistic cutoff and Omega' use the smaller exponential and
  independently evaluated logistic complement. Core/outer branches are exact.
  Omega, h and L must be positive for every r<1 query, without floors/clamps.
* The bounded height defect D_h is integrated from -r0 using
  -L/[h(h+b)] over the fixed transition panels in the JSON recipe. The outer
  constant matches at r1; the outer defect uses the analytic rationalized
  square-root expression. Each precision and declared height quadrature order
  computes its own height, rather than treating a previous table as exact.
  The reference cache is bounded at16384 entries and the defect cache at4096;
  this limits retained quadrature intermediates without changing arithmetic.
* Coarea endpoints solve the reviewed monotone equations in compact r by
  bisection on [0,1], with r=1 an analytic bracket only. The fixed iteration and
  residual gates are in the source/recipe. Every integral is split at the
  declared layer panels lying inside its source interval. Gauss nodes exclude
  endpoints; a cosine outside [-1,1] is a failure, never clipped.
* All four initial Pi=s/Omega^2 components are evaluated with a factored
  f/Omega^3. In the exact outer branch this is (2a)^4*Omega*exp(-r^2/.35^2).
  The compact coarea weight is r*L*Pi/h; no separately huge physical s or
  height subtraction is needed. The field normalization and radial q factor
  follow KIRCHHOFF-PENCIL.md.
* The exact event center has a separate sphere-average formula. The ray
  comparison uses an independent null-intersection parameterization, optional
  fixed normal-aligned boost, and the nonnegative denominator
  h*K=Omega^2*k0/(h+b)+b*|kspace-k0*nu|^2/(2*k0). This avoids aligned-ray
  cancellation. It is not a new damping/regularity prescription.
  Its center has one shared geometric root; exact-core sources use the flat
  ray root. For an event whose computed full source interval is outside r1,
  the independently derived outer ray root is
  ell=[(T-C)^2-Re^2-a^2]/[2*((T-C)*k0-X.dot(kspace))]. The actual graph/root
  equation is still checked at every ray. General layer rays retain fixed
  bisection, but the first recipe does not request that expensive family.

## Fixed samples and refinement

The JSON recipe pins eight reference events: zero center/outer controls,
native pulse at center/core/transition/outer, and the two stopped RWM failure
coordinates/times as reference queries. Independent ray comparisons cover
center, exact core, and a very short outer event whose light-cone source stays
in the exact outer region. None is a replacement for full future target
coverage. Runtime reports retain the actual source bounds.

For each of 80/110 digits, full coarea rules16/32/64/128 refine radial and
periodic azimuth orders together. Separate (128,64) and (64,128) rules measure
the two refinements independently. The two last controls use the same height
order128 as the full128 rule. Height comparison uses full64/full128. Ray
orders are the fixed16x32,32x64,64x128 sequence. There is no after-the-fact
selection of a smaller error or an automatic additional refinement.

Initial-data checks use the exact normal formula, an independently assembled
embedding E_i and reciprocal-lapse formula at seven fixed finite points, and
an actual4x4 determinant compared with h/alpha. All four amplitude controls
(.2,.1),(.2,0),(0,.1),(0,0) are included. This comparison is not a source/RHS
gate and has no numerical time derivatives. Stronger scri tail comparisons
are deferred; the factored evaluator itself works for finite r<1.

The flat constant-velocity controls integrate the coarea/ray sphere measure
and compare with u=v*T, including center and Re<T/Re>T. Pure-CMC controls use
the independently derived zero-Dirichlet l=0,1,2 exact solutions. They are
nontrivial scalar quadrature checks, not the native initial data. No old dipole
is imported.

## Declared numerical thresholds and failure handling

All comparisons use max-component |a-b|/max(1,|a|,|b|), with the complete
comparison operands also retained in the saved rows. Last full coarea refinement and each separate
radial/azimuth refinement must be <=1e-10. The exact scalar controls and
ray/coarea comparisons must be <=1e-10. Identical-rule precision comparison
must be <=1e-30; initial normal/determinant identities <=1e-50; height
full64/full128 <=1e-25. Roots use compact bracket width1e-55, at most512
iterations and absolute equation residual <=1e-40. Failure is retained and
does not authorize tolerance relaxation. Convergence at these rules is finite
event numerical evidence, not an interval error certificate.

Coarea/radial/azimuth convergence, identical-rule precision, ray convergence,
ray/coarea agreement and exact-zero controls apply independently to both
physical u=deltaY and conformal phi=u/Omega_event, with those same thresholds.
All operands in both normalizations are retained. Initial s/Pi/det checks and
the independent exact physical-wave controls retain their displayed meanings.

The future runner must use the recorded Python3.9/mpmath1.3.0 interpreter,
-B, PYTHONDONTWRITEBYTECODE=1 and a fresh nonexistent output directory. The
root authorization must set scalar_values_execution_admitted=true and bind
exact absolute source/recipe/VALUES-PLAN hashes and that fresh path. Full
source/runtime/mpmath package pins are verified before and after. Partial
rows are append-preserved as whole JSON snapshots after each event/control
group. Any exception leaves a failed receipt and logs in the new attempt.
Source/runtime drift fails even if the mathematical comparisons passed.

The recipe records a held command only. Source preparation itself has used
only reads, pencil derivation, file writes and metadata hashing; no Python
import of this prototype, mpmath arithmetic, numerical batch or source query
has run. A later launcher must separately record exact outer argv/env,
stdout/stderr/exit and source-before copies before executing this file.

## Next gates remain separate

After scalar values pass, a new source/recipe must derive differentiated
coarea endpoint/azimuth limits and independent ray derivatives. Only then can
J, its singular values, target-time causal margin and the full4D inverse map
be queried. Global injectivity/coverage needs further estimates. The old
native scalar matrices can subsequently compare the same IVP at matched
physical events; their failure alone is not a continuum caustic verdict.
