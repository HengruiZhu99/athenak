# Proposed finite-point third-jet and actual-RHS gate

Status: **SOURCE/PENCIL ONLY, not an execution recipe or release**. The
DERIVATION.md specifies the complete geometry and consumed derivatives. This
plan fixes a candidate oracle coverage before implementation, and states the
additional admission decisions needed for an independent binary64 binder.
No source here imports numerical packages or performs a scientific calculation.

## Candidate local coverage

Retain the exact reference S=1,a=1/2,geometry cutoff(.05,.95),H(0)=0.
Use the same 21 exact decimal radii as the completed native inverse screen,
plus .05−1e−6,.05+1e−6,.95−1e−6,.95+1e−6. Use native times 0,.1,2,6.
For each p in −.5,−.375,−.25,−.125,0,.125,.25,.375,.5 use
n1=sqrt[(1+sqrt(1−4p²))/2],n2=p/n1,n3=0. Add n=e2,n=e3,
n=(1,1,1)/sqrt3 and n=(2,−3,6)/7. This is 13 directions at nonzero
radius; the exact origin is included once per time. With 25 radii there are
24*4*13+4=1,252 spacetime points. At every point construct sigma=7/20,
epsilon=0 and3/4: 2,504 oracle cases at each precision.

Use two complete geometry precisions, proposed110/150 digits, with accepted
height128/256 comparison retained separately. Include one sigma=1/2,epsilon=3/4
negative control at r=.75,t=.1,n=(1,1,0)/sqrt2. Save its J/D and expected
ADM refusal, never attempt its metric square root. This gives 2,505 records
per precision including refusal, 5,010 records total. All directions and
coordinate conventions must be written as exact formula/exact decimal strings
in a source-pinned registry before any implementation execution. Root can
review or replace this proposed bounded coverage before it becomes a recipe;
no existing grid or threshold has been changed.

The two exact layer endpoints and neighboring points test cutoff tails and
branch continuity. The exact origin tests the nonzero quadrupolar Hessian.
The off-plane points test full Cartesian angular tensor jets, not merely an
s=1 radial reduction. The near-scri sequence tests the compact formulas at
finite positive Omega; no point at exact scri is included.

## Oracle implementation requirements, still held

Use complete multivariate ordinary jets or Taylor coefficients with explicit
factorial conversion and checked derivative orders. Preserve mixed time-space
coefficients and implicit-root derivatives through total order three. The
accepted value-only radial functions provide context, but their value/CT/CR
interface is insufficient by itself: analytic profile derivatives and the
complete origin/compact formulas in DERIVATION.md are needed. No nested FD,
finite-radius floor, dropped advanced tail or absent derivative padding.

The primary outer construction uses the bounded c_ret implicit jets and
factored conformal ADM fields. A separately implemented physical graph
composition supplies the independent finite-radius metric/K comparison.
At the origin use the exact Cartesian polynomial/integral representation.
Transition H derivatives are analytic; height quadrature supplies only H's
value. The prior source values_context89c96 is an unchanged reference-height
context, not an automatic third-jet backend. A future implementation must
expose and pin the derivative formulas it adds.

Before constructing ADM data, require J>0 and D>0 (outer Delta>0), with no
clamp. A negative event is a retained refusal, not a failed algebra identity
and not an allowed source query. Report analytic global J bound separately
from finite sampled D positivity. The completed sigma=.35/.75 grid positivity
is not a proof over the new angular or continuous domain.

Independently check at each valid event:

* inverse value/first/second/third chain-rule residuals and all derivative
  permutation symmetries;
* scalar Box_eta F=0 from analytic scalar jets;
* the physical four-metric from X_,a, and its first/second derivatives from
  X through3, versus reconstructed ADM metric and compact primary fields;
* full4D Riemann/Ricci from those metric jets, physical wave-map connection
  difference and harmonic coordinate scalars;
* physical ADM H/M, Theta and Z, determinant and A-trace identities from the
  exported consumed jets; input det=1 and trace=0 are never projected;
* compact versus graph values/jets where the graph representation has usable
  precision; precision110/150 and height128/256 discrepancies separately;
* epsilon0 identity with the retained complete Minkowski reference, including
  reference time rates0. The manufactured core is generally not constant.

Proposed analytic scaled identity/precision tolerance is1e−55; the height
comparison remains its previously fixed1e−30. Retain absolute residuals,
separate term scales and derivative orders so an unscaled physical tensor
with powers of Omega cannot be silently relabeled as a bounded conformal one.
Near-scri raw graph cancellation may fail a proposed gate; preserve the first
failure and require a fresh reviewed change, rather than lowering a tolerance.
These proposed tolerances must be reviewed with source/runtime/recipe before
any computation. No current execution is authorized by this note.

## Payload schema and independent root binder

Follow the frozen prior ordinary-jet schema: spacetime axes(t,x,y,z), multiindex
and ordinary decimal coefficient; physical metric order2; inertial inverse and
reference embedding order3; alpha/beta/chi/gtilde/Omega order2;
P/A/Lambda/Theta order1; independent raw22 time values. Retain complete4D
metric and independent graph/compact residuals as separate records. All22
expected rates come from oracle geometry, never from ConformalRHS or Gauge.

A future root-owned binder must map the explicit raw22 order from
DERIVATION.md, use actual C0 ConformalRHS with runtime input kappa10
(argument10/alpha) and kappa2=0, and the unchanged physical-reference RWM
helper56d61c56. Assemble each regular/pole pair once; no projection or live
reference subtraction in the tested equation. Record the stationary Omega
normal and its gradient from the submitted live alpha/beta. Missing
P/Lambda/Theta second jets use unavailable sentinels, and ASan/UB must not
consume them. Compare geometric/gauge rates, full physical8 constraints,
input determinant/A-trace and their time tangent, reference connection and
scaled4D source with independent oracle expectations. Release and ASan/UB
build outputs must match for identical fixed inputs; compiler/dependencies,
source wrappers, payload conversion and attempts are separately pinned.

Report both raw error e(u)=RHS_native(u)−u_t and the explicitly paired
reference diagnostic e(u)−e(uref), with e(uref) itself retained. Reference
subtraction is an attribution diagnostic, not an altered RHS or a replacement
for raw live error. Keep regular numerators, pole numerators, assembled values
and Omega-scaled errors distinct. A scaled residual or paired cancellation
cannot convert a failed raw assembled comparison into a PASS.

## Binary64 representability and cancellation admission

The high-precision registry is not automatically a native binary64 registry.
For example, the decimal radius1−1e−18 rounds to1.0 in binary64. It therefore
cannot be passed to a finite-Omega native kernel as an interior radial point.
Retain that event in the analytic compact-jet oracle, and explicitly exclude
it from any actual finite-interior kernel acceptance matrix. Every future
binary64 coordinate must be separately pinned in hexadecimal, with the oracle
regenerated at those exact Cartesian coordinates and actual native Omega>0.
Angular rounding changes the actual radius; nominal r is insufficient.

Native LayerPoint/reference jets are authoritative for an actual-native binder.
Record all submitted-minus-native Omega derivatives and reference coefficients,
including relative/term-weighted differences when Omega is small. A fixed
absolute Omega difference alone cannot establish consistency of an equation
containing Omega^-1 and Omega^-2. If native/analytic reference rounding becomes
large relative to Omega, do not silently replace one reference, transplant
unrelated reference jets, enlarge a source tolerance or call the event exact.
It is a separately reported coefficient/precision limit.

The earlier finite-point binder's5e−9 raw22/G/scaledF and physical8 gates,
5e−11 normals gates and2e−10 absolute Omega gate provide a conservative
starting protocol only on a reviewed finite-Omega subset. Before actual
source queries, root must fix the native coordinate registry and either prove
its native-reference error is sufficiently small for those unchanged gates,
or explicitly declare a separate near-scri diagnostic with no original gate
admission. All representable near-scri failures remain visible. No query at
Omega0, floor, omitted negative event or a posteriori tolerance selection.

## Limits and sequence

First review the pencil and fixed registry; then prepare and review oracle
source/runtime/admission without numerical execution. Only after an independent
oracle PASS may root prepare a separate exact binder recipe and release its
finite-Omega point calls. No mesh stencil, ghost fill, native propagation,
finite-time stability, caustic avoidance, global slicing theorem or BH
acceptance follows. The sigma=.5/.75 future slicing failure remains explicit.
A later single BH must still survive wormhole-to-trumpet transition with the
Minkowski hyperboloidal reference throughout; this manufactured physical-RWM
family supplies neither that inner gauge nor BH data.
