# Source-only interval implementation and proof obligations

This sibling implements the algorithm proposed in frozen plan index
6686e3eea4ca73d55f2bcf1617c2b91cc332797ce8f608af9f13da031465909a.
No module has been imported, no unit or domain arithmetic evaluated and no
certificate produced. Only standard-library text, AST and hash preparation
is permitted before root review. The false external authorization template
does not release any stage. Every future stage has a fresh output directory.

The target remains sigma=7/20 or1/2, a=2 and |epsilon|<=3/4. A compact
epsilon=3/4 full-sphere endpoint certificate combines with the separately
pinned regional and concavity proofs, rather than proving a PDE statement.
The actual layer need not be CMC at R=4. No native or BH adoption is implied.

## Outward arithmetic

All source endpoints are Fraction rationals. For nonzero x define
e=floor(log2|x|) by integer shifts/comparisons and step=2^(e-bits+1).
Down/up rounding is floor(x/step)*step or ceil(x/step)*step. Negative values
use the same signed integer floor, with no float or finite exponent range.
Each elementary interval operation is evaluated exactly on rational
endpoints and then rounded outward. Multiplication uses all four endpoint
products; a crossing-zero square has lower endpoint0. Division rejects a
zero-containing interval and encloses the monotone reciprocal before
multiplication. Exact zero has no artificial floor. Signed IEEE zero is
irrelevant to these rational set enclosures.

For sqrt(x), choose a dyadic step at sqrt exponent floor(e/2), and put
k=isqrt(floor(x/step^2)). The integer identity
k^2<=x/step^2<(k+1)^2 proves the enclosure. Equality is detected by an exact
numerator/denominator comparison; otherwise the upper endpoint is k+1.

For exp(-y), y>=0, reduce y=2^m z with0<=z<=1/16. The alternating series
has decreasing terms. Its odd degree65 sum is a lower bound and its even
degree64 sum an upper bound, with even omitted-term bound z^65/65!.
The two rational endpoint sums are outward rounded and squared m times,
using positivity/monotonicity. For an interval [ylo,yhi], exp(-yhi) supplies
the lower endpoint and exp(-ylo) the upper. There is no uncertain exp call,
mpmath acceptance or silent exponential underflow. The fixed range-reduction
limit64 is a guard, not a proof of an event outside the declared domain.

## Taylor parity and independent function structure

For C, Taylor f5(T+Rv) through degree2K+1 in the symmetric integral. All odd
terms integrate to zero, and the order2K+2 remainder uses f^(2K+7). CT raises
that index to2K+8. For Crho, start from
-(16R)^-1 integral(1-v^2)^2 v f6(T+Rv)dv and Taylor f6 through degree2K.
Only odd retained terms survive. The degree2K term is even and integrates to
zero after the prefactor v; the remainder order2K+1 uses f^(2K+7). Applying
the explicit moment16/[(2K+3)(2K+5)(2K+7)] before division by2R gives the
regular R^(2K) bound. The producer scales C,CT,Crho by sigma,sigma^2,sigma^3
and uses forward powers plus the Hermite recurrence. The replay uses
ordinary-dimensional derivatives, explicit integer Hermite coefficients,
parity Horner polynomials and descending rho Horner. Both have rigorous
remainders from the pinned pencil. Neither forms a tiny-R advanced/retarded
difference in the regular branch.

On R>=sigma the producer uses sums of inverse powers; replay uses common
positive denominators. Their angular coefficients are grouped differently,
and the replay minimizes its independently enclosed quadratic by checking
endpoints and the permitted vertex. Shared interval primitives are an
explicit limitation: replay is independently structured model/coverage
validation, not an independently written interval arithmetic library.

## Separately admitted stages

1. `units`: a fixed144 exact controls. Directed rounding/enclosure and
   dyadic serialization; products/squares/invalid division; integer sqrt
   inequalities; exponential containment of separately computed degree
   200/201 rational bounds; K32 remainder inequalities; exact polynomial
   moments and rho derivative coefficients; origin parity/CT identity;
   all quadratic-minimum branches. The degree200/201 reference is rounded
   outward at512bits between squarings to avoid huge exact denominators.
   No domain box or numerical scientific payload is evaluated in this stage.
2. `certificate`: only after the same source-index's successful units
   receipt. Four exact roots cover two profiles and the two R branches
   [1/50,sigma],[sigma,4], each with T in[0,4+8sigma]. Producer leaves require
   a strictly positive rational lower bound. The optional two diagonal
   sphere-tail lemmas are checked by exact cell inequalities. Otherwise
   mlo/qlo/Lhi and the exact quadratic minimum are enclosed and recorded.
   The longer coordinate width is bisected at its exact rational midpoint.
   All roots/children are closed cells, so their shared edges cause harmless
   duplicate coverage, with no gap or unproven edge.
3. `replay`: separately released only after successful units and producer
   receipts with exact source and certificate hashes. A LIFO reconstruction
   requires every node in deterministic DFS order, verifies all exact
   subdivisions, consumes every root and rejects missing/extra records. It
   rebinds the producer's exact mlo/qlo/Lhi witnesses, then separately
   requires its own higher-precision grouped enclosure to be positive on
   every nonregional leaf. If its broader enclosure cannot prove a leaf,
   replay fails/returns unresolved; it does not adopt the producer's bound.

The producer precision is256bits and replay384bits, K32. Domain maxdepth60,
maxleaves2^20, output64MiB and time600seconds are all fixed. Replay has a
900second limit. Reaching any limit is an unaccepted partial result, with
the exact partial file, stderr and source/input identity retained. These
budgets deliberately make a first attempt bounded; this source does not
promise termination. The one-shot wrapper has longer hard stage timeouts
to capture reports/failures and performs complete before/after pin checks.

The certificate is scalar UTF-8 JSONL and may exceed repository file policy;
it must remain external/metadata-only in a later compact archive. To bound
output it records exact outward dyadic coefficient/lower witnesses and
tree paths, rather than all internal Gaussian arithmetic endpoints. The
independent replay re-derives the full function/remainder enclosures from
each exact cell and pinned source. This is the implementation's compact
realization of the plan's full enclosure/coverage obligation. Both reports
and final interpretation must state this choice.

Only the conjunction of accepted producer, independent replay and reviewed
regional pencils permits the manufactured slicing conclusion. A producer
PASS alone has `global_slicing_acceptance=false`. Positivity is physical
event admissibility; R^2D controls the exterior, not a uniform lower bound
on unweighted D. No source, jets, continuum/global evolution, original pulse
or wormhole-to-trumpet conclusion is released by these stages.
