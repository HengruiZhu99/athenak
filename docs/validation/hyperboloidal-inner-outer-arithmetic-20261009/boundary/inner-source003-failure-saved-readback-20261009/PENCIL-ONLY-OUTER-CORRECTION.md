# Source003 failure: immutable outer arithmetic, not an inner correction failure

This is an algebraic proposal only. No helper implementation, compiler,
oracle/target recomputation or scientific query belongs to this saved readback.
The actual source003 Release remains FAIL and the original arithmetic supplement
remains held and ineligible under its original full-PASS prerequisite.

The original oracle recorded 1,872 source003 failures: 1,104 source-parts and
768 source-rhs. Native binary64 values and the original update ordering identify
every failure uniquely. All 1,872 were already source002 failures; 1,364 old
failures disappear and no new failure signature appears. Every remaining failure
is in one of 288 query rows with W exactly1. At each row, all eight returned parts
are bitwise identical to the frozen RWM baseline, and each failing native value is
bitwise identical to its source002 value. All recorded failures have fixed radii
.84/.85/.9/.95/.98/.995; the float cutoff already returns W1 at .84.

The remaining family counts are small-alpha-large-chi1200,
large-alpha-small-chi336, and chi-gradient-contrast336. Only beta rows fail.
The first source003 failure is sources.jsonl line684: a=.5,r=.84,direction0,
G0=.375, alpha1e-150, chi1e60, regular beta_x. The saved native value is
-9.290492122249053e43 and the saved high-precision target is6.107017700426954.
This is the tiny-alpha/large-chi family, not the large-alpha family. The worst
saved scaled parts discrepancy is a=1,r=.84,direction1,G0=.375, regular beta_z,
line1888; all these discrepancies round to1 in the compact reported maxima.
The complete original error strings and all exact query labels are retained.

Consequently no inner004 change can repair the remaining old tests while keeping
the required `if(W==1) return frozen_rwm::Gauge(...)` bitwise contract. The
recorded source003 oracle has no failed W<1 comparison, but this is a qualified
statement about the fixed completed test suite, not overall helper acceptance,
arbitrary-state correctness or a changed old PASS status.

## The algebraic source of the outer defects

Use a=live alpha,h=reference alpha,x=live chi,y=reference chi, G=live inverse
gtilde,Gh=reference inverse gtilde. The frozen RWM deviation graph contains

    da2=(a+h)(a-h),
    dC=(x-y)G+y(G-Gh),
    dV=da2*x*G+h^2*dC.                                    (1)

Its real value is a^2*x*G-h^2*y*Gh. For the first mapped family, a is tiny and
x huge: da2*x*G and h^2*dC are opposing terms of order h^2*x*G, whereas the
desired tensor is approximately-h^2*y*Gh. The small reference term can vanish
inside the rounded deviations before their sum. This affects the beta pole
through dV and dL, and the Lambda regular term contains the same scalar pattern.
The observation follows directly from the immutable source, without replaying
any saved products numerically.

Hardening only dV would still leave a separately ill-conditioned gradient graph.
The frozen lapse-gradient group is

    -a*x*G*(grad a-grad h)
    -( (a-h)*x*G+h*dC )*grad h
       =-a*x*G*grad a+h*y*Gh*grad h.                       (2)

For tiny a and huge x, the split reference terms can be enormous compared with
the finite right side. At large a and tiny x, the frozen chi-gradient expansion

    .5*[a^2*G*(grad x-grad y)
         +((a^2-h^2)*G+h^2*(G-Gh))*grad y]
       =.5*a^2*G*grad x-.5*h^2*Gh*grad y                  (3)

likewise creates O(a^2*grad y) intermediates that need not occur in the desired
answer. A dV-only or individual log-gradient-only patch does not cancel the
complete artificial gradient terms.

## A separately named future outer arithmetic proposal

Preserve the entire frozen RWM helper and all old failed outputs. A new source
identity could use the old exact-deviation graph near reference, and a directly
grouped far graph, with a fixed reviewed branch criterion and the same real
equations. For the far regular-beta graph evaluate the complete live/reference
terms, using scaled composite products and complete field-dual product rules:

    (a^2*x*Lambda-h^2*y*LambdaHat)
    +(beta.grad beta-betaHat.grad betaHat)
    +.5*(a^2*G*grad x-h^2*Gh*grad y)
    -(a*x*G*grad a-h*y*Gh*grad h).                         (4)

The advection difference may retain its current deviation form. Composite
tensor differences should be formed as Product(a,a,x,G_ij) minus
Product(h,h,y,Gh_ij), rather than from an already rounded/underflowed scalar A0.
Then form dL consistently from that tensor and the existing beta deviations.
All nonflat reference connections and Omega derivatives remain present.
No claim is made that a generic difference of legitimately huge physical terms
can be evaluated accurately by this graph for arbitrary positive states.

If this new outer graph is later rebound to the inner gauge, the full regular
beta gradient group at general W must be handled coherently. With
k=B/(A0+B), A0=a^2*x, its exact form is

    2*a^2*k^2*G*grad x
    +(.5-2*k^2)*A0*G*grad y/y-.5*h^2*Gh*grad y
    -a*x*G*grad a+h*y*Gh*grad h.                           (5)

The Lambda group is B*(Lambda-LambdaHat)+(A0-Ah)*LambdaHat. Formula(5)
becomes (2)+(3) at W1,k=.5. A new reviewed arithmetic graph must not be described
as bitwise retaining the old outer source. Old-vs-new recorded comparisons,
exact-reference zero/branch/principal identities, unchanged fixed oracle cases
and thresholds, complete field-dual gates and a new explicit source identity
would be required before any acceptance. This note authorizes none of those
queries or changes.
