# Restart diagnostic gates

Declare these gates before the first run. The process test reuses the accepted
cycle-one, two-rank checkpoint from `history-lifecycle-v1/history-ranks2`. Start
the ordinary AthenaK executable on eight ranks with native restart, unchanged
physics/FastFlow parameters and `nlim=1`, so no RK step executes. Reset output
counters in the new directory and request ordinary history and restart output.
The original Driver emits final output at the restored cycle/time.

Require cycle, exact time/dt, atlas, policy identity, topology and the complete
31-field/eight-ChiRecord checkpoint payload to remain bitwise equal. Require
exactly one history row and one complete harmonic shape/summary record at cycle
one. Compare all ten normalized history columns to the source run with the
short arithmetic gate `1e-12 + 5e-10*abs(reference)`.

FastFlow's restart stores its previous monopole as the next initial guess; a
same-time restart can follow a different convergent iteration path. Use the
adopted comparison protocol's mass/radius gate `1e-10 + 1e-7*abs(reference)` in
M units for mass, mean radius and minimum radius. Coordinate lengths/mass are
multiplied by three. For area (multiplied by nine), use absolute `1e-10` plus
relative `2e-7`, the first-order propagation of the radius gate through area.
Spin components/magnitude are multiplied by nine and use absolute `1e-10`
plus relative `1e-7`. Report expansion RMS/mean changes and require finite
values; they remain convergence diagnostics, as specified in the adopted
protocol, rather than requiring identical iteration histories. Require the
converged complete shape (the history coordinator already rejects failure).
The normalized constraints and complete-state checks remain the stricter gates.

The separate same-object C++ test initializes through the original Driver,
packs its complete native state, caches constraints, changes one rank-zero
Theta value and verifies that change enters the cache. `restart::Unpack` then
restores the original state in the same module object. Require an incremented
state token and exactly one new collective refresh, followed by cache reuse.
All owned eight-component constraint/geometry records must equal their initial
values bitwise, and packing after refresh must recover every original field
and ChiRecord bitwise. This test complements, and does not replace, the actual
process restart on a different decomposition.

Preserve source, native object, executable and source-checkpoint hashes. Never
modify the previous passing run. These are local CPU MPI correctness tests;
they do not establish GPU performance or long-time diagnostic accuracy.

## Clarification after the first process run

The adopted N12 protocol's horizon radius is the **areal radius**,
`3*sqrt(area/(4*pi))`. Applying the same threshold to compact-coordinate mean
and minimum radii above is an additional stricter gate introduced by this test;
it must not be relabeled as the adopted areal-radius requirement. The first
run preserves its failure of those additional gates. The follow-up controls
report the adopted areal-radius gate and the two additional compact-radius
gates separately, without relaxing either.

The original FastFlow summary field named `hrms` contains the mean square;
the expansion RMS is its square root. The summary `hmean` is an integrated
surface quantity. Follow-up controls report these conventions explicitly.
The diagnostic control's initial byte-equality assertion on complete 2/8-rank
summary text also detected area/spin reduction roundoff; its preserved v1 run
is followed by the already-adopted short arithmetic comparison in v2. This
does not change either radius gate or the required bitwise evolved-state gate.
