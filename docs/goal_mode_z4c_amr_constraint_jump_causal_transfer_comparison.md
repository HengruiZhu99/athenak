---
status: plan
last_source_verification: 2026-08-16
owner: z4c
sources:
  - src/z4c/amr_jump_diagnostic.cpp
  - src/mesh/mesh_refinement.cpp
  - src/bvals/prolongation.cpp
  - src/outputs/history.cpp
  - src/z4c/cartoon_meridional_sampler.hpp
generated: false
---

# Goal Mode prompt: isolate and correct the Cartoon Z4c AMR constraint jump

Work autonomously until the N256 Brill-wave level-2-to-3 refinement event has a
prospectively classified transfer diagnosis, any source correction is justified
by matched causal evidence, and the accepted candidate has been compared against
the two existing transfer modes over the requested evolution window.

This goal is about the **constraint jump at refinement**, not the later strict
positive-`chi` termination. Do not broaden it into terminal-`chi` provenance,
gauge tuning, dissipation tuning, AMR-threshold sweeps, vertex centering, or a
new Figure-3 reproduction campaign.

Do not mark the goal complete because a case runs longer, a plot looks smoother,
or a low-order method is more dissipative. Completion requires the zero-PDE
causal gates, operator tests, three-method comparison, and authenticated report
defined below. If a gate is not met, stop at that gate and report the result as
`inconclusive` rather than inventing or promoting a correction.

## Repository and authenticated starting point

Use the real AthenaK worktree:

    /home/hzhu/Desktop/research/gr/collapse/worktrees/
      axisymmetric_cartoon_remaining/brill_2d_high_order_restriction

At the time this prompt was written:

    branch: codex/brill-amr-frozen-hierarchy-20260816
    branch head: 0a59ca23b0fdd0d65f0b10d5cad6b1a3540876f9
    source tree: 9fc2b6296aa6cc5c9bd7cf538dd31c07b533e751
    numerical source commit: 21a268e4735308a39ac4f040d3621ea114b4ef1d

Recheck the branch, commit graph, remotes, applicable repository instructions,
and worktree status before acting. Existing tracked or untracked changes belong
to the user. Do not reset, clean, overwrite, or incorporate unrelated work.

The governing completed handoff is:

    docs/brill_amr_hierarchy_causality_20260816/

Authenticate at least:

    REPORT.md SHA-256:
      83921907e813120d245d8df3086831112fa081d903ad0fdbd823b42afc27a700
    EVIDENCE_MANIFEST.json SHA-256:
      ec0270470a4ea97365b70fc189c1f4a12808462daaa7e4711bc713c33365fb23

The raw job-57098562 terminal evidence is bound by:

    root manifest:
      c5819a40dec0ae278b63e3867d1ca1bb9661fee94f781dc1837ff49a7d8c28c6
    detached manifest:
      4c82deaf1d467dd2b4462698615378e2bda90f3d055dc3cf08b7522423a3210c
    settled sacct:
      b4e8c7838ecb6e4a13763389790484d2a938bb7e413a3090b576ba3f064a02fa

Use the authenticated cycle-1721 restart immediately before the target event:

    restart SHA-256:
      83e996d2d5069307888a69fff47a7524c2f63f11869fb628630bca54dd5943ea
    restart time: 9.5015625 M at cycle 1721
    target transaction: level 2 -> 3 at cycle 1722, t=9.50625 M

Do not mutate the completed handoff, its raw production evidence, or any prior
remote campaign root.

## Established facts and interpretation boundary

Treat these as established unless new contradictory evidence is authenticated:

1. The Cartoon history constraint measure is already the proper ring measure,

       2*pi*rho*dx1*dx2*sqrt(abs(det(gamma))),

   with no fictitious collapsed `dx3`. The jump is not a collapsed-y
   normalization artifact.
2. At the target event, coordinate ring volume is conserved to roundoff, while
   the proper-volume C/H/M integrals jump substantially.
3. The existing T0--T5 ledger closes at roundoff. Active evolved fields are not
   silently rewritten by MPI receive or boundary prolongation after the
   canonical refinement transfer; stored ghosts do change as expected.
4. The largest fixed-lattice discrepancy lies near an internal MeshBlock edge
   around `(rho,z)=(5.13,-0.008)`, far from the symmetry axis.
5. Parent self-shadow and O6--O4 disagreement are much larger in the four-cell
   MeshBlock-edge band than in parent interiors.
6. Matched A/B/C evidence strongly implicates repeated dynamic regridding in A's
   runaway, but buffered frozen case C proves that a larger frozen fine
   representation is not automatically stable.

Keep observations, inferences, and hypotheses separate. Do not claim that a
missing axis ghost fill, a particular one-sided coefficient, parent
under-resolution, or restriction overshoot is proven before the new prospective
tests classify it.

## Non-negotiable numerical boundaries

For every production-state probe and evolved comparison, preserve the
authenticated N256 setup except for the explicitly selected AMR-transfer mode
and default-off diagnostic controls:

- O6 bulk finite differences, RK4, CFL `0.15`, and KO dissipation `0.02`;
- `dchi_max=0.01` and derefinement threshold `0.5*dchi_max`;
- the authenticated refinement interval and maximum refinement level;
- the same gauge, damping, outer boundary, initial-data, MeshBlock, and domain
  bytes;
- `floor_chi=false` and unchanged strict finite/positive-`chi` gates;
- the same rank/GPU decomposition and hardware-binding contract used by the
  authenticated campaign;
- one case at a time on one A100 when the matched campaign contract permits it.

Do not change gauge, constraint damping, dissipation, CFL, block size, AMR
criterion, hysteresis, initial resolution, physical domain, centering, or
positivity thresholds. Do not add a floor, clipping, or run-selection retry.

## Required diagnostic interface

Preserve the existing runtime modes:

    <z4c>/amr_transfer = high_order
    <z4c>/amr_transfer = limited_o2

Their meanings are fixed:

- `high_order`: current configured point-value restriction/prolongation,
  including the one-sided MeshBlock-edge closure and positivity-aware `chi`
  prolongation;
- `limited_o2`: sibling-average restriction plus existing limited second-order
  prolongation everywhere for Z4c, with strict positive-sibling handling for
  `chi`; the bulk PDE remains O6/RK4.

Extend the default-off AMR-jump diagnostic so that:

    <z4c>/amr_jump_post_cycles = 0

and, for this matched comparison:

    <z4c>/amr_transfer = high_order
    <z4c>/amr_jump_target_transfer = high_order | limited_o2

means: capture the complete target T0--T5 transaction, finalize evidence, and
stop cleanly before any RHS or RK update. Negative values remain invalid. The
default diagnostic-off path must remain allocation-, reduction-, fence-, and
output-free.

`amr_jump_target_transfer` is a diagnostic-only override for the exact matched
T1--T5 transaction. It must not affect the RK cycle that advances the common
cycle-1721 restart to cycle 1722, and the production transfer policy must be
restored after T5. This makes the two arms byte-identical at T0 and isolates the
transfer operation causally. Missing target identity, an unknown mode, or use
without the diagnostic must fail closed.

The zero-PDE output must retain:

- the exact topology proposal and old/new maps;
- T0--T5 evolved, ADM, constraint, and coarse arrays;
- ordered writer labels for restriction, receive, physical/axis BC, same-level
  coarse refresh, prolongation, algebraic projection, and recomputation;
- exact ledger closure and finite/nonfinite counts;
- current transfer mode, source, executable, input, restart, rank, and hardware
  provenance.

Unknown modes, malformed parameters, non-Cartoon use, non-AMR use, or incomplete
target settings must fail before evolution.

## Phase 1: matched zero-PDE event probes

Build one fresh executable from one clean source state. Run exactly two initial
probes from identical cycle-1721 restart bytes:

1. `high_order`;
2. `limited_o2`.

Each probe must accept the identical level-2-to-3 topology transaction and stop
after T5 without advancing cycle or coordinate time.

Require the reconstructed T0 evolved, ADM, and constraint bytes to be identical
between arms. Reject the comparison if the target-only override was active
before T1 or was not restored after T5.

Reject the comparison if topology, input, restart, rank layout, source,
executable, hardware, or target-event identity differs between arms.

Require for each arm:

- all files finite and schema-valid;
- the T0--T5 field/writer ledger closes within its recorded tolerance;
- old-to-new mappings and unique ownership close;
- coordinate ring volume is conserved to roundoff;
- proper-volume and coordinate-volume sums reconstruct from leaf cells;
- no RHS, RK update, later cycle, or unplanned topology event occurs.

## Phase 2: common-lattice regional constraint audit

Use the same child lattice for both transfer arms. The authoritative before
state is the T0 parent constraint representation mapped onto that child lattice;
the after state is the T5 constraint field recomputed from the accepted child
evolved variables. Also report native-hierarchy T0 and T5 values as secondary
context.

For C, H, and M, report:

- squared integrals;
- RMS values;
- maxima and physical coordinates;
- finite/nonfinite counts;
- post/pre common-lattice jump ratios;
- phase-resolved maps and transfer differences.

Use the proper axisymmetric ring measure as authoritative:

    2*pi*rho*sqrt(gamma)*d(rho)*dz

Report the coordinate-ring measure

    2*pi*rho*d(rho)*dz

as a cross-check. Require the two measures to support the same spatial
localization before making a densitization-independent claim.

Partition every active child cell exactly once, in this precedence order:

1. `AXIS_OR_PHYSICAL_BOUNDARY`: within four fine cells of the symmetry axis or
   a physical domain boundary;
2. `COARSE_FINE_INTERFACE`: within four fine cells of a coarse-fine face after
   excluding category 1;
3. `MESHBLOCK_EDGE_OR_CORNER`: within four fine cells of a same-level MeshBlock
   edge or corner after excluding categories 1--2;
4. `INTERIOR`: all remaining active cells.

Write strict regional tables for every phase and constraint family. The disjoint
regional sums must reproduce each global integral to roundoff. Produce 2D maps
with MeshBlock and coarse-fine boundaries overlaid.

## Prospective causal classification

Classify high-order transfer as materially causal to the event jump only if
`limited_o2` satisfies all of the following:

1. reduces the post/pre common-lattice jump ratio by at least a factor of two in
   at least two of C/H/M;
2. worsens none of C/H/M by more than 25 percent;
3. reduces the contribution in the same disjoint spatial region that dominates
   the high-order excess.

Apply the branches exactly:

### Gate passes with MeshBlock-edge/corner dominance

Authorize one new candidate mode, provisionally named:

    <z4c>/amr_transfer = edge_limited_o2

Its semantics must be:

- retain the existing high-order operator for center-center transfer groups;
- use sibling-average restriction for a target when either active direction
  requires a MeshBlock-edge closure;
- use the existing limited second-order prolongation for the corresponding
  edge/edge-edge sibling group;
- route `chi` through the corresponding strict limited-positive path;
- apply the same classification in dynamic restriction/refinement and ordinary
  same-level coarse refresh/boundary prolongation;
- support only 2D Cartoon Z4c initially and fail closed if requested elsewhere;
- leave the default, generic variables, non-Z4c behavior, and established 3D
  implementation unchanged.

### Gate passes but improvement is spatially distributed

Do not invent an edge-only repair. Treat global `limited_o2` as the diagnostic
control and report that a local source correction is not isolated. Stop before
the three-method evolution because no distinct third candidate is authorized.

### Gate fails

On the same frozen child evolved state, recompute diagnostic constraints with
O2, O4, and O6 derivative operators without changing the evolution state.
Compare regional norms, spectra, extrema, and pairwise disagreement. This is a
diagnostic calculation only; do not alter production derivatives.

If the derivative-order audit does not isolate a correctable operator seam, stop
with an `inconclusive_parent_resolution_or_derivative_sensitivity` disposition.
Do not launch a resolution campaign or propose a transfer fix from correlation
alone.

### Mixed result

Authorize the edge candidate only if the edge/corner regional contribution
independently satisfies the same factor-of-two and no-more-than-25-percent-worse
rules. Otherwise follow the Gate-fails branch.

## Phase 3: operator validation for the edge candidate

Before using `edge_limited_o2` on the production restart, require focused tests
that prove:

- constants and linear fields are reproduced exactly by the limited edge path;
- nominal smooth-polynomial behavior is unchanged in high-order interiors;
- sibling-average restriction and limited prolongation preserve the parent
  group average exactly;
- low/high orientation and face/edge/corner classification are correct;
- minimum supported MeshBlock sizes remain in bounds;
- collapsed-direction and axis parity behavior are correct;
- finite positive `chi` remains finite and positive;
- invalid parent or invalid limited siblings fail closed;
- no floor, clipping, threshold, or tolerance is introduced;
- both regridding and ordinary coarse-boundary refresh use identical policy;
- default `high_order`, existing `limited_o2`, generic, non-Z4c, and 3D behavior
  remain unchanged.

Run serial debug/bounds, MPI, and relevant CUDA/Kokkos focused tests. A compile or
unit-test pass is necessary but not sufficient.

Repeat the exact zero-PDE cycle-1722 event with `edge_limited_o2`. The candidate
must:

- pass all ledger and provenance gates;
- satisfy the same prospective factor-of-two/no-worse causal gate;
- reproduce most of the global `limited_o2` improvement specifically through
  the edge/corner budget;
- leave high-order interior results consistent with the `high_order` arm.

If it fails, do not tune thresholds or silently broaden its low-order region.
Stop and report the failure.

## Phase 4: three-method evolved comparison

Only after the edge candidate passes Phase 3, launch three fresh matched dynamic
hierarchy cases from the authenticated cycle-1721 restart:

| Case | Transfer mode | Meaning |
|---|---|---|
| H | `high_order` | current high-order/off-centered closure |
| L | `limited_o2` | global sibling-average/minmod transfer |
| E | `edge_limited_o2` | validated high-order-interior/limited-edge candidate |

Use one executable, one source tree, one input template, and one allocation
contract. Run one case at a time. Preserve all numerical settings listed above.

For every case:

- target `t=12.5 M`;
- apply the same prospective 50-minute per-case step cap;
- retain dynamic refinement and derefinement;
- keep the detailed target-event diagnostic plus compact records for all later
  topology transactions;
- retain history, topology, coarse-fine exposure, restart, terminal log,
  accounting, source-status, executable, and hardware evidence.

A timeout, nonfinite field, strict-`chi` failure, scheduler failure, or resource
failure is a terminal disposition. Do not rerun-select, weaken a gate, or extend
only one case. A case that fails before `t=12.5 M` remains part of the comparison.

This evolved comparison characterizes stability; it does not by itself qualify
convergence or production use.

## Phase 5: analysis, plots, and report

Produce matched overlays and tables for all three methods:

- proper-ring C/H/M integrals and RMS histories;
- constraint maxima and locations;
- event-aligned constraint-jump distributions;
- maximum Kretschmann scalar and maximum absolute K;
- timestep, maximum refinement level, and MeshBlock count;
- MeshBlock creation/deletion and cumulative topology changes;
- coarse-fine exposure `X_CF`;
- minimum `chi`, lapse diagnostics, and terminal failure evidence;
- target-event regional proper/coordinate budgets;
- target-event 2D constraint maps with mesh and coarse-fine overlays.

The final Markdown report must contain:

1. a concise verdict;
2. exact source, executable, restart, input, hardware, job, and artifact identity;
3. the prospective zero-PDE gate results;
4. operator-test evidence for the candidate;
5. the three-method terminal table and comparison plots;
6. a strict distinction between observation, inference, hypothesis, and open
   question;
7. explicit limitations and any skipped/failed phase;
8. a strict JSON evidence manifest, root checksum manifest, and detached hash.

Do not claim convergence, Figure-3 reproduction, continuum correctness, or a
production-default transfer from this three-method N256 comparison. A preferred
candidate may be named only as the best-supported next qualification target.

## Preservation, resumption, and stop rules

- Use a fresh successor artifact root and job identity for every frozen campaign
  attempt. Never patch an immutable staged or executed bundle.
- Preserve every substantive failure and its manifests.
- Keep source changes, diagnostic evidence, and scientific conclusions
  separately identifiable.
- Before any allocation, require a clean pushed source identity, exact bundle
  checksums, login-node preflight, absent build/run/allocation state, and an empty
  exact-name queue.
- On interruption, record the active phase, last completed gate, source and
  artifact hashes, scheduler state, and exact next unrun command.
- Do not continue to Phase 3 or 4 when a prior causal gate is inconclusive.
- Do not commit or push unrelated user work.

## Completion criteria

This goal is complete only when one of these terminal conditions is reached:

1. **Validated comparison:** the existing two-mode zero-PDE comparison passes,
   the edge candidate is causally authorized and validated, all three evolved
   cases receive honest terminal dispositions, and the report/manifests verify;
   or
2. **Fail-closed diagnosis:** a prospective gate fails or remains mixed, no
   source correction is justified, no unauthorized downstream run occurs, and a
   complete authenticated report explains exactly why the goal stopped.

Budget exhaustion, a visually improved plot, a longer-lived case, or successful
compilation is not completion.
