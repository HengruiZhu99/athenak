---
status: plan
last_source_verification: 2026-08-15
owner: z4c
sources:
  - src/mesh/mesh_refinement.cpp
  - src/bvals/prolongation.cpp
  - src/outputs/history.cpp
  - src/z4c/cartoon_meridional_sampler.hpp
generated: false
---

# Goal Mode prompt: diagnose the N256 Cartoon Z4c AMR constraint jump

Work autonomously until the repaired N256 Brill-wave refinement-level constraint
jump has a quantitative, phase-resolved diagnosis backed by authenticated source,
runtime, and artifact evidence. This is a **diagnosis-only** goal. Do not change the
production transfer operators, gauge, dissipation, constraint damping, AMR
criterion, floors, timestep policy, or initial data.

Do not mark the goal complete because the event is reproduced, a plot looks
suggestive, a test passes, or one writer is merely plausible. Completion requires
the T0--T5 accounting and changed-patch attribution defined below to close. If the
accounting cannot close, preserve the evidence and report the result as
`inconclusive`; do not infer a repair.

## Repository and immutable starting point

Use the real AthenaK worktree:

    /home/hzhu/Desktop/research/gr/collapse/worktrees/
      axisymmetric_cartoon_remaining/brill_2d_high_order_restriction

At the time this prompt was written:

    branch: codex/cartoon-2d-high-order-restriction-20260815
    branch head: ea81e528ccd9b8d0b142f1d77b3cc6eb775f1ef4
    restriction repair: 345dd31d59cebd9c0c7231be43dcc6a72524bcc7

The branch-head commit documents the run; the production repair is commit
`345dd31d`. Begin by rechecking the branch, commit graph, remotes, applicable
repository instructions, and worktree status. Existing tracked or untracked
changes belong to the user. Do not reset, clean, overwrite, or incorporate
unrelated work.

The frozen paired-run artifact is:

    /home/hzhu/Desktop/research/gr/collapse/artifacts/
      axisymmetric_cartoon_z4c_2026-08-10/
      r4_brill_figure3_2d_restriction_fix_n128_n256_mpi4_
      345dd31d_v1_20260815

Its terminal evidence records Perlmutter job `57034787`. Authenticate the local
bundle, terminal manifest, detached manifest, input, executable identity, source
status, histories, result JSON files, and accounting before using any value from
it. Do not mutate the frozen artifact or its remote campaign root.

## Established facts that must not be reopened without contradictory evidence

The narrow collapsed-x3 restriction repair is source-correct. It replaces the
inconsistent generic four-cell average with configured point-value restriction
for two-dimensional Z4c data in both:

1. dynamic refinement/derefinement; and
2. ordinary same-level coarse-boundary refresh.

Generic non-Z4c data and the established three-dimensional behavior remain
unchanged. The repaired transfer is non-monotone and is not positivity preserving
for `chi`, but restriction overshoot has not yet been demonstrated in the failing
run.

The global Cartoon constraint history normalization is already correct. In
Cartoon mode, `Z4cDiagnosticCellMeasure` uses

    2*pi*rho*dx1*dx2*sqrt(abs(det(gamma)))

and does not use the collapsed `dx3`. Do not add an extra `1/dx3` normalization.
The global `C-norm2`, `H-norm2`, `M-norm2`, and `Z-norm2` columns are squared
proper-volume integrals. Some `ax-*` and per-layer columns are intentionally
unweighted cell sums and must not be described as resolution-independent physical
norms.

The primary reproducible event is the repaired N256 transition:

    cycle: 1724
    coordinate time: 9.5109375 M
    maximum level: 2 -> 3
    MeshBlocks: 74 -> 86
    sqrt(C) factor: approximately 2.628
    sqrt(H) factor: approximately 7.652
    sqrt(M) factor: approximately 8.232

Across the adjacent history rows, the recorded proper volume changes only from
approximately `47723.4405744` to `47724.7284709`, a relative change of about
`2.7e-5`. The constraint jump is therefore not explained by a missing azimuthal
measure or fictitious y width.

## Non-negotiable scientific and numerical boundaries

- Reproduce the exact N256 case freshly from `t=0`; do not begin from a restart.
- Preserve the archived N256 initial-data and runtime input bytes except for a new
  diagnostic output basename or explicitly required diagnostic keys.
- Preserve O6 bulk derivatives, RK4, CFL `0.15`, KO dissipation `0.02`, the
  max-domain-|K|-scaled telegrapher lapse with `tau=kappa=1`, fixed-eta advective
  Gamma-driver shift, `kappa1=kappa2=0`, `floor_chi=false`, `dchi_max=0.01`, root
  grid `128 x 256 x 1`, MeshBlocks `32 x 32 x 1`, domain `rho=[0,16]` and
  `z=[-16,16]`, and maximum refinement level 20.
- Use the same four-rank, one-node CUDA/MPI execution shape and authenticated GPU
  binding contract as the frozen N256 run. Do not substitute a CPU run or a
  different rank decomposition for the qualifying replay.
- Do not enable a chi floor, weaken positivity checks, relax thresholds, change
  block size, freeze the hierarchy, lower transfer order, switch centering, or
  perform gauge/CFL/AMR parameter sweeps.
- Do not implement a production repair in this goal. The final report may rank
  repairs, but a repair and its validation require a separate authorization.

## Required default-off diagnostic interface

Add a runtime-gated diagnostic interface under the Z4c input block. Follow local
input naming conventions, with behavior equivalent to:

    amr_jump_diagnostic = false
    amr_jump_target_level_before = 2
    amr_jump_target_level_after = 3
    amr_jump_post_cycles = 8
    amr_jump_output_basename = z4c_amr_jump

The default must be `false`. With the option absent or false, no diagnostic arrays,
files, reductions, fences, callbacks, or task dependencies may be added to the
production path. Unknown, malformed, non-Cartoon, non-AMR, or inconsistent target
settings must fail before evolution with a clear configuration error.

When enabled, record compact aggregates for every actual AMR transaction. Emit
detailed changed-patch data only when the maximum refinement level changes. The
target detailed event is the first accepted `2 -> 3` transition. After that event,
continue for exactly eight accepted evolution cycles and stop cleanly with final
diagnostic output, restart evidence, accounting, and checksums.

## T0--T5 transaction ledger

Capture the following ordered states for every detailed event:

### T0: accepted old hierarchy

Capture the completed pre-AMR evolution state, current ADM fields, current
constraints, proper and coordinate-ring integrals, extrema and locations, current
MeshBlock tree, and refinement flags before moving any field data.

### T1: balanced topology proposal

Capture the requested flags, balance-induced refinements, old and proposed logical
locations, old-to-new and new-to-old mappings, rank ownership, and the exact set of
refined and derefined parents. No evolved-field value should change between T0 and
T1.

### T2: redistributed/refined active state

Capture the new active arrays after local copies, MPI load-balance transfer,
restriction for derefinement, coarse-source copies, and `RefineCC`, but before
ordinary ghost reconstruction. T2 is valid for evolved-field comparison only;
ordinary derivative constraints must not consume its incomplete ghosts.

### T3: completed boundary reconstruction

Capture the state after the AMR-created hierarchy has passed through restriction,
MPI exchange, physical and axis boundary handling, same-level coarse refresh, and
coarse-to-fine prolongation. Maintain an internal T3 sub-ledger after each of these
operations:

    RESTRICT
    MPI_RECEIVE
    PHYSICAL_OR_AXIS_BC
    SAME_LEVEL_COARSE_REFRESH
    PROLONGATION

Every changed value in detailed output must carry its most recent writer label.

### T4: projected Z4c state

Capture the state after Z4c algebraic projection and the subsequent parity-ghost
refresh. Do not combine this record with ADM conversion.

### T5: accepted new hierarchy and constraints

Capture the final Z4c state after Z4c-to-ADM conversion and ADM/Z4c constraint
recomputation, together with the accepted new hierarchy, timestep, proper and
coordinate-ring integrals, extrema, and locations.

## Common physical comparison lattice

For every newly refined parent, construct a canonical fine physical lattice over
its four two-dimensional children plus the full O6 derivative halo. The lattice
coordinates and ownership must be independent of rank and stable across T0--T5.

Use AthenaK's own geometry, interpolation, derivative, ADM, and constraint code to
evaluate data on this lattice. External scripts may validate, join, summarize, and
plot recorded values, but may not substitute rederived physics formulas for the
authoritative AthenaK evaluation.

Record at least:

- every evolved Z4c component;
- `chi`, lapse, shift, `K`, `Theta`, and `Atilde_ij Atilde^ij`;
- `C`, `H`, `M`, `Z`, and their squared local contributions;
- `det(gamma)` before applying any absolute value;
- rank, GID, logical location, level, indices, `(rho,z)`, axis status, physical
  boundary status, block-boundary distance, and coarse-fine-interface distance;
- source stencil, target, and center-center, edge-center, or edge-edge rule class
  for restriction and coarse refresh.

T2 has incomplete production ghosts. For T2, record evolved fields and provenance
but mark derivative constraints unavailable. For T0, T3, T4, and T5, use
diagnostic-only scratch storage to evaluate ADM fields and constraints without
modifying the production state or its task graph.

At each restriction and same-level coarse-refresh target, compute two values:

1. the production high-order result; and
2. a diagnostic-only convex `2 x 2` sibling average.

Record both even when they remain finite and positive. Never feed the shadow value
back into evolution. Aggregate minimum production and shadow chi, maximum absolute
and relative discrepancy, positive-source/nonpositive-target counts, and counts by
rule class and writer.

## Quantitative closure and classification

Produce a telescoping ledger for each evolved field and each valid constraint
diagnostic:

    T0 -> T1 -> T2 -> T3 subphases -> T4 -> T5

The sum of phase increments must reproduce the recorded T0-to-T5 difference within
declared floating-point reduction tolerance. Reconstruct native global history
integrals from per-block partial sums and verify agreement with the authoritative
history row. Separately report the fixed-lattice change so a changed AMR quadrature
cannot masquerade as a field discontinuity.

Classify the result using the earliest phase with a verified change and the full
phase contribution ledger. Allowed dispositions include:

- `topology_or_measure_only`;
- `refine_or_derefine_transfer`;
- `mpi_redistribution`;
- `physical_or_axis_boundary`;
- `same_level_coarse_refresh`;
- `coarse_to_fine_prolongation`;
- `algebraic_projection`;
- `adm_or_constraint_recomputation`;
- `quantified_multi_stage`;
- `inconclusive_accounting_not_closed`.

Do not force a unique culprit if multiple stages contribute. A multi-stage result
is acceptable only when each contribution and its spatial support are quantified.
Do not call the result diagnosed if bytes lack a writer, changed patches are
missing, phase increments do not telescope, or rank aggregation is incomplete.

## Required tests before the N256 replay

1. Preserve and rerun the existing focused Cartoon restriction, chi prolongation,
   coarse-refresh, shared-geometry, restart, Serial, and MPI tests.
2. Verify that Cartoon diagnostic measure is independent of `dx3` and that flat
   coordinate ring volume of one parent equals the sum of its four children exactly.
3. Verify constant and smooth polynomial fields across a synthetic `2 -> 3` regrid.
   No unexplained fixed-lattice discontinuity is allowed.
4. Inject one test-only perturbation at each T phase and T3 writer subphase. The
   ledger must attribute every mutation to the correct phase and location.
5. Exercise center-center, edge-center, and edge-edge changed patches with the O6
   halo present.
6. Verify one-, two-, and four-rank aggregation produces identical global ledgers,
   unique ownership, and complete changed-patch inventories.
7. Mutation-test malformed target levels, missing target events, duplicate patch
   ownership, nonfinite data, nonpositive metric determinant, incomplete writer
   provenance, accounting mismatch, and premature stopping.
8. Verify that disabling the option produces no diagnostic files and preserves the
   established default path.

Run focused host tests first, then MPI tests, then the exact CUDA-focused tests in a
fresh Perlmutter build. A configure/build pass is not CUDA qualification; record
the executable hash, cache, compiler/toolchain, Kokkos commit, device identity,
rank binding, and source status.

## Fresh N256 diagnostic replay

Stage a fresh immutable campaign root. Do not reuse the prior run directory or a
failed staging identity. The login-node preflight must bind:

- diagnostic source commit and tree;
- exact parent source and the diagnostics-only diff;
- Kokkos identity;
- input and IrisK payload hashes;
- build configuration and resource contract;
- expected target event and stop condition;
- exact output inventory and accounting policy.

Run the exact N256 case from `t=0`. Compact transaction summaries must be written
for all AMR changes. Detailed capture must occur at every maximum-level change, with
the `2 -> 3` event designated as the target. Require reproduction of cycle 1724,
time `9.5109375 M`, MeshBlocks `74 -> 86`, and the archived native jump within a
declared tight replay tolerance. If the topology or event is not reproduced, stop
as `reproduction_mismatch`; do not analyze a different transition as equivalent.

After T5 for the target event, run exactly eight more accepted cycles. Then stop
cleanly without waiting for the later chi failure. Preserve the final restart,
history, transaction ledger, detailed changed-patch records, logs, accounting,
source status, root checksum manifest, and detached manifest hash.

## Analysis products

Produce deterministic analysis scripts and the following plots/tables:

- native proper-volume and coordinate-ring `C/H/M/Z` histories around the event;
- fixed-lattice T0--T5 constraint norms;
- telescoping phase contributions for each constraint family;
- changed-patch maps of constraint and evolved-variable deltas;
- worst-point trajectories through all phases;
- production high-order versus shadow-average chi by rule class;
- distances of worst changes to the axis, block edges, and coarse-fine interfaces;
- proper volume and minimum determinant by phase;
- eight-cycle post-event persistence or relaxation.

Create a concise Markdown report and a machine-readable strict JSON verdict. The
report must distinguish observations, quantitative attribution, hypotheses, and
unresolved questions. It may recommend a later limited-O2, positivity-aware,
edge-only, frozen-hierarchy, block-size, or CFL experiment, but it must not perform
one.

## Completion gates

The goal is complete only when all of the following hold:

- the reviewed source and frozen predecessor evidence authenticate;
- diagnostic functionality is default-off and its focused negative matrix passes;
- a fresh four-rank CUDA N256 replay reproduces the target event;
- T0--T5 and T3 subphase data exist for every changed patch;
- native history sums and per-block accounting agree within declared tolerance;
- the common-lattice ledger telescopes within declared tolerance;
- every detailed value has unique ownership and writer provenance;
- the target event is classified quantitatively, including an explicit multi-stage
  classification when appropriate;
- eight accepted post-event cycles establish whether the jump relaxes or persists;
- all raw evidence, scripts, reports, schemas, manifests, and detached hashes verify;
- the final verdict retains `qualification_claim=false` and explicitly states that
  no production repair or Figure 3 qualification was performed.

If a blocking external condition recurs for the number of turns required by Goal
Mode, preserve the current immutable evidence and mark the goal blocked with the
exact missing authority or resource. Do not weaken a completion gate to finish.

## Commit and publication boundary

Keep diagnostic work isolated from unrelated user work. A clean local diagnostic
commit may be created for exact build provenance. Do not push, merge, open a pull
request, or publish artifacts unless the invoking user separately authorizes that
external action. This Markdown file itself is a planning artifact, not evidence of
implementation or qualification.
