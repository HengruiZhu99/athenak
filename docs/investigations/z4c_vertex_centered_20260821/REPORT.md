# Native vertex-centered vacuum Z4c: implementation and bounded qualification

Date: 2026-08-22

## Executive verdict

The branch implements a native vertex-centered (VC) vacuum Z4c path alongside
the unchanged-default cell-centered (CC) path.  Native storage has `N+1` active
points in each non-collapsed direction, the Cartoon axis is an evolved
`rho=0` vertex, shared nodal degrees of freedom are synchronized
deterministically, AMR uses nodal injection/interpolation, and restart/output
carry centering metadata.

The bounded fixed-grid O4 Brill test is qualified: all nontrivial evolved
fields converge at effective order 3.31--4.08 through `t=0.5 M`, and every
shared state and derived-constraint vertex agrees bitwise across MeshBlocks.
The common-tree Brill discriminator is not qualified: the three resolutions
replay the same six topology events at exactly the same physical times, but all
three fail the post-RK metric-SPD gate, and failure occurs earlier with
increasing resolution.  Constraint norms diverge with resolution over the
common admissible interval.  This result is evidence against the repaired
false-axis classification being the sole instability; it does not identify a
unique remaining source defect or establish a coarse-fine-interface cause.

The exact-final-source SYCL retry still faults in the first dynamic VC
refinement on PVC.  Because device portability is part of the production
contract, the overall verdict is therefore:

`VC_Z4C_NOT_QUALIFIED`

No Figure-3, critical-collapse, horizon, long-time, convergence, or production
readiness claim is made.

## Source authority

- Repository: `git@github.com:HengruiZhu99/athenak.git`
- Branch: `codex/z4c-vertex-centered-cartoon-amr-20260821`
- Base commit: `6daa774d7451dbc5f7cac640c6e32a6fd11de7f9`
- Base tree: `cbb702f4da954cf630da261790d5c21ef3142235`
- Production implementation commit after the SYCL portability repair:
  `5d37b5e5c278ac4a1afd52f9553dee6ffed48d0e`
- Production implementation tree:
  `e391e2889647471e2e8c0cf8bfbfeb5fe3c00edf`
- Kokkos commit: `6739bc623081648af9e752b616d9671527922cbf`
- Pinned GR-Athena++ reference commit:
  `8583eb9b13639ef7cef65b93f12e9858884527a9`

`EVIDENCE_MANIFEST.json` binds this production implementation commit.  The
later documentation/evidence commit is reported in the final handoff rather
than embedded self-referentially in the manifest it contains.

## What was implemented

The public selector is `<z4c>/grid_centering = cell | vertex`; `cell` remains
the default.  The VC path includes:

- immutable active/stored/coarse nodal geometry with collapsed directions
  remaining one plane;
- compile-time CC/VC kernel dispatch for Cartesian and Cartoon O2/O4/O6
  derivatives;
- an evolved Cartoon `rho=0` vertex with analytic SO(2) state/RHS regularity;
- canonical integer vertex identities and deterministic same-level and
  coincident coarse-fine synchronization;
- nodal injection restriction and symmetric midpoint prolongation for dynamic
  refinement, derefinement, load balance, and ordinary boundary exchange;
- accepted-state ordering for synchronization, algebraic projection, coarse
  cache reconstruction, ghost reconstruction, ADM conversion, constraints,
  curvature, and timestep selection;
- native VC ADM/constraint/curvature calculations and an explicit VC-to-CC ADM
  adapter for the qualified CC consumers;
- centering-aware restart, history, table, binary, VTK `POINT_DATA`, and PDF
  output contracts;
- centering-independent AMR-history record/replay with explicit, audited
  topology-only CC-to-VC compatibility;
- direct IrisK Brill import at native VC coordinates; and
- allocation-free rejection of nonvacuum physics and unadapted consumers.

The exact changed-file inventory and commit sequence are in
`implementation_inventory.md`.  Unsupported combinations are in
`unsupported_feature_inventory.md`.

## Regression and focused qualification

### Cell-centered regression

The pre-change host/MPI baseline registered 65 tests; all 63 enabled tests
passed, with two CUDA-required tests disabled.  After adding the selector and
native dispatch, the one-cycle CC history and timestep-contract files remained
byte-identical:

- history: `4896c333ceda81d99cf1e4c15a28996d73c999c6222d4b83e770c9f4f4d0f598`
- timestep contract:
  `dad954f5938eea76aca74493ec5bd1ac8c66cdc67ac7ad24225988c19e5e3037`

The numerical payload hashes for the initial/final constraints and Z4c state
also match exactly; see `baseline/README.md`.

### Host and MPI

The selected VC matrix contains 30 tests covering layout, coordinates,
collapsed dimensions, O2/O4/O6 Cartesian and Cartoon derivatives, evolved-axis
regularity, canonical topology, same/coarse/fine boundary indices, task order,
dynamic AMR, static multilevel exchange, restart/rank change, output/history,
AMR record/replay, direct Brill import, and O4 wave convergence.  Release host
and MPI builds pass the focused numerical tests.  In the exact-final local
Serial build, all 28 fast selected tests pass.  The O4 linear-wave driver
exceeds CTest's 180-second harness limit in this Serial configuration, but its
explicit bounded invocation completes and reports orders 3.994--4.001 for the
four excited fields.  The debug/bounds build has the same documented
long-test timeout.

### CUDA, Perlmutter

The Perlmutter A100 selected matrix from the Brill campaign source passes
30/30.  The exact test inventory and log are in
`evidence/perlmutter/qualification/`.

Pre-SYCL-portability-fix executable:
`d1b9fa83a5dc2b2f8e9048465dda7d2b896c140880ea9e6af5eeb526421a7cfa`.

The later source change replaces device-lambda captures of two
`Kokkos::DualView` wrappers with their device views only.  It does not alter
indices, arithmetic, operators, state, or ordering.  Exact-final-source CUDA
status is reported separately in `EVIDENCE_MANIFEST.json`; the bounded Brill
data below are preserved as pre-portability-fix numerical evidence rather than
silently relabeled as final-source output.

The exact-final Perlmutter build at `5d37b5e5...` completed successfully:

- executable SHA-256:
  `6dadf42591f77bd9236a39230a8cf70290a68b61565dd0e9b6b624567691f54a`;
- cache SHA-256:
  `9809192574df929b2b70a550f48e1bea29d50369df62bbd65a5bb5e566fa4ec4`.

Its short one-GPU device matrix was not launched because the NERSC SSH
certificate expired before allocation.  This is recorded as pending, not
inferred from the earlier CUDA executable.

### SYCL, Aurora

The first PVC attempt passed tests 1--16 and then faulted in the first dynamic
VC refinement.  One narrow portability repair changed
`MeshRefinement::RefineVC` to capture device views rather than dereference
captured `DualView` wrappers and added a static regression.  The exact-final
retry at `5d37b5e5...` again passes tests 1--16 and then produces the same GPU
page-fault/segmentation failure immediately after the first accepted dynamic
refinement transaction.  Therefore the first diagnosis was incomplete, SYCL
dynamic AMR is not qualified, and no second blind repair/retry is claimed.
After normalizing paths, timings, GPU virtual addresses, and node names, the
two test-17 failure tails are identical.

Exact-final Aurora evidence:

- job: `8774420.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`;
- executable SHA-256:
  `4189f822560b979693c2336f66e175d183db7254f2e2479e10e91e9d44f4bd85`;
- cache SHA-256:
  `d376655892259902bb67a6a7015cdd2e130921202123f3967657018641b322a9`;
- raw manifest-file SHA-256:
  `7e0a3225f091d11110e775bd51a3b87384785e702f1ba796fbde3f7385f0c0c0`;
- raw detached-file SHA-256:
  `eb55d88b1302ce8d0bfef3ad81d33952759d2d69f6557e315bc692c48491e245`.

## Bounded Brill evidence

All cases use native VC O4, direct IrisK data, a common physical domain and a
single A100.  Gate E changes only uniform resolution.  Gate F replays one
authenticated CC topology authority as geometry, with the same root
MeshBlock lattice and identical physical MeshBlock extents at N128/N256/N512.

### Gate E: uniform fixed grid

All three runs reached `t=0.5 M`.  Whole-domain nontrivial-field orders range
from 3.314 (`Azz`) to 4.081 (`gzz`); representative values are:

| field | effective order |
|---|---:|
| `chi` | 3.779 |
| `gxx` | 3.849 |
| `Khat` | 3.550 |
| `Axx` | 3.513 |
| `Gamma_x` | 3.532 |
| `Theta` | 3.666 |
| `alpha` | 3.769 |

Suppressed odd components remain exactly zero.  Across 821, 1653, and 3317
duplicated physical vertices at N128, N256, and N512 respectively, the maximum
mismatch is zero for every one of the 25 evolved fields and all seven derived
constraint fields.

### Gate F: exact common-tree replay

Every resolution executes six topology events with zero-ULP event-time error
and the same last hierarchy checksum `2db37c60ec3c806c`.  All stop fail-closed:

| case | time | cycle/stage | SPD reason | `(rho,z)` | nearest CF interface |
|---|---:|---:|---|---|---|
| N128 | 3.4832261887 | 702/1 | pivot 2 | (11.0,-1.25) | none |
| N256 | 2.4372449837 | 1092/1 | pivot 0 | (3.875,0.3125) | none |
| N512 | 1.3892645558 | 1356/4 | pivot 1 | (11.78125,-1.78125) | none |

The failure locations are not on the physical Cartoon axis and have no
coarse-fine interface in the retained local failure record.  Axis regularity
corrections remain at roundoff.  Over the common admissible interval through
`t=1.3892645558 M`, median effective orders of the constraint histories are
negative: C -2.907, H -2.885, M -3.040, and Z -2.122.

Observation: the same topology is reproduced exactly, yet the solution does
not converge and failure moves earlier with resolution.

Inference: the formerly misclassified lower vertex of every radial MeshBlock
was a real bug and its correction removes the visible MeshBlock-stripe error,
but it was not the sole common-tree instability.

Open question: current evidence does not decide whether the remaining mode is
bulk semidiscrete/gauge behavior, an ordinary block-interface mode, or another
representation/accepted-state defect.  The retained terminal records do not
support blaming a coarse-fine interface simply because AMR is present.

## Formal verdicts

| gate | verdict | boundary |
|---|---|---|
| `cc_regression` | `BITWISE_OR_QUALIFIED_UNCHANGED` | exact bounded CC payload/history checks |
| `vc_uniform_grid` | `QUALIFIED` | bounded O4 through `t=0.5 M` |
| `vc_cartoon_axis` | `QUALIFIED` | manufactured/device tests and axis telemetry |
| `vc_shared_nodes_mpi` | `QUALIFIED` | deterministic rank/decomposition tests and zero mismatch |
| `vc_amr` | `FAILED` | host/CUDA lifecycle tests pass, but exact-final SYCL dynamic refinement faults; Brill common-tree physics also does not qualify |
| `vc_restart_output` | `QUALIFIED` | same-centering rank-change restart and nodal outputs |
| `vc_vacuum_support` | `COMPLETE` | within the explicitly enumerated vacuum consumers |
| `vc_matter_support` | `EXPLICITLY_UNSUPPORTED` | construction fails before allocation without an adapter |
| `brill_discriminator` | `VC_AND_CC_SHARE_Z_MODE` | both show resolution-worsening Z/constraint behavior; mechanism identity is not claimed |
| `overall` | `VC_Z4C_NOT_QUALIFIED` | exact-final SYCL dynamic AMR fails and Gate F is not numerically qualified |

## Evidence and reproducibility

- Raw Perlmutter archive:
  `/pscratch/sd/h/hzhu/z4c-vc-qual-684402ae-v2-20260821/campaign-v2`
- Local raw subset:
  `/home/hzhu/Desktop/research/gr/collapse/artifacts/z4c_vertex_centered_20260821/perlmutter/campaign-v2-axis-fix`
- Repository evidence subset: `evidence/perlmutter/`
- Comparison tables: `data/`
- Figures: `figures/`
- Deterministic analysis: `analyze_axis_fix_campaign.py`
- Exact-final CUDA launcher:
  `allocate_perlmutter_final_source_qualification.sh` and
  `run_perlmutter_final_source_qualification.sh`

Raw Perlmutter manifest-file SHA-256:
`296a14d6c95a8e756c036e410249dd0bd388e90972330c8df062be059aeeafc1`.

Raw Perlmutter detached-file SHA-256:
`6e48c3b66739935de78bb8c405c3afafe7772364e7fcdbcb8c6d7f6ebc955bf3`.

The repository evidence manifest and its detached checksum are generated only
after the Aurora disposition is final and all files are frozen.

The compact first Aurora failure is mirrored under
`/pscratch/sd/h/hzhu/z4c-vc-cross-backend-evidence-20260822/`.  The exact-final
Aurora failure is retained both on Aurora and in this repository; mirroring it
to Perlmutter remains pending the same credential renewal.

## Limitations and next diagnostic

- Gate F is not convergent and does not reach the requested bounded target at
  all three resolutions.
- The terminal SPD failures are post-RK accepted-state failures, but the first
  writer of the invalid metric representation has not been localized.
- No long collapse, apparent horizon, Figure-3 curve, critical exponent, or
  production performance result is qualified.
- VC has no matter coupling.  Unsupported consumers fail closed.
- The full debug linear-wave convergence test timed out; the exact-final
  Release bounded invocation provides the qualified order result.
- Exact-final SYCL dynamic AMR faults immediately after its first accepted
  refinement transaction.  The retained log does not yet localize the invalid
  device access beyond that production phase.
- The selected tests exercise most requested semantics, but the full Phase-16
  stable-name matrix is not present.  In particular, no production
  `z4c_vc_gauge_wave` test exists; combined tests must not be relabeled as that
  missing evidence.
- Memory-per-MeshBlock and buffer geometry are defined by the native-layout
  and boundary-index contracts, but backend-comparable synchronization, RHS,
  AMR, restart, and output timing costs are not qualified because the required
  SYCL device gate failed.  No performance claim is made from partial timers.

The smallest next scientific diagnostic is a bounded stage-resolved metric-SPD
provenance census on the earliest N512 failure window, comparing immediately
before and after RHS/RK update, shared-node synchronization, restriction,
exchange, physical/axis boundary reconstruction, and coarse-fine prolongation.
It should preserve the exact hierarchy, gauge, CFL, KO, and strict gates.  This
is preferable to another long run or broad parameter sweep.
