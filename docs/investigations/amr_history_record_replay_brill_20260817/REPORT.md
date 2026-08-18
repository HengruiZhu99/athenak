# Brill AMR history record/replay qualification

Date: 2026-08-18

Repository: `HengruiZhu99/athenak`

Branch: `codex/amr-history-record-replay-brill-20260817`

Final source: `ac75c8d348da91b38cbc6855b5fba51cd3089663`
Final tree: `6284882bd06e8db379495675aba7a4f153fb4afa`

## Verdict

The same-resolution replay bug is fixed. A fresh N128 dynamic-AMR recording and
an N128 replay from that recording now reproduce all 210 accepted hierarchy
events and every common binary payload bitwise for `z4c`, `con`, `adm`, `weyl`,
`z4c_diag`, and `telegraph_mu`.

The requested N256 replay was then executed once with twice the cells per
MeshBlock, the same 2x4 root MeshBlock layout, and the N128 hierarchy events
scheduled by recorded coordinate time rather than cycle. It passed the replay
scheduler gates and applied events 1 through 11 as an exact authority prefix.
It subsequently failed the unchanged strict-positive chi parent-stencil gate
at `t=10.53633 M`, cycle 5547, before the next recorded hierarchy event.

This is a successful qualification of deterministic N128 replay and of the
cross-resolution physical-time scheduling path. It is not a successful N256
evolution, a convergence result, or a Figure-3 reproduction.

## Bugs isolated and fixed

### 1. Same-resolution one-ULP timestep perturbation

The original N128 replay recomputed `next_event_time - time` even when the
unmodified production timestep already rounded to the recorded event time.
The subtraction could be one ULP smaller, so replay perturbed the state from
event 2 onward and eventually failed before the authority endpoint.

Commit `0396d281fe4568f9033e16376510e4ca1c35eaea` preserves the production
timestep whenever `time + dt` already equals the recorded event time. This
removed every N128 replay clip and restored the authority cycle sequence.

### 2. Cross-resolution roundoff microstep

The first N256 attempt reached event 6 only `3.885781e-15 M` below its recorded
time. The old tolerance did not classify those values as the same coordinate
time, so replay attempted an RK step of that microscopic remainder. The
central-proper-time diagnostic correctly rejected it as not one forward
accepted step.

Commit `ac75c8d348da91b38cbc6855b5fba51cd3089663` raises the event-time
comparison tolerance from 8 to 32 machine epsilons times the time scale. This
is still many orders of magnitude smaller than the smallest campaign CFL
timestep. An exact hexadecimal regression fixture covers the observed N256
case. Replay now applies an event that is equal within bounded accumulated
roundoff without taking a near-zero PDE step.

## Qualification tests

Local tests at `ac75c8d3`:

- Serial `athena.amr_history_format`: pass.
- MPI `athena.amr_history_format`: pass.
- Serial production-path AMR history integration: pass.
- MPI production-path AMR history integration: pass.

Perlmutter fresh CUDA/MPI build:

- job `57202042`, QOS `gpu_shared_interactive`, one A100 on `nid001001`;
- fresh executable SHA-256
  `8fe79ff1a428756369defeeb7f73313a8323b1331c9250cd8f99b158023bfd1a`;
- CMake cache SHA-256
  `78a04f1db436e9cc4a2dab0742a92fba8fefadeabf5d642a19ea17515c31dff2`;
- five focused CTests: 5/5 pass;
- direct bounded AMR history integration: pass.

## Fresh N128 authority

The v15 N128 record reproduced the established authority exactly:

- accepted events: 210 after event zero;
- final event cycle: 14715;
- final event time: `0x1.b3ec5e9999acfp+3`
  (`13.622603702545716 M`);
- leaves: 977;
- maximum logical level: 22;
- final tree checksum: `9a6db3afa653129c`;
- history SHA-256:
  `d0e1289757bd8f5b6510ca8a7e8b8c5c42bec54f5f08480f607abc866af57555`.

The authority then reproduced its known strict-chi termination at cycle 14722
with 107 invalid parent stencils. That later failure is intentionally outside
the last accepted hierarchy event and is not hidden or reclassified.

## N128 replay identity

The replay stopped normally at the final recorded event time and established:

- 210/210 event records equal in event number, hexadecimal time, leaf count,
  maximum level, and tree checksum;
- endpoint cycle equal to the authority: 14715;
- zero `AMR_HISTORY_TIMESTEP_CLIP` records;
- every common payload bitwise equal in all six output families.

The replay has one additional terminal snapshot per output family because a
normal `tlim` termination writes the endpoint output, while the authority ran
seven more cycles and then aborted at the strict-chi gate before producing a
corresponding terminal output. The identity analyzer compares every authority
snapshot to the replay snapshot with the same output index, cycle, and time,
and records the replay-only endpoint separately.

Identity JSON SHA-256:
`d69619684bd9a723362a5db758d8dc154c961ce76d6d360951b4823e6d392dbb`.

## N256 physical-time replay

Configuration relation:

| quantity | N128 authority | N256 replay |
|---|---:|---:|
| root cells | 64 x 128 | 128 x 256 |
| cells per MeshBlock | 32 x 32 | 64 x 64 |
| root MeshBlock layout | 2 x 4 | 2 x 4 |
| physical MeshBlock extents | authority | identical |
| hierarchy authority | dynamic dchi | N128 recorded tree |
| event scheduler | production cycle | recorded physical time |

The N256 run applied 11 events. Its replay ledger is an exact prefix of the
N128 authority ledger through event 11:

- event-11 time: `0x1.3a79999999976p+3`;
- leaves: 98;
- maximum logical level: 7;
- tree checksum: `cf359244e1483352`;
- genuine event-alignment clips: 5;
- no near-zero replay step after the fix.

At cycle 5547 and `t=10.53633 M`, ordinary boundary prolongation rejected
3928 chi parent stencils and zero limited sibling groups. The first recorded
rejection was on GID 8, logical level 2, near
`rho=0.03125`, `z=-8.09375`. The latest retained history already shows a
large constraint/curvature runaway before the strict gate, so the gate is a
terminal invariant check rather than the first sign of numerical trouble.

N256 summary SHA-256:
`83de74713a5a4b83d0311f82994a8aa581354a83e457805efaa2ace3afdc6c01`.

## Interpretation boundary

Observation:

- The deterministic replay implementation can reproduce N128 bitwise.
- A higher-cell-resolution run on the same physical AMR tree fails much
  earlier than N128, despite correctly replaying every due hierarchy event.
- The N256 failure is not caused by cycle-number scheduling or by the former
  event-6 floating-point microstep.

Inference:

- A simple explanation in which only the N128 parent cells are too coarse is
  disfavored: N256 has twice the linear cell resolution in every replayed
  MeshBlock yet fails earlier on the same physical tree.
- The result instead points toward a resolution-sensitive evolution or
  persistent AMR-interface/transfer instability.

Still unresolved:

- The N128-derived event times need not be optimal for the diverging N256
  solution, so this one comparison does not completely rule out
  variable-specific under-resolution or a missing refinement sensor.
- The evidence does not distinguish active-state chi loss from restriction,
  exchange, physical-boundary, or same-level coarse-refresh provenance at the
  N256 terminal stage.
- No convergence order, continuum instability, or physical critical-collapse
  conclusion is supported.

## Evidence and provenance

Remote root:

`/pscratch/sd/h/hzhu/axisymmetric-cartoon-amr-history-ac75c8d3-v15-20260818`

Authenticated run root:

`run/exact_timestep`

Key remote hashes:

- root `SHA256SUMS` file:
  `d72721e77011c39dec4bbcddd256ba1e7e1938d70903470e9cdbe5cd60209d6f`;
- detached checksum file:
  `bfd4c40351560868da79fa53c72a8e4fa997273dd0250670954b2213b9cd16d6`;
- N128 record log:
  `f34bac8280d81113d850e5c40acf3fec2d8a15e7b7faef0255cd5a9c75aba142`;
- N128 replay log:
  `3e3d7176f8af762700e7ae75d286c9e74079ea940da376f5eee27d0d5accb286`;
- N256 run log:
  `879b2ceaf1a38147c1b4ac1cb9778a9b5ddf99ddf21aa02cc6a448a144a688ab`.

The committed `evidence/v15/` directory contains the identity result, N256
summary, compact ledgers, build/test evidence, and copies of both remote
checksum layers. `evidence_manifest.json` authenticates those committed
files.

## Failed-attempt lineage

- v13/job `57201579`: zero science steps. Its outer allocation environment
  exposed `SLURM_TRES_BIND=gres/gpu:map_gpu:0`; the harness had expected the
  alternative `per_task` spelling.
- v14/job `57201669`: N128 identity was already accepted from v12, but N256
  stopped at event 6 when the scheduler attempted the observed
  `3.885781e-15 M` remainder as an RK step.
- v15/job `57202042`: fresh source/build, fresh N128 authority, exact N128
  replay pass, then the requested one-shot N256 physical-time replay and its
  authenticated strict-chi terminal result.
