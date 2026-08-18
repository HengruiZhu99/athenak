---
status: active
last_source_verification: 2026-08-17
owner: mesh-refinement
repository: https://github.com/HengruiZhu99/athenak
base_branch: codex/brill-amr-coarse-cache-coherence-20260817
generated: false
---

# Goal Mode prompt: deterministic AMR-history record/replay and matched Brill resolution test

Work autonomously until AthenaK has a minimal default-off, deterministic,
restart-safe AMR hierarchy record/replay capability, the existing authenticated
cycle-1722 Brill zero-PDE transaction has passed an exact replay gate, and a
bounded N128-to-N256 pilot has compared 32x32 and 64x64 cells per MeshBlock
under the same recorded physical hierarchy and event times.

This goal is a controlled-resolution diagnostic. It is not a new Figure-3
production campaign and not authority to change AMR transfer operators, gauge,
damping, dissipation, CFL, floors, positivity gates, or initial data.

Do not mark the goal complete merely because the higher-resolution run survives
longer. Completion requires the narrow implementation, essential focused tests,
frozen-event replay gate, matched N128/N256 fresh-start pilot, event-local
comparison, strict evidence manifests, and qualified interpretation defined
below. N256-to-N512 is not required for completion and must not be launched
without a later explicit authorization. If any gate fails, stop at that gate
and report the failure without weakening it.

The default resolution comparison is deliberately the cheaper N128-to-N256
pair. Do not substitute an N256-to-N512 pair merely because it more closely
matches the historical event: that follow-up is expected to cost roughly four
times the memory per leaf and roughly eight times the work to the same physical
time. Implement and exercise only the minimum machinery needed for the bounded
N128/N256 decision unless the user later expands the scope.

## 1. Repository and preservation boundary

Use the real AthenaK worktree:

    /home/hzhu/Desktop/research/gr/collapse/worktrees/
      axisymmetric_cartoon_remaining/brill_2d_high_order_restriction

At prompt creation time:

    branch:
      codex/brill-amr-coarse-cache-coherence-20260817
    branch head and evidence/report commit:
      b1b6a4e10ad5515e2e7664814dae076feeaa0d1c
    diagnostic source commit:
      55f9147bc80d574636c47bcd1dac86178d921988
    diagnostic source tree:
      cb2ad270f0675230b77023877dc0fdf93b52cd59
    coarse-cache production fix:
      ab651f0ebd113f8718fefbf6d802976e6b3e8738

Before acting, verify the branch, commit graph, pushed remote, applicable
repository instructions, worktree status, and ownership of every existing
change. Preserve unrelated tracked and untracked work. Do not reset, clean,
overwrite, amend, or silently incorporate user work.

Create a dedicated successor branch and fresh local/remote artifact identities.
Prior completed reports, raw evidence, restart bytes, and remote roots are
immutable inputs.

## 2. Governing evidence and current scientific boundary

Read and verify first:

    docs/investigations/brill_amr_constraint_localization_20260817/REPORT.md
    docs/investigations/brill_amr_constraint_localization_20260817/verdict.json
    docs/investigations/brill_amr_constraint_localization_20260817/
      evidence_manifest.json
    docs/investigations/brill_amr_stitched_transfer_diagnosis_20260817/
      REPORT.md
    docs/brill_amr_hierarchy_causality_20260816/REPORT.md

The authenticated frozen event is:

    case: N256 Brill, A=-0.047
    accepted old state: cycle 1721, t=9.5015625 M
    target transaction: cycle 1722, t=9.50625 M
    topology: 74 -> 86 leaves
    maximum physical level: 2 -> 3
    explicitly refined parents: old GIDs 28 and 45
    target region: rho approximately 5.07--5.15, z approximately 0
    restart SHA-256:
      83e996d2d5069307888a69fff47a7524c2f63f11869fb628630bca54dd5943ea

The local frozen raw phases are under:

    /home/hzhu/Desktop/research/gr/collapse/artifacts/
      brill_amr_constraint_localization_20260817/raw_v5_event/

The governing report establishes:

- the immediate constraint jump is dominated by derivatives across a newly
  formed same-level child seam;
- audited same-level ghost copies are bitwise owner-correct;
- local and sender-stitched O6 derivative reads agree exactly;
- coarse-fine support, the rho=0 Cartoon axis, and algebraic projection are not
  dominant for this event;
- the proper Cartoon history measure is already
  `2*pi*rho*sqrt(gamma)*drho*dz` and contains no fictitious collapsed-y width;
- the upstream mechanism remains unresolved between an under-resolved parent
  representation and independently constructed child seam representations.

The A/B/C hierarchy experiment also established that dynamic case A runs away
while a small frozen hierarchy B reaches `t=12.5 M`, but a larger buffered
frozen hierarchy C fails late. Regridding is strongly implicated, while a
larger frozen fine representation is not automatically stable.

Earlier `dchi_max=0.01` versus `0.02` runs are not a pure resolution test. They
changed refinement timing, hierarchy history, interface locations, and churn.
The longer survival of `0.02` disfavors the simplest statement that merely
refining earlier cures parent under-resolution, but it does not rule parent
under-resolution out.

Keep observations, inferences, and hypotheses separate. Do not claim a source
bug, convergence, Figure-3 reproduction, or physical critical behavior from
survival time alone.

## 3. Goal and intended controlled experiment

Implement two optional modes in generic MeshRefinement infrastructure:

    <mesh_refinement>/amr_history_mode = off | record | replay
    <mesh_refinement>/amr_history_file = <path>

The default is internally `off`. If the keys are absent, do not add them to the
runtime ParameterInput or restart header merely by reading defaults.

The default bounded Brill experiment consists of:

1. an N128 reference run from the original initial data with
   ordinary dynamic AMR in `record` mode;
2. an identical-resolution run from the original initial data in `replay`
   mode, used as the record/replay identity control;
3. an N256-effective, 2x-linear-resolution run from the same original physical
   initial data in `replay` mode, with the same root MeshBlock counts, physical
   MeshBlock extents, LogicalLocation leaf sets, and physical AMR event times.

All three are fresh starts. Do not initialize either replay from evolved field
bytes of the reference run. Each run must independently interpolate the same
authenticated IrisK/global-coefficient initial data onto its own AthenaK grid.

The historical frozen N256 event remains a separate mandatory zero-PDE replay
gate because it is the exact event already understood. The cheaper N128-to-N256
fresh-start pilot uses a different root-block layout and therefore provides a
broad resolution diagnostic, not a direct reproduction of old parents 28 and
45. Preserve that distinction in every report.

## 3.1 Fast and decisive implementation boundary

Implement the smallest reusable capability that closes this experiment:

- one compact history parser/writer and one current-tree-to-target-tree flag
  derivation path;
- one physical-time limiter integrated at the existing final timestep-selection
  seam;
- one restart cursor/hash carrier;
- direct reuse of the existing tree mutation and transfer machinery.

Do not create a database, a general event framework, a new transfer layer,
compression infrastructure, a new scheduler abstraction, or visualization UI.
Prefer a documented canonical JSON-lines format and small pure host-side helpers
that can be unit-tested without a GPU. Reuse the existing AMR-jump hierarchy
control, canonical LogicalLocation collection, restart metadata, and timestep
limiting seams where they are sound.

The implementation sequence must be linear and gated:

1. format/tree-difference helpers;
2. record/off identity;
3. replay/time/restart tests;
4. frozen N256 zero-PDE gate;
5. N128-to-N256 pilot.

Do not begin a later phase while an earlier gate is unresolved. Avoid redundant
fresh builds and remote staging identities; one validated source state and one
fresh production build should serve all matched arms when compatible.

## 4. Exact Brill configurations

Use the existing authenticated N256 input as the numerical-parameter contract
and frozen-event contract:

    docs/investigations/brill_r16_resolution_isolation_20260815/data/
      brill_r16_n256_dchi001.athinput

Input SHA-256 at prompt creation:

    b0c7f6d998bfbf501aa2e608e3a1aebfbd050f8888d321b1270e971252af964a

The accepted global-coefficient payload must remain exact:

    docs/investigations/brill_local_telegraph_scaling_20260814/data/
      brill_global_48x32.coefficients
    SHA-256:
      ff0993c390513c15d6aa65857a0a3c710f2e2c3faf5717d9d63245203ccf2d6b

Preserve for both resolutions:

- domain `rho in [0,16]`, `z in [-16,16]`, collapsed `y`;
- identical physical domain and physical MeshBlock bounds within each matched
  record/replay pair;
- Cartoon SO(2), direct global-coefficient IrisK import, Brill `A=-0.047`, and
  pre-collapsed `psi^-2` lapse;
- O6 bulk finite differences, RK4, CFL 0.15, KO dissipation 0.02;
- max-K-scaled telegrapher lapse `(tau,kappa)=(1,1)` and the existing Gamma
  driver shift parameters;
- zero Z4c constraint damping, `floor_chi=false`, and unchanged strict chi
  positivity gates;
- dynamic reference authority `dchi_max=0.01`, derefinement threshold
  `0.25*dchi_max`, `ncycle_check=1`, and `refinement_interval=1`;
- the same maximum logical refinement level and physical boundaries.

The mandatory frozen N256 replay gate retains the historical geometry:

    <mesh>      nx1=128, nx2=256, nx3=1
    <meshblock> nx1=32,  nx2=32,  nx3=1

This is a `(4,8,1)` root-MeshBlock layout. It is not the default fresh-start
reference in this goal.

The default N128 fresh-start reference and identical replay use:

    <mesh>      nx1=64,  nx2=128, nx3=1
    <meshblock> nx1=32,  nx2=32,  nx3=1

The matched N256-effective replay uses:

    <mesh>      nx1=128, nx2=256, nx3=1
    <meshblock> nx1=64,  nx2=64,  nx3=1

Thus both pilot resolutions have exactly `(2,4,1)` root MeshBlocks and
identical physical MeshBlock bounds. The matched N256-effective replay is not
the historical N256 configuration, whose 32x32 blocks form a `(4,8,1)` root
layout.

Only a separately authorized exact-event N256-to-N512 follow-up would use:

    <mesh>      nx1=256, nx2=512, nx3=1
    <meshblock> nx1=64,  nx2=64,  nx3=1

paired with the historical N256 `(4,8,1)` layout. Do not launch that follow-up
under this prompt's default execution.

For every pair, `root_grid_cells / cells_per_meshblock` must be exactly equal
between record and replay. The history is expressed in logical blocks, never
global cell indices. A high-resolution replay must not evaluate `dchi` as
authority; recorded topology is authoritative even if the replay state would
have made a different refinement decision.

## 5. Record-mode contract

Record mode must leave the existing criterion, hierarchy controls, 2:1 balance,
tree transaction, load balancing, and field transfers unchanged.

Rank zero records:

1. the initial accepted leaf hierarchy before the first evolved step;
2. every accepted topology-changing transaction after the complete balanced
   tree is known.

Record complete accepted leaf sets rather than requested flags. For each event
store:

- schema version and monotonically increasing event index;
- coordinate time in canonical round-trip decimal and hexadecimal Real form;
- cycle as provenance only;
- canonically sorted `(level,lx1,lx2,lx3)` leaves;
- leaf count and maximum level;
- deterministic checksum over a precisely documented canonical byte encoding.

The file header must bind:

- dimension, symmetry mode, coordinate system, root level, refinement ratio,
  root block counts, domain bounds, maximum level, and Real representation;
- source/history schema identity;
- cells per MeshBlock as provenance, while explicitly allowing this field to
  differ in a compatible replay;
- complete-file or finalized-prefix integrity information.

Flush each completed event deterministically. A partial/truncated final record
must be detectable and must never be treated as accepted history.

Record mode must be restart-safe: on restart, verify the existing file prefix
and resume without duplicating, skipping, truncating, or forking events.

## 6. Replay-mode contract

Replay mode replaces only refinement-decision authority. It must bypass:

- the ordinary refinement criterion;
- `refinement_interval` and derefinement hysteresis;
- Cartoon mirror reconciliation or diagnostic hierarchy controls that would
  change already-authoritative replay flags.

It must still use the existing production functions for tree mutation, 2:1
balance, redistribution, restriction, prolongation, communication, ghost/BC
work, algebraic projection, ADM reconstruction, constraint evaluation, and
timestep rebuilding. Do not implement a parallel transfer path.

For every event:

1. parse and validate the target leaf set before mutating production state;
2. derive the unique current-to-target refine/derefine flags;
3. apply those flags to a shadow logical tree using production-equivalent
   refine-before-derefine and 2:1-balance semantics;
4. require the shadow leaf set to equal the target exactly;
5. require the target to be reachable in one ordinary AMR transaction;
6. only then apply the flags to the production tree and execute the ordinary
   production transfer path;
7. compare the post-transaction accepted leaf set to the requested set exactly.

If validation fails, exit before production-tree or field mutation. Do not
perform several hidden refinement generations at one time in order to catch up.

Replay compatibility must require identical physical domain, root block counts,
dimension, symmetry, refinement ratio, root level, and maximum-level capacity.
It must permit different cells per MeshBlock.

## 7. Physical-time scheduler

Replay events are keyed to physical time, never cycle number.

After the normal CFL calculation and global MPI minimum, limit the proposed
timestep by the next replay event time as well as `tlim`. If the next step would
cross an event, set the step to land on it. Define and test:

- canonical Real parsing and comparison;
- an event already due at the current time;
- no zero-length step;
- roundoff/ULP tolerance and deterministic event-time snapping;
- output or restart schedules coincident with an AMR event;
- restart immediately before and immediately after an event.

An event is applied after the state has advanced to its recorded time, matching
the existing Driver ordering. Every rank must agree on the next event and
clipped timestep.

Restart metadata must bind the complete history-file digest, last applied event,
next event, and whether the stored tree is post-event. Do not reconstruct the
cursor from time alone.

## 8. Deterministic diagnostics

Emit one concise rank-zero record per written or replayed event, for example:

    AMR_HISTORY_RECORD event=... time_hex=... cycle=... leaves=... \
      max_level=... checksum=...

    AMR_HISTORY_REPLAY event=... time_hex=... requested_leaves=... \
      accepted_leaves=... max_level=... checksum=... exact_match=true

Also archive a machine-readable event ledger with current/target checksums,
created/deleted leaf counts, balance-induced counts, rank count, and source
identity. Diagnostic output must not become part of the numerical authority.

## 9. Minimal decisive automated tests

Do not build a broad test campaign. Close the following six gates with the
smallest meshes and shortest runtimes that exercise the real path:

1. **Pure host contract test:** parse/write round trip; canonical sorting and
   checksum; current-to-target flags; 2:1-balance target; malformed/truncated,
   nonmonotonic, duplicate/overlapping, incomplete, incompatible, unreachable,
   and wrong-hash rejection before mutation.
2. **Off/record identity micro-run:** absent options, explicit `off`, and
   `record` produce identical event-boundary hierarchy and evolved bytes; only
   record mode writes a history.
3. **Replay identity and 2x topology micro-run:** same-resolution replay is
   event-byte exact under one decomposition; doubled cells per MeshBlock retain
   identical LogicalLocations and physical block bounds.
4. **Physical-time micro-run:** use a deliberately different CFL so at least one
   step is clipped; require exact event times, no zero step, and no repeat/skip.
5. **Restart/MPI micro-run:** restart immediately adjacent to one event and
   replay under a different small MPI ownership decomposition; verify cursor,
   history hash, hierarchy, and record-prefix continuation. If local MPI is
   unavailable, run this single focused case on the eventual target system and
   record the limitation locally.
6. **Frozen cycle-1722 gate:** the production zero-PDE replay in the next
   section is the authoritative end-to-end integration test.

Use pure host tests for format and tree logic, one small serial integration
fixture, and at most one small MPI fixture before the frozen gate. Do not create
separate tests for every malformed input if a table-driven fixture closes them
all. Do not require a GPU until the frozen production-path gate.

## 10. Mandatory frozen cycle-1722 replay gate

Before any from-scratch comparison, exercise the implementation against the
same frozen event used by the preceding investigations.

Construct or authenticate a minimal history containing:

- the cycle-1721 accepted 74-leaf tree at `t=9.5015625 M`;
- the cycle-1722 accepted 86-leaf tree at `t=9.50625 M`.

Start from the exact restart SHA listed above. Use the same source, executable,
MPI decomposition, input bytes, and target-only zero-PDE stop contract for the
record-derived and replay arms.

Require:

- replay lands at the exact target physical time;
- requested and accepted leaf sets match the frozen T1/T2 topology exactly;
- created/deleted and balance-induced mappings match;
- the same-resolution replay T2 active Z4c bytes match the frozen production
  transfer bytes exactly, or stop with a precisely isolated deterministic
  explanation;
- T0--T5 proper-ring constraints reproduce the authenticated values within the
  already recorded tolerance;
- no extra RHS step, later cycle, transfer mode, or numerical parameter change.

This gate validates replay semantics. It is not a new scientific evolution and
does not authorize a long campaign.

## 11. Bounded fresh-from-initial-data comparison

Only after all preceding gates pass, run the three fresh-start pilot arms:

### P-R: N128 dynamic record reference

- 32x32 cells per MeshBlock;
- global root cells 64x128 and exactly 2x4 root MeshBlocks;
- ordinary authenticated `dchi_max=0.01` dynamic AMR;
- record the accepted hierarchy;
- start directly from the authenticated IrisK/global-coefficient initial data,
  not the cycle-1721 restart.

### P-I: N128 identical-resolution replay

- 32x32 cells per MeshBlock;
- global root cells 64x128 and exactly 2x4 root MeshBlocks;
- start independently from the same initial data;
- replay P-R's hierarchy schedule;
- use as the numerical identity/control arm.

### P-H: N256-effective, 2x-linear-resolution replay

- 64x64 cells per MeshBlock;
- global root cells 128x256, preserving the same 2x4 root blocks and physical
  block bounds;
- start independently from the same spectral initial data evaluated on the
  higher-resolution grid;
- replay P-R's exact hierarchy and physical event times.

Run P-R until its first strict numerical failure or the existing prospective
input endpoint `t=20 M`, whichever comes first. Its last fully accepted state
defines the common comparison endpoint. Run P-I and P-H until their own earlier
strict failure or that exact accepted P-R endpoint. Never extrapolate the
recorded hierarchy beyond P-R's accepted schedule, and do not freeze the final
tree to continue a replay tail under this experiment.

Do not substitute the historical N256 cycle-1722 schedule into the pilot. P-R's
new authenticated schedule is authoritative for P-I and P-H. The historical
event remains the separate frozen implementation gate.

Stop all arms fail-closed at the first strict numerical invariant failure. Do
not retry-select, extend the endpoint, or replace a failed arm after inspecting
results.

## 12. Event-local comparison required for interpretation

At every matched pilot event, compare P-R, P-I, and P-H using the correct
proper-ring measure and physical coordinates. Give special attention to every
level-2-to-level-3 event whose parent physical bounds intersect
`rho in [5,6], z in [-1,1]`. Compare the historical N256 cycle-1722 event only
within its separate frozen gate.

Archive:

- exact event time, LogicalLocation set, physical block bounds, leaf count, and
  maximum level;
- pre-event and immediate post-transfer evolved fields;
- constraints before transfer, after transfer/ghost/BC work, and after
  projection/recomputation;
- proper-ring C/H2/M2/Z2 integrals, RMS values, maxima, and locations;
- parent high-frequency/derivative-disagreement indicators;
- same-level seam value jumps and O2/O4/O6 normal-derivative disagreement;
- timestep, MeshBlock count, and topology-event ledger;
- strict finite and positive-chi gate telemetry.

Compare resolutions at common physical points or after an explicitly documented
restriction to the N128 pilot lattice. Never compare equal array indices or
native-grid Nyquist fractions as if they represented equal physical scales.

For P-R versus P-I, require the strongest practical identity check. A
meaningful P-R/P-I mismatch invalidates the P-H interpretation until explained.

## 13. Decision rules

Choose exactly one primary disposition:

### `REPLAY_IDENTITY_FAILED`

Record/replay changes the identical-resolution evolution or cannot reproduce
the frozen event. Do not interpret the high-resolution arm.

### `PARENT_UNDER_RESOLUTION_SUPPORTED`

P-R and P-I agree; P-H has materially smaller pre-event
high-frequency/derivative disagreement and smaller immediate seam/constraint
jump on the same physical tree and event schedule.

State that this supports a resolution-dependent parent/transfer error. Do not
claim that bulk parent evolution is uniquely isolated unless the pre-transfer
comparison, rather than only the post-transfer jump, improves.

### `TRANSFER_OR_INTERFACE_RESOLUTION_SENSITIVE`

P-R and P-I agree; pre-event parent states are comparably resolved on a common
physical lattice, but P-H has a smaller immediate post-transfer seam jump.
This implicates resolution dependence in AMR transfer/interface closure.

### `RESOLUTION_DOES_NOT_CURE_MATCHED_HIERARCHY`

P-R and P-I agree; P-H exhibits the same localized jump or invariant failure
at the same event and physical time within prospective tolerances.

### `MIXED_OR_INCONCLUSIVE`

Differences in both pre-event and post-transfer diagnostics prevent a unique
classification, or the runs diverge before the matched event.

Survival time is secondary evidence only. No disposition establishes
convergence from two resolutions.

## 13.1 Conditional exact-event N256-to-N512 follow-up

Do not launch this follow-up during the default goal. At completion, recommend
it only if all of the following hold:

- frozen N256 replay identity passes;
- P-R/P-I identity passes;
- the N128-to-N256 pilot reaches its prospective target;
- the pilot supports resolution sensitivity but cannot distinguish parent
  evolution from transfer/interface truncation, or a direct check of the exact
  historical parents 28 and 45 remains scientifically necessary;
- a prospective cost and memory estimate confirms the bounded run fits the
  selected allocation.

If later authorized, use historical N256 with 32x32 cells per block as record
reference and N512-effective with 64x64 cells per block as replay, both on the
same `(4,8,1)` root-block layout. Expect approximately four times the memory
per leaf and, because the CFL timestep is about half, approximately eight times
the work to equal physical time. Retain the same event-local gates and stop no
later than the first complete diagnostic after the historical target window.

## 14. Resource and execution discipline

- Perform all parsing, unit tests, format tests, and shadow-tree tests locally.
- Use the minimum remote resources needed for production MPI/CUDA validation.
- Use fresh immutable remote roots and exact job names.
- Stage and run one campaign identity at a time.
- Confirm source, executable, input, initial-data, history, rank/GPU binding,
  and output hashes before interpreting results.
- Monitor to terminal accounting and archive both success and failure evidence.
- Do not reuse a partial or failed remote identity.
- Do not run beyond P-R's first strict failure or `t=20 M`, nor beyond the
  corresponding last accepted P-R state in either replay arm.

## 15. Required deliverables

Produce:

1. implementation summary and exact files changed;
2. documented history schema and canonical checksum definition;
3. source tests, commands, logs, and results;
4. frozen cycle-1722 replay report and strict evidence manifest;
5. fresh P-R/P-I/P-H event ledger and comparison tables;
6. constraint, seam-derivative, timestep, topology, and MeshBlock-count plots;
7. concise final `REPORT.md` separating observations, inferences, hypotheses,
   unsupported claims, and limitations;
8. strict checksummed artifact manifest and detached checksum;
9. a short `REMOTE_REVIEW_PROMPT.md` pointing a read-only reviewer to the exact
   branch, commits, report, evidence, and remaining questions.

Commit only owned, reviewed files in logical commits. Push the working branch
and verify the remote commit. Do not claim completion until remote identity and
all evidence manifests verify.

## 16. Prohibited actions and qualification boundary

Do not:

- modify restriction/prolongation formulas or transfer modes;
- change `dchi`, gauge, damping, KO, CFL, boundary conditions, centering, or
  physical domain; the only authorized hysteresis change is the prospective
  derefinement threshold `0.25*dchi_max`;
- add floors, clipping, filtering, or weakened finite/positive-chi gates;
- use cycle number as replay authority;
- initialize P-H by interpolating an evolved N128 state;
- launch the conditional N256-to-N512 follow-up without new explicit authority;
- perform a broad parameter sweep or run beyond the common accepted endpoint;
- claim convergence, Figure-3 reproduction, or physical critical behavior.

The capability may be generic and production-quality while the Brill result
remains scientifically inconclusive. Report that boundary honestly.
