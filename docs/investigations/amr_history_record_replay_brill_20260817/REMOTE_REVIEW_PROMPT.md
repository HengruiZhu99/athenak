# Read-only review prompt: Brill AMR history record/replay

Please perform a skeptical, read-only review of the deterministic AMR-history
record/replay implementation and its N128/N256 evidence.

Repository: <https://github.com/HengruiZhu99/athenak>

Branch: `codex/amr-history-record-replay-brill-20260817`

Qualified source commit: `ac75c8d348da91b38cbc6855b5fba51cd3089663`
Qualified source tree: `6284882bd06e8db379495675aba7a4f153fb4afa`

Start with:

- `docs/investigations/amr_history_record_replay_brill_20260817/REPORT.md`
- `docs/investigations/amr_history_record_replay_brill_20260817/evidence_manifest.json`
- `docs/investigations/amr_history_record_replay_brill_20260817/evidence/v15/n128-identity.json`
- `docs/investigations/amr_history_record_replay_brill_20260817/evidence/v15/n256-replay-summary.json`
- implementation commits `7f563038`, `0396d281`, and `ac75c8d3`

## Authenticated result

The fresh N128 authority recorded 210 accepted hierarchy events. N128 replay
matched all 210 events, the terminal event cycle/time/tree, and every common
binary payload bitwise for all six output families. It required zero timestep
clips. This establishes same-resolution deterministic replay.

The one-shot N256 replay doubled cells per MeshBlock while retaining the same
2x4 root MeshBlock layout, physical MeshBlock extents, and N128 hierarchy
schedule keyed by coordinate time. It replayed events 1 through 11 as an exact
authority prefix and used five genuine timestep clips. It then failed at cycle
5547, `t=10.53633 M`, before event 12: boundary prolongation rejected 3928 chi
parent stencils and zero limited sibling groups. Retained history already shows
a large constraint/curvature runaway before that gate.

The N128 authority continued to its final accepted event at
`t=13.622603702545716 M`, level 22, and later failed its known strict-chi gate.
Thus N256 failed earlier despite twice the linear cells per MeshBlock on the
same physical tree.

## Questions

1. Audit whether the record/replay authority, physical-time scheduling,
   tolerance, restart position, and exact-tree checks are logically sound.
2. Does the exact N128 identity provide adequate evidence that replay itself
   is not perturbing the same-resolution solution?
3. How strongly does the earlier N256 failure disfavor simple parent-cell
   under-resolution? What alternatives remain because the hierarchy was
   recorded from N128 rather than generated from the evolving N256 state?
4. Given the first rejected N256 parent near `rho=0.03125`, `z=-8.09375`, what
   is the smallest decisive diagnostic to distinguish active RK chi loss from
   restriction, exchange, physical-boundary, axis-parity, or same-level
   coarse-refresh provenance?
5. Is there a source-level correctness risk in treating event times within
   `32*epsilon*scale` as equal, particularly across restart or long event
   schedules? Suggest a tighter formulation if needed without reintroducing
   microscopic PDE steps.
6. Identify any missing fail-closed test that could invalidate the stated
   record/replay qualification.

Keep observations, inferences, and hypotheses distinct. Please recommend the
single smallest next diagnostic or correction.

Do not propose chi floors or clipping, weakening the strict-positive gate,
broad parameter sweeps, gauge/damping/KO changes, or unsupported convergence,
Figure-3, or physical-critical-collapse claims.
