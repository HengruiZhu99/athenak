# Matched native N12 timing and stage-observer control

The current goal is two-node performance parity, using an already validated
saved state. The former fresh-through3551 prerequisite and eight-node timing
campaign are superseded for this experiment. Preserve their tools and evidence.

Start all native controls from the unchanged ordinary cycle2486 checkpoint,
SHA256 `dd4b86ab42c10827f14bd339930cb4d2ffaf9fb5e2b43953d1e5d7924a9fc1ec`.
Its complete historical field/history/diagnostic and independent-review gates
already pass. Previous-implementation controls use a separately verified exact
conversion of this same state; retaining merely similar historical fields is
not an exact common-state conversion. Import all eight native chi-history
members into the legacy correctness helper because SP1 omits those records.
Keep the original SP1 and native files intact.

Run three native cases at24 GPU-tile ranks,12 per node:

1. Unchanged normal executable f3a85db7: numerical/output control only.
2. Qualified whole-cycle-timed normal executable e4788800: primary native cost.
3. Timed observer: link the qualified observed main object and the qualified
   timed driver object together with the original native numerical objects.

`native_timed_observer.py` performs only that isolated link with make-j64. Both
input builds must share exactly the same native definitions, compile database,
link recipe and baseline executable. Preserve all FP/device policies and prove
that only main, driver and executable-output tokens changed. No runtime source
or existing timing/observer source is modified. The separate inline observer
and timer namespaces add no changes to native class definitions.

Use10 warmup cycles plus3 windows of20:70 complete cycles,2486→2556, at the
unchanged dt0.00046944203735617445. Set only time/nlim=2556, time/tlim=2 and
time/ndiag=1000000000. Retain ordinary output metadata; its next scheduled
diagnostic is outside these70 cycles. The unchanged timing preflight must
confirm this rather than suppressing output. Final outputs occur normally and
are checked separately. Both timed cases use identical existing timer code,
maximum-rank elapsed time and window-boundary fences/barriers; no per-stage
timing probe is added. Detailed profiling is disabled. The observed case retains
its original per-stage reductions, collectives and CSV writes so their complete
cycle cost is measured. Observer installation and Finish occur outside the
timed cycles and are not included in the reported per-cycle observer overhead.

Before accepting any timings, require all three final states to agree bitwise
across every padded31-field/eight-record scalar, initialized checkpoint metadata,
physics/atlas/topology, unit costs and parameter maps. Report the inherited nine
unused coarse-index header integers separately, as existing tests do. Require
the ordinary history, horizon summary and full169-coefficient shape outputs to
be present, finite and byte-identical. Require280 ordered observer stage rows,
exact stage clocks/dt, finite diagnostic values and compatibility≤1e-7.
Each timed case must report exactly the three20-cycle windows with no scheduled
or timed outputs, no timed console writes and no loaded profiling library.

Retain all three window times, their median/min/max, startup, warmup and final
diagnostic/I/O costs. Report observer overhead as the measured timed-observer
median relative to the timed-normal median, with the raw window spread. The
normal control's whole-run wall time is not the primary solver throughput.
Compare the primary native median to the matched fastest validated applicable
legacy implementation using the same cycle/window workload. Three windows are
the first bounded sample; if parity/improvement is indistinguishable from noise,
collect further repeated measurements before accepting a performance claim.

All commands, actual job/rank/tile binding, environment, source checkpoint,
executable/build proofs and before/after frozen-input checks must be recorded.
Allocations are serialized and limited to two nodes. This document authorizes
no submission; the parent reviews the concrete run package before Donor submits.
