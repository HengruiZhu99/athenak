# Common-hierarchy symmetric-O4 Brill convergence campaign

This campaign records the accepted `dchi_max=0.01` N256 AMR hierarchy and
replays that exact physical-time/LogicalLocation schedule at N128 and N512.
All three cases use the same 4x8 root MeshBlock lattice, O4 method, RK4,
`CFL=0.15`, gauge, damping, KO dissipation, and physical domain. Only active
cells per MeshBlock change (16, 32, and 64 per active direction).

The campaign is fail closed. `aurora_build_qualify.pbs` had to pass before any
long authority or replay segment was submitted. At the user's direction,
production then moved to Perlmutter `shared_interactive`; the final evidence
tree combines the authenticated Aurora qualification with the complete
Perlmutter production triplet and offline analysis.

## Aurora qualification

Aurora debug job `8770096` passed on source `16931a5f9830e7c8a75a9b72e93c4c7230cb6906`
with Kokkos `6739bc623081648af9e752b616d9671527922cbf`. It built the common
MPI+SYCL/PVC executable, passed all 12 selected CTests, passed explicit MPI2
coarse-cache and record/replay checks, exercised the production state-failure
extractor on PVC, and completed short N256-record/N128-replay/N512-replay
checks. Every exercised replay event landed at zero ULP and reproduced its
authority tree exactly. The immutable remote qualification root is
`/lus/flare/projects/CompactBinaryMerger/hzhu/brill_o4_replay_16931a5f_20260821/qualification`;
selected evidence is copied under `evidence/qualification/`.

Failed predecessors remain classified rather than promoted: job `8770069`
stopped before the AthenaK build because its fresh clone lacked the Kokkos
submodule, and job `8770076` exposed a brittle static-test token plus a real
714-ULP early 2x replay event. The latter led to the exact-time scheduling
repair qualified by `8770096`.

## Production segmentation

`aurora_run_segment.pbs` is the single fail-closed segment runner. It accepts
only the frozen N128 replay, N256 record, or N512 replay geometry/resource
tuples. N256 must run first from fresh initial data and owns the only mutable
history file. Restart segments authenticate both their restart and the
pre-segment history. N128 and N512 authenticate and only replay the finalized
N256 history.

The runner writes history every four cycles, restart checkpoints every 512
cycles, and `z4c`, constraints, and Weyl fields every 128 cycles. The common
cycle cadence makes the physical sampling interval scale with `h`, permitting
fourth-order temporal interpolation to the requested common central proper
times. Unneeded ADM, auxiliary diagnostic, and telegrapher-mu field dumps are
disabled; their required scalar diagnostics remain in history output.

Run the local static/evidence gate before submission:

```bash
python3 docs/investigations/brill_o4_dchi001_replay_convergence_20260821/test_campaign_contract.py
```

## Perlmutter production successor

Aurora job `8770135` was cancelled while still queued, before allocation or
science, when the user authorized moving production to a single A100 in
Perlmutter `shared_interactive`.  The platform-specific wrappers are
`perlmutter_allocate_segment.sh` and `perlmutter_run_segment.sh`.  They retain
the same frozen physics, common 4-by-8 physical MeshBlock lattice,
record/replay semantics, `CFL=0.15`, exact event-time landing, output cadence,
and fail-closed evidence policy.  All three resolutions use one rank and one
80-GB A100.  N128 and N256 use `max_nmb_per_rank=16384`.  N512 uses the
reviewed capacity 2048: the immutable N256 authority tree peaks at 1166
leaves, while job 57347610 proved that 16384 is wasteful and fails during
initialization (`u_adm` alone requested 8.227 GiB).  The lower capacity changes
no accepted replay tree and retains 1.76x headroom over the complete authority
schedule.

The CUDA executable is built freshly on the Perlmutter login node.  Device
binding and runtime tests must pass inside the shared allocation before the
first authority evolution is accepted.  Run the additional static gate with:

```bash
python3 docs/investigations/brill_o4_dchi001_replay_convergence_20260821/test_perlmutter_campaign_contract.py
```

## Offline analysis

`scripts/analyze_common_tree_histories.py` merges restart-overlapped Athena
histories, validates the executed N128/N512 replay prefixes against the N256
authority checksums and event times, reduces native-AMR shadow decisions,
computes trusted-window scalar-constraint orders and authority-event jumps,
and produces the scalar history plots.  It requires an explicitly reviewed
`--trusted-tau-max`; it intentionally does not promote field convergence or a
Figure-3 overlay, which remain separate binary-sampling and reference-curve
gates.  `scripts/test_analyze_common_tree_histories.py` exercises restart
overlap, exact replay, shadow reduction, plotting, and an exact synthetic
fourth-order triplet.

`scripts/analyze_common_tree_fields.py` samples retained binary fields onto a
common meridional lattice and measures ring-coordinate self convergence.
`scripts/plot_common_tree_figure3.py` applies the already authenticated
rendered-paper Figure-3 curves. `scripts/analyze_timestep_contracts.py`
separates exact-event replay clips from the underlying Z4c timestep candidate.

The completed disposition and evidence boundaries are in [REPORT.md](REPORT.md)
and [comparison_summary.json](comparison_summary.json). The primary result is
`O4_NONCONVERGENT`: replay is exact for every executed prefix, but trusted
fields and constraints do not converge and N512 fails earlier than N128.
