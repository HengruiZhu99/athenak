# Common-hierarchy symmetric-O4 Brill convergence campaign

This campaign records the accepted `dchi_max=0.01` N256 AMR hierarchy and
replays that exact physical-time/LogicalLocation schedule at N128 and N512.
All three cases use the same 4x8 root MeshBlock lattice, O4 method, RK4,
`CFL=0.15`, gauge, damping, KO dissipation, and physical domain. Only active
cells per MeshBlock change (16, 32, and 64 per active direction).

The campaign is fail closed. `aurora_build_qualify.pbs` must pass before any
long authority or replay segment is submitted. The final evidence tree and
scientific verdicts are populated only from authenticated Aurora runs.

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
80-GB A100; N512 therefore uses the same global
`max_nmb_per_rank=16384` capacity rather than Aurora's four-way split.

The CUDA executable is built freshly on the Perlmutter login node.  Device
binding and runtime tests must pass inside the shared allocation before the
first authority evolution is accepted.  Run the additional static gate with:

```bash
python3 docs/investigations/brill_o4_dchi001_replay_convergence_20260821/test_perlmutter_campaign_contract.py
```
