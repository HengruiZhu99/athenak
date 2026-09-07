# Synchronized reconstruction budget and immediate RHS attribution

Two diagnostic snapshots now bracket `CompleteCoherentTransfer` after ordinary
periodic SMR prolongation. They use the existing opt-in intrinsic stage dumper,
with operation names `pre-coherent` and `post-coherent`, both with valid state
ghosts and no RHS payload. Uniform-mesh dump behavior and stage-zero skipping
are unchanged. No equation, transfer rule, rate or default changed.

The FD6 2D seven-leaf n8 point6_2d fixture runs two RK3 steps at dt0.000125,
for both ordinary and residual transfer. In serial and two ranks, final states
are bitwise equal to the preserved f8d91ed4 executables without snapshots.
All twelve mode/stage budget arrays are bitwise identical between serial and
MPI. Each rank emits 30 stage payloads per arm. A separate collision directory
confirms exclusive-write rejection without modifying the successful run's
files. The 19 legacy restart controls pass.

At all six stages per arm, ordinary transfer preserves active state;
reconstruction preserves every stored primary and every active state value.
Post-coherent and post-exchange snapshots agree exactly. For ordinary transfer,
the coherent correction is exactly zero. For residual transfer, active E
correction is exactly zero, while auxiliary ghost and active curl corrections
are nonzero. Auxiliary ghost corrections reach about 1.18333e-4 and intrinsic
curl component corrections about 4.22170e-5 across these stages. These are
corrections, not errors against an exact evolving solution.

Signed arrays retain before/after reduction, intrinsic curl and Q-curl values,
all component RMS/maxima/signed extrema/cell indices, raw ordinary stored-state
increments and coherent stored-state increments. Derivatives use only valid
pre/post-coherent ghosts. The post-RK state is explicitly stale in its ghosts:
its ordinary-transfer difference is retained as a raw state vector only. This
is not a semidiscrete subsidiary-defect oracle, which would also require a
valid tangent reconstruction of RHS fields at the interface.

## Counterfactual RHS replay

At cycle 0/stage 1 and cycle 1/stage 3, a compiled actual PointRHS replay with
independently assembled FD/KO terms matches the recorded pre-RK RHS to maximum
normalized discrepancies 3.895e-15 and 4.194e-15. The replay then evaluates the
pre/post-coherent states without changing active values. Their first ten RHS
components are exactly unchanged. The strongest RHS correction occurs in C,
followed by K; this measures the immediate consequence of auxiliary ghosts,
not a change to the underlying continuum equations.

Four additional replays replace only p, l, S or B ghosts. Their signed RHS
increments sum to the total within 1.388e-17 at the first sampled stage. The
maximum C correction is at gid 3, active kji=(0,7,7), coordinate
(0.484375,0.6296875,0.85), a corner of the fine block. At cycle 0/stage 1:

| Contribution at the same cell | C RHS change |
|---|---:|
| p ghosts | -3.7091374552873646e-4 |
| S ghosts | -1.6420913631397943e-4 |
| l ghosts | 0 |
| B ghosts | 0 |
| Total | -5.351228818427159e-4 |

At cycle 1/stage 3 the maximum is at the same cell and totals -5.354793663745316e-4.
Family-separated signed arrays and maxima in every RHS component are retained.
For K, p/l/S contribute and partly cancel; one must not sum maxima at different
cells. This is a direct instantaneous attribution from reconstructed p/S ghosts
to the GH source at one measured location. It is not proof that this correction
causes the earlier long-ladder alignment failure, nor evidence of an instability.

Next isolate the scalar p reconstruction/transfer commutator at this corner,
using source residual and primary interpolation data, before changing another
operator. Repeated-stage qualification remains failed; this budget provides a
specific mechanism to investigate. CUDA, physical-wave, puncture and binary
qualification remain incomplete.

## Evidence and reproduction

`qualification-runs-20260907/pcgh-clean-reduction/intrinsic-smr-stage-budget-001/`
contains the frozen plan, compact results, inputs, source/binary manifests and
raw inventory. Large snapshots, signed arrays and replay text streams remain
under the same named external test directory in
`/Users/hz0693/research/pcgh-clean-reduction-tests-20260907/`.

`check_intrinsic_smr_stage_budget.py` accepts `--binary`, `--reference` (the
preserved f8 executable), `--fixtures` (point-restriction n8 inputs), `--output`,
and optional `--ranks 2 --launcher '/opt/homebrew/bin/mpiexec -n 2'`.
`check_intrinsic_smr_rhs_correction.py --dumps <serial residual_shifted-dump>
--kernel <intrinsic_rhs> --output <new directory>` reproduces the located
family decomposition. Analyses ran with the external venv Python and `-W error`.
The original total-only replay and subsequent family/location extensions are
preserved separately. No remote resource was used this turn.
