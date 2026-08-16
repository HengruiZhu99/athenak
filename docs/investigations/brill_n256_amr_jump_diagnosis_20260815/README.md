# N256 Cartoon Z4c AMR constraint-jump diagnosis

## Outcome

The cycle-1724 history jump is real, but `1724` is the history-sample label,
not the AMR-transaction label.  The accepted 74-to-86 MeshBlock, maximum-level
2-to-3 transaction occurs at runtime cycle `1722`, time
`9.506249999999907 M`.  Job 57052389 captured that transaction on four CUDA/MPI
ranks, closed the T0--T5 ledger, evolved exactly eight further cycles, and
produced the required plots and tables.

The quantitative disposition is `quantified_multi_stage`, with
`qualification_claim=false`.

The jump is **not** caused by a fictitious collapsed-direction width or missing
axisymmetric normalization.  Coordinate ring volume is constant to roundoff,
and proper volume changes by only `-2.711e-8` relative from T0 to T5.  Instead,
the jump is produced when constraints are evaluated on the newly reconstructed
hierarchy: active evolved values survive refinement to roundoff, while MPI
receive and coarse-to-fine prolongation populate different ghost/interface
representations.  Algebraic projection adds a smaller, separately measured
contribution.

## Quantitative result

Across the immediate T0-to-T5 transaction:

| integral | before | after | integral factor | square-root factor |
|---|---:|---:|---:|---:|
| C | 14.1510600 | 88.2282141 | 6.23474 | 2.49695 |
| H | 0.562379676 | 12.8947501 | 22.9289 | 4.78841 |
| M | 0.775713649 | 62.5304751 | 80.6103 | 8.97832 |
| Z | 2.19527556 | 2.19646484 | 1.00054 | 1.00027 |

The history row at cycle 1724 is two accepted evolution cycles later, so it is
recorded as a non-phase-identical reference, not falsely compared byte-for-byte
to T5.  The replay reproduces that row exactly within floating-point reduction
order: `C=97.7302686063`, `H=32.6010797129`, `M=52.3237200429`, 86
MeshBlocks, and maximum level 3.

All accounting closes:

- parent-to-child coarse-source residual: `0`;
- canonical O6 transfer residual: `1.776e-15` versus `6.451e-12` tolerance;
- evolved-state telescoping residual: `0`;
- constraint telescoping residual: `8.882e-16` versus `5.632e-10` tolerance.

The active refine/derefine transfer changes the evolved state by only
`2.832e-14` in unweighted L2.  MPI receive and coarse-to-fine prolongation make
no active-cell change, but change stored ghost/interface data by L2
`42.9445` and `17.3583`, respectively.  When valid derivative halos are
complete at T3, the fixed-lattice constraint-field change from T0 is
`615.250` in L2.  Algebraic projection contributes `9.52149` more, only
`1.55%` of the T0-to-T3 magnitude.  ADM/constraint recomputation adds zero.

Intermediate T3 states intentionally lack complete derivative halos.  It is
therefore not mathematically valid to compute separate constraints immediately
after MPI receive and immediately after prolongation.  Their evolved ghost-data
changes are individually quantified, but their constraint effects form one
valid completed-boundary contribution.  This is the explicit remaining
subphase-resolution limit; it does not prevent the broader multi-stage
classification.

## Spatial and temporal behavior

The worst fixed-lattice change is a reflection-symmetric pair near
`(rho,z)=(5.1328125,+/-0.0078125)`.  It is far from the axis, one half-cell
from a MeshBlock edge, and `0.1328125` from the coarse-fine interface.  The
worst `delta_C` is `147.456`.  This rules out a symmetry-axis origin for this
event and strongly associates the jump with the new block/interface derivative
representation.

The jump does not relax as an instantaneous quadrature artifact.  Over the
eight accepted post-event cycles, C grows from `88.2282` to `132.287` and H
from `12.8948` to `107.622`; M decreases from `62.5305` to `11.7122` while the
state remains finite.  The different families evolve rather than sharing a
common measure rescaling.

The diagnostic-only convex sibling average never disagrees with a positive
production chi target: zero positive-source/nonpositive-target cases were
found.  The largest high-order-versus-shadow relative chi difference is
`0.108%` in an edge-center class.  Thus chi restriction overshoot is not the
trigger for this early constraint jump.

## Operational qualification

The N256 science step and analyzer completed.  The outer allocation then exited
1 because its final restart check searched only `run/n256/*.rst`; all 15 restart
files, including the final `.00014.rst`, are actually under `run/n256/rst/`.
This is a post-science harness defect and has no effect on the recorded result.
The final restart is preserved locally.

This repository directory is the curated review package: verdicts, plots,
derived ledgers, history, run log, accounting, and the complete raw-ledger hash
inventory.  The full 4.2 GiB raw T0--T5 dumps and final restart remain in the
local campaign artifact root rather than Git; their frozen hashes are recorded
in `final_verdict.json` and `evidence/raw-ledger.SHA256SUMS`.  All listed
remote-root files, the detached manifest, the analysis manifest, and the local
raw manifest verify.

During the terminal audit, a verification command wrote `manifest-check.log`
and `manifest-check.status` into the remote root after its manifest inventory
was frozen.  These two audit-only files are unlisted by the manifest, copied
locally, and did not alter any listed evidence byte.

## Recommended next step

The cleanest controlled follow-up is O6 bulk evolution with a runtime-selectable,
genuinely limited-O2 Z4c AMR transfer pair at both regridding and ordinary
coarse-boundary reconstruction.  Retain the same phase provenance diagnostic.
If the jump disappears, the interface ghost representation is central; if it
remains at the same location and magnitude, the bulk or persistent AMR coupling
is responsible.  No such repair or ablation was performed here.

No transfer operator, gauge, damping, dissipation, AMR threshold, floor,
timestep policy, or initial-data byte was changed.  This artifact does not
qualify Figure 3.
