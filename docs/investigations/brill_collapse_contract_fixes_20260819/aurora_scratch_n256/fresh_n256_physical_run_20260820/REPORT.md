# Fresh Aurora N256 Brill run through the target AMR event

Date: 2026-08-20
Disposition: `bounded_physical_run_pass_with_quantified_amr_constraint_jump`
Scientific qualification claim: **false**

## Exact execution

The AthenaK executable was built on an Aurora login node with
`cmake --build --parallel 64`, as requested, then run on one PVC tile.  The
compiled source is clean commit
`e8781f4057c73a0e97f5802413aefde899e24123`, source tree
`f7a0d222b7f08c035a1cf4c2a783c2d4ff8f8154`, Kokkos
`6739bc623081648af9e752b616d9671527922cbf`, and executable SHA-256
`8d9035cc22f4406788792f585648243e4dc043e484ee5ab6f54ac14009a063ed`.
The head build manifest is `52edddfc76428a39d5bfa1eb5dc364d4bd7392aac7a8beac13abe1663f0eec3c`.

The evolution used the archived N256 input and direct IrisK coefficient
payload.  It began from the physical initial slice at `t=0` in Aurora job
`8768636`.  Its cycle safety bound stopped normally at cycle 1800,
`t=5.639311105449374 M`; the wrapper returned failure only because the old
historical cycle-1722 event identity was not reproduced.  The terminal restart
SHA-256 is
`8771db9d9cf7ed6973513d4f81cd6fe3cc7f8edb58538e21014dd5b804fd058a`.
Job `8768689` continued those exact bytes with only `nlim=-1`, `tlim=10.0`, and
fresh output basenames.  It completed successfully in 10m55s at cycle 3329,
`t=10 M`.  This is one from-scratch trajectory split into two authenticated
segments, not two independently initialized cases.

No production numerical option changed: root `128 x 256`, `32 x 32`
MeshBlocks, O6/RK4, CFL `0.15`, KO `0.02`, `dchi_max=0.01`, telegraph lapse,
Gamma-driver shift, zero Z4c damping, no floors, and high-order AMR transfer.

## Target event

The fresh trajectory reached the physically corresponding refinement at cycle
2833, `t=9.476710063617325 M`.  Old GIDs 28 and 45 cover
`5 <= rho <= 6`, `-1 <= z <= 1` and are the two explicitly refined parents.
Balance also refined neighboring old GIDs 29 and 48.  The hierarchy changed
from 74 to 86 MeshBlocks.

The native proper-volume integral ratios from accepted old state T0 to accepted
new state T5 are:

| integral | T5 / T0 |
|---|---:|
| C | 3.0469579492491485 |
| H | 12.252389385128769 |
| M | 61.773604381064786 |
| Z | 1.0003466317838878 |

Coordinate ring volume changes by `-2.22e-15` relatively and proper volume by
only `6.59e-8`.  AthenaK's Cartoon history/diagnostic measure is already the
proper axisymmetric ring measure; the jump is not a fictitious collapsed-y
normalization effect.

The phase ledger closes to roundoff.  Canonical parent-to-child transfer has
maximum residual `8.88e-16`; active evolved fields change only at algebraic
projection (`L2=0.00875`).  Stored ghost data change during MPI receive and
coarse-to-fine prolongation, as expected.  The fixed-child-lattice constraint
change occurs predominantly between T0 and completed boundary reconstruction
T3 (`L2=607.006`); algebraic projection adds `L2=8.588`, about 1.40% of their
sum.  The worst C change is at `(rho,z)=(5.1328125,-0.0078125)`, one fine cell
from a MeshBlock edge, far from the Cartoon axis, and `0.1328125 M` from the
nearest coarse-fine interface.

This evidence does **not** establish that an individual ghost-fill writer is
wrong.  It establishes a representation/derivative jump localized at the
equatorial MeshBlock edge after a canonically correct transfer and completed
boundary reconstruction.

## Parent-state evidence

The parent self-shadow audit shows that chi is comparatively smooth, whereas K
and Atilde have large edge-local high-order restriction/prolongation residuals:

| family | full PR rel. L2 | edge-band rel. L2 | interior rel. L2 | D2 O6-O4 rel. L2 |
|---|---:|---:|---:|---:|
| chi | 1.67e-4 | 2.44e-4 | 6.11e-5 | 3.95e-2 |
| K | 3.53e-1 | 3.89e-1 | 2.11e-2 | 7.28e-2 |
| Atilde | 2.80e-1 | 3.23e-1 | 3.06e-2 | 5.26e-2 |
| Gammatilde | 1.36e-2 | 2.09e-2 | 2.74e-3 | 9.42e-2 |

The largest K/Atilde self-shadow values occur on GID 28 at the equatorial
block edge.  This strongly supports parent under-resolution or transfer
sensitivity in fields that `dchi` does not directly sense.  It does not, by
itself, distinguish an under-resolved parent solution from a high-order
MeshBlock-edge/interface mode.

## Terminal state and admissibility

At `t=10 M` the run has 98 MeshBlocks, maximum relative level 4,
`dt=9.563893e-4 M`, `C=212.939`, `max|K|=18.6632`, and maximum Kretschmann
`1133.57`.  These are signs of rapidly growing numerical/physical fields, not
a qualification result.  The default-off provenance window from `t>=9.49`
contains 9780 stage/checkpoint rows and 1956 pre-update rows.  It records zero
nonpositive or nonfinite active/coarse/candidate chi values; the minimum active
chi is `0.3575294667572515`.  The run reached its requested bound without a
chi-admissibility or device failure.

## Evidence and limitations

- Combined trajectory: [`fresh_n256_history.png`](fresh_n256_history.png)
- Strict summary: [`SUMMARY.json`](SUMMARY.json)
- Phase/writer plots and ledger: [`analysis/target_event/`](analysis/target_event/)
- Parent self-shadow plot/data: [`analysis/parent_state/`](analysis/parent_state/)
- Raw compact logs/history/provenance: [`segment0/`](segment0/) and
  [`continuation/`](continuation/)
- The full 489 MiB event bytes and restart sequence remain checksum-bound in
  the Aurora run root; they are intentionally not duplicated in Git.

This is one N256 run.  It is not convergence evidence, a Figure-3
reproduction, a horizon result, or proof of a unique production source bug.
No transfer operator was changed or promoted by this execution.
