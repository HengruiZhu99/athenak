# Native-VC AMR root cause

## Verdict

The negative-order 2D/3D dynamic-AMR discriminator had two source-level causes.

1. Lower-side odd fine indices used C++ signed division (`offset / 2`).  C++
   truncates toward zero, so `-1 / 2` selected coarse interval `[0,1]` instead
   of `[-1,0]`.  This corrupted the first lower-side coarse-to-fine ghost layer.
2. The midpoint interpolation order matched the bulk finite-difference order.
   Z4c consumes second derivatives at the interface, so an interpolation error
   `O(h^q)` contributes `O(h^(q-2))` to an interface RHS.  Full bulk order `p`
   therefore requires `q >= p+2`.

A later four-rank 3D/PVC qualification exposed a third, distributed-only
defect.  Remote child-migration packing used the persistent coarse-cache ghost
width (5 for q8) as a halo in the source fine array.  For an upper N16 child it
therefore requested index 25 from storage ending at index 24.  Same-rank child
initialization already used the correct refinement halo `q/2-1 = 3`, which is
why single-rank and some smaller-rank cases passed.  Commit
`480de5f7bd7d510d9c5984a5e5b7dcbf60d2b3a2` makes remote send/receive ranges
use the same bounded refinement halo and adds host-side storage checks.

The primary repair is commit `21b9121339185ba2629ead53b6993596bbc64b62`.
Test expansion and valid multirank fixture geometry are commit
`cb4f173b5326302431c1d18e4edc113750471cef`.

## Localization evidence

Before the repair, the dynamically refined state agreed at coincident shared
vertices immediately after topology construction.  The first nonconvergent
discrepancy appeared in lower-side coarse-to-fine reconstruction at T5 and was
then exposed by T7 RHS evaluation and T8 RK update.  The dominant component was
the `Atilde_zz` RHS near the coarse/fine interface:

| resolution | representative coarse/fine RHS mismatch |
|---:|---:|
| N16 | about `1.3291e-2` |
| N32 | about `2.6887e-2` |
| N64 | about `5.509e-2` |

The approximately `O(1/h)` growth excluded regrid geometry and initial shared
copy disagreement as the first cause.  Raw default-off state/RHS contributor
censuses are preserved as compressed CSV files under `evidence/phase1/`.

Changing signed division to floor division removed that resolution-growing
signature.  With the otherwise unchanged nominal-order transfer, however, the
dynamic-AMR discriminator converged at only about second order.  This separated
the indexing defect from the transfer-order defect.

## Repair semantics

- `FloorHalf` maps negative odd fine offsets to the geometrically correct
  coarse interval.
- Restriction of point-valued VC state is exact injection at coincident
  vertices.
- Prolongation is exact injection in even/collapsed directions and centered
  tensor-product midpoint interpolation only in odd directions.
- Transfer orders are O4/O6/O8 for O2/O4/O6 Z4c bulk discretizations.
- Coarse-cache halo width is derived from fine ghost width and transfer order.
- A MeshBlock that cannot obtain that centered halo in one communication hop
  is rejected with an actionable error; no one-sided or lower-order fallback is
  used.
- Same-level contributors are reconciled symmetrically.  At cross-level
  coincident vertices, only canonical finest-level contributors are
  authoritative, after which exact injection populates the coarse copy.
- Physical and Cartoon-axis coarse ghosts are filled with explicit fine/coarse
  widths before centered prolongation.
- Migrating refined children communicate only `q/2-1` source-halo vertices;
  the wider persistent coarse-cache halo is reconstructed after migration.

## Numerical effect

For O4/RK4, the repaired RMS discriminator changed from negative order to:

| geometry | N16 | N32 | N64 | orders |
|---|---:|---:|---:|---:|
| 2D Cartesian | `2.1422396284e-9` | `1.2344561378e-10` | `5.2347102591e-12` | `4.117`, `4.560` |
| 3D Cartesian | `2.9259661935e-9` | `1.6295833338e-10` | `6.8920689792e-12` | `4.166`, `4.563` |

O2 gives orders `2.094/2.373` in 2D and `2.158/2.380` in 3D.  O6 uses
N24/N36/N48 because the centered q8 halo requires a larger MeshBlock; measured
orders are `5.201/6.469` in 2D and `5.162/6.567` in 3D.  The N48/N96 O6
interval is roundoff-saturated and is not used for a convergence claim.

## What this does not establish

This identifies and repairs the synthetic native-VC AMR defect.  It does not
establish convergence of the Brill-collapse common-tree campaign, explain its
prior resolution divergence, qualify a vertex-centered puncture at `r=0`, or
qualify CUDA/performance.  Those remain distinct gates.
