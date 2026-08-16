# Brill-wave N256 AMR hierarchy-causality audit

Date: 2026-08-16  
Source branch: `codex/brill-amr-frozen-hierarchy-20260816`  
Numerical source commit: `21a268e4735308a39ac4f040d3621ea114b4ef1d`  
Source tree: `394aa38e76951249de0f247c3a893e0af4a0f1d9`  
Perlmutter campaign: job `57098562`

## Concise verdict

The matched continuation provides **high-confidence evidence that continued
dynamic regridding is causal to the runaway seen in case A**: A underwent 67
actual topology changes, refined to level 12, and timed out near
`t=11.9547843933 M`; B allowed the identical level-2-to-3 event and then froze
the hierarchy, remaining bounded through `t=12.5 M`.

This does **not** isolate a complete source-level bug. The larger buffered
frozen hierarchy C still became catastrophically unstable and failed the
unchanged strict-positive chi parent-stencil gate near `t=12.4828125 M`.
Therefore “freeze any hierarchy” is not a production remedy: the represented
fine state, MeshBlock-edge closures, and persistent multilevel coupling still
matter. No convergence or Figure-3 reproduction is claimed.

## Matched setup

All cases start from the same cycle-1721 restart, SHA-256
`83e996d2d5069307888a69fff47a7524c2f63f11869fb628630bca54dd5943ea`,
immediately before the level-2-to-3 event at cycle 1722 and
`t=9.50625 M`. Fixed numerical choices include N256, `dchi_max=0.01`,
derefinement ratio 0.5, O6 spatial differencing, RK4, CFL 0.15, KO
dissipation 0.02, the same gauge/damping/initial-data bytes, high-order AMR
transfer, and the unchanged strict chi checks.

| Case | Hierarchy control | Terminal disposition | Level / MeshBlocks | dt/M | C norm | Topology changes | cumulative X_CF |
|---|---|---|---:|---:|---:|---:|---:|
| A | dynamic | step timeout at `11.9547843933 M` | 12 / 350 | `4.57764e-6` | `3.42820e7` | 67 | 1,130,760 |
| B | freeze after target | reached `12.5 M` | 3 / 86 | `1.56250e-3` | 86.3261 | 1 | 398,736 |
| C | buffered freeze | strict chi-parent failure at `12.4828125 M` | 3 / 128 | `2.34375e-3` | `2.39261e32` | 1 | 549,072 |

Case C rejected 346 chi parent stencils and zero limited sibling groups.
Case B2 was prospectively conditional on B failing and was correctly skipped
because B reached the requested terminal time.

The comparison plots are:

- [`campaign/constraint_histories.png`](campaign/constraint_histories.png)
- [`campaign/curvature_histories.png`](campaign/curvature_histories.png)
- [`campaign/hierarchy_histories.png`](campaign/hierarchy_histories.png)
- [`campaign/coarse_fine_exposure.png`](campaign/coarse_fine_exposure.png)

## The constraint jump is not a collapsed-y normalization artifact

`Z4cDiagnosticCellMeasure` already uses the physical axisymmetric ring
measure in Cartoon mode:

```text
2 pi rho * dx1 * dx2 * sqrt(abs(det(gamma)))
```

There is no fictitious `dx3` factor. At the target event the diagnostic
coordinate ring volume is conserved to roundoff:

```text
T0: 25735.927018208145
T5: 25735.927018208087
```

The proper volume changes only from `47724.299054367882` to
`47724.297760549001`, while the reported C/H/M integrals change from
`14.15096 / 0.562303 / 0.775692` to
`88.2281 / 12.8947 / 62.5305`. The jump is consequently a real change in the
represented derivative/constraint fields, not a hierarchy-dependent history
measure.

## Target-event phase/writer audit

The cycle-1722 event ledger closes at roundoff:

- authenticated T0 parent to T2 child-coarse source residual: exactly `0`;
- canonical O6 prolongation residual: `1.77636e-15`;
- evolved-field telescoping residual: exactly `0`;
- fixed-lattice constraint ledger residual: `8.88178e-16`.

On the newly refined active cells, no communication or prolongation writer
changes the evolved fields after the canonical transfer; the only measurable
active-state change is the algebraic projection (`L2=8.73e-3`). Stored ghost
arrays do change substantially during MPI receive (`L2=43.84`) and
coarse-to-fine prolongation (`L2=21.47`), as expected for boundary closure.

On a fixed child lattice, however, the authoritative parent constraints
interpolated to that lattice have `C=0.7861`, while recomputation after the
representation/boundary phases gives `C=74.36`; algebraic projection raises it
to `74.86`. The worst fixed-lattice change is at
`(rho,z)=(5.1328125,-0.0078125)`, one fine cell from a MeshBlock edge, but
`5.13 M` from the symmetry axis and `0.133 M` from the nearest coarse-fine
interface.

**Evidence:** the target transfer bytes and writer ordering are internally
consistent, and the large jump appears when derivatives/constraints are
represented on the new fine lattice near a MeshBlock edge.  
**Inference:** this points toward an inadequately resolved or transfer-incompatible
block-edge representation, not a simple missing ghost-fill write. It does not
yet identify which discrete operator must change.

Detailed artifacts are under [`target_event/`](target_event/).

## Parent-state audit

Immediately before refinement, the selected parents are smooth in their
interiors but not uniformly within the high-order transfer regime at their
block edges:

- chi self-shadow `||u-PRu||/||u||`: `1.56e-4` overall,
  `2.36e-4` in the four-cell edge band, `1.86e-6` in the interior;
- K self-shadow: `0.357` overall, `0.394` edge, `0.00579` interior;
- Atilde self-shadow: `0.287` overall, `0.329` edge, `0.0169` interior;
- block-local Nyquist maxima are about 0.047 for chi, 0.085 for K, and 0.067
  for Atilde.

All maximum self-shadow residuals occur in parent GID 28 at
`(rho,z)=(5.109375,-0.015625)`, at an internal block corner near the Brill-wave
ring—not at the symmetry axis. O6-O4 derivative disagreement is likewise
non-negligible. These numbers support, but do not prove, the hypothesis that
repeated refinement/derefinement repeatedly samples a marginal block-edge
representation and amplifies its high-frequency content.

Detailed artifacts are under [`parent_state/`](parent_state/).

## Interpreting A, B, and C without overclaiming

1. **A versus B:** the topology sequence is causal to A's observed refinement
   runaway. B has the same initial event and persistent multilevel interfaces,
   but no subsequent topology changes and no runaway through `12.5 M`.
2. **C versus B:** merely adding a full fine MeshBlock buffer is not sufficient.
   C has more fine representation (128 versus 86 blocks), more coarse-fine
   exposure (108 versus 78 face incidents per RK stage), and fails late even on
   a frozen tree. Its terminal C maximum is about `0.977 M` from a coarse-fine
   interface but only `0.0234 M` from a MeshBlock edge.
3. **Combined inference:** repeated regridding is the leading trigger for A,
   while a larger fine representation can independently expose a late
   block-edge/bulk instability. The evidence does not separate parent
   under-resolution, non-monotone point restriction, one-sided block-edge
   closure, or persistent-interface feedback sufficiently to authorize a code
   correction.

## Dissipation-0.5 disposition

The requested matched `diss=0.5` rerun **did not execute**. Interactive job
`57102293` was revoked before allocation because its live `salloc` connection
timed out. Scheduler evidence records zero elapsed time, zero allocated nodes,
no node assignment, no run directory, no GPU use, and no science data. It must
not be used in any numerical comparison.

## Provenance and limitations

Job `57098562` completed its outer allocation successfully. Its terminal
evidence is authenticated by:

- root manifest: `c5819a40dec0ae278b63e3867d1ca1bb9661fee94f781dc1837ff49a7d8c28c6`;
- detached manifest: `4c82deaf1d467dd2b4462698615378e2bda90f3d055dc3cf08b7522423a3210c`;
- settled accounting: `b4e8c7838ecb6e4a13763389790484d2a938bb7e413a3090b576ba3f064a02fa`.

The strict compact handoff is indexed by `EVIDENCE_MANIFEST.json` and
`SHA256SUMS`. Raw production evidence remains on Perlmutter at
`/pscratch/sd/h/hzhu/axisymmetric-cartoon-r4-brill-amr-frozen-abc-21a268e4-v1-20260816`.

Limitations:

- A ended by the prospective 50-minute step limit, not a strict chi failure;
- C's first invalid chi writer was not stage-localized beyond the boundary
  parent-stencil gate;
- the target-event audit covers the first level-2-to-3 event, not C's terminal
  failure window;
- no limited-O2 B2 result exists because its prospective trigger was not met;
- no `diss=0.5` result exists;
- there is no convergence, Figure-3 reproduction, physical-collapse, or
  production-fix claim.

