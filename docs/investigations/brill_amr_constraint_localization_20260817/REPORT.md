# Cycle-1722 Brill AMR constraint-jump localization

Date: 2026-08-17

Repository: `HengruiZhu99/athenak`

Branch: `codex/brill-amr-coarse-cache-coherence-20260817`

Source diagnostic commit: `55f9147bc80d574636c47bcd1dac86178d921988`

Target: N256 Brill `A=-0.047`, cycle 1722, `t=9.50625 M`, 74 to 86 MeshBlocks,
old maximum level 2 to new maximum level 3.

## Verdict

**Primary disposition: `same_level_seam_derivative_dominates`.**

The dominant immediate jump is already present before algebraic projection and
is concentrated in production O6 derivative stencils that cross a newly formed
same-level seam between explicitly refined child blocks. This is not a simple
ghost-copy failure: every audited same-level receiver ghost is bitwise equal to
its sender-active value, and every local-block O6 derivative is bitwise equal to
the corresponding derivative stitched directly from neighboring active cells.
The field representation itself is derivative-incompatible across that seam.

This investigation does **not** isolate whether that incompatibility originates
from independently prolongating the two neighboring parent representations,
from an under-resolved parent feature near `rho ~= 5.1, z = 0`, or from another
block-local closure property. It provides no convergence or Figure-3
reproduction claim and does not qualify a production correction.

## Numerical and provenance scope

- Existing authenticated restart SHA-256:
  `83e996d2d5069307888a69fff47a7524c2f63f11869fb628630bca54dd5943ea`.
- Production setup: O6, RK4, CFL 0.15, KO 0.02, `dchi_max=0.01`,
  derefinement threshold `0.5*dchi_max`, `kappa1=kappa2=0`,
  `floor_chi=false`, high-order AMR transfer.
- Existing zero-PDE source evidence: job 57168348, raw-manifest SHA-256
  `bba4126d3d0f33f4d63369feb3bb8894abeb21255ea3a4a41ebea84d87b5d40c`.
- Fresh derivative-order state audit: job 57185308, using the checksummed fresh
  CUDA build made by predecessor job 57184215 from `55f9147b`, tree
  `cb2ad270f0675230b77023877dc0fdf93b52cd59`, Kokkos
  `6739bc623081648af9e752b616d9671527922cbf`; root
  `/pscratch/sd/h/hzhu/axisymmetric-cartoon-brill-constraint-localization-55f9147b-v4-20260817`.
- A v1 allocation request was rejected before job creation because the shared
  A100 queue requires 32 CPU cores per GPU and the harness requested 16. It
  used no node/GPU and produced no scientific data. V2 used 32 total cores but
  stopped after a successful build because the one-task allocation could not
  launch the two-rank ownership test; no zero-PDE science ran. V3 performed a
  fresh build and passed all five focused tests, then stopped before the AMR
  event because its inherited input lacked the newly added diagnostic key. V4
  reuses that exact checksummed V3 executable/build and adds only a diagnostic
  input declaration; no production numerical parameter differs.

Remote terminal evidence for job 57185308: state `COMPLETED`, exit `0:0`, one
numbered CUDA step, elapsed 38 s. The V4 run-manifest SHA-256 is
`fdec782c07cc37b34eef50eb0ba0613418b1d212ce6ab89eb50985e4675755f9`;
its detached checksum is
`a30c2aee1b00a4e271f153d9d55e787337b749edc9e00cdb55a07ccaa8fece01`.
The reused executable SHA-256 is
`638bdf0d60daba67f2c20cbfbe127a3e0f65991832fe632af0085b2639aa2e4d`.

The Cartoon history/diagnostic measure is already the correct axisymmetric
ring measure,

`dV = 2*pi*rho*sqrt(gamma)*drho*dz`.

There is no fictitious collapsed-y cell width in the integrals below; the jump
is not a normalization artifact.

## Observations

### 1. Phase decomposition

| phase | proper C | proper H^2 | proper M^2 | proper Z^2 |
|---|---:|---:|---:|---:|
| T0 accepted old state | 14.1509665303 | 0.5622987772 | 0.7757016837 | 2.1952754672 |
| T3_06 after all ghost/BC work, before projection | 87.7346940100 | 12.8143694748 | 62.1172137184 | 2.1964954044 |
| T4 after algebraic projection | 88.2282293576 | 12.8946620527 | 62.5305790489 | 2.1964647524 |
| T5 production recomputation | 88.2282293576 | 12.8946620527 | 62.5305790489 | 2.1964647524 |

T4 and T5 are byte-identical for every captured field. Algebraic projection
accounts for only 0.559% of final C, 0.623% of H^2, and 0.661% of M^2. The full
qualitative jump is present at T3_06, so projection is secondary.

Projection does alter conformal-metric/extrinsic-curvature values and therefore
the constraints; byte equality is claimed only for T4 to T5, not T3 to T4.

### 2. Active writer audit

For every accepted active cell and all 25 evolved Z4c components, T2 and T3_06
are byte-identical. This holds for unchanged leaves, explicitly refined
children, and balance-induced children. Boundary work after T2 did not rewrite
the active child state.

### 3. Exact O6 stencil provenance

Every active T3_06 cell was classified by the complete production O6 read
support, including pure first/second derivatives, mixed derivatives, Cartoon
suppressed-direction support, physical boundaries, and parity ghosts. The
classes are mutually exclusive and based on actual provenance rather than
distance alone.

Fractions of the **T0-to-T3_06 proper-integral jump** are:

| stencil class | C | H^2 | M^2 |
|---|---:|---:|---:|
| SAME_LEVEL_SEAM | 62.220% | 96.586% | 55.290% |
| ACTIVE_ONLY | 34.935% | 3.316% | 41.296% |
| COARSE_FINE | 0.518% | -0.268% | 0.691% |
| MIXED_CORNER | 2.325% | 0.355% | 2.722% |
| PHYSICAL_BOUNDARY | 0.002% | 0.011% | approximately 0 |

The Z^2 jump is small and sign-changing, so percentage attribution is
ill-conditioned and is retained in the CSV rather than used for classification.

The corresponding fractions of the total T3_06 proper integral are 55.707%,
93.025%, and 54.801% for C, H^2, and M^2 at same-level seams. Coarse-fine and
axis-only regions are small. The dominant seam is near `rho=5.07--5.15` and
`z=+-0.0078125`, not at the symmetry axis `rho=0`.

### 4. Parent-region comparison

Native old-parent constraints were integrated over each parent physical region
and compared to the union of native new-child constraints over the identical
region. No coarse constraint field was interpolated.

| old parent | new children | cause | proper C ratio | H^2 ratio | M^2 ratio | fraction of global proper-C jump |
|---:|---|---|---:|---:|---:|---:|
| 28 | 28--31 | explicit refinement | 94.602 | 33.734 | 193.116 | 50.124% |
| 45 | 51--54 | explicit refinement | 93.672 | 33.156 | 192.740 | 49.853% |
| 29 | 32--35 | 2:1 balance | 0.990 | 1.068 | 1.068 | negligible |
| 48 | 57--60 | 2:1 balance | 0.985 | 1.067 | 1.083 | negligible |

The two explicitly refined parents immediately below and above `z=0` explain
essentially the whole C jump. Balance refinement is not the driver.

### 5. Conditional Case-B seam audit

For the union of the top eight same-level cells by proper C, H^2, and M^2
contribution (12 unique cells):

- 7,644 exact ADM stencil values were recorded;
- 2,730 of those reads were same-level ghost reads;
- sender-active and receiver-ghost bytes had zero bit mismatches and zero
  maximum absolute mismatch;
- local-block O6 `D1`, `D2`, and mixed derivatives agreed exactly with
  derivatives stitched from neighboring active values; maximum residual 0.

The large derivative changes across the seam are therefore real properties of
the accepted child samples, not communication corruption. The top cells lie in
GIDs 30 and 51 on opposite sides of `z=0`; the seam is between child blocks
created from different explicitly refined parents.

### 6. Production C++ derivative-order shadow audit

The fresh O6 bytes reproduce the established T3_06 production constraints
exactly. Proper-ring integrals on the complete state are:

| metric | O2 | O4 | O6 | O2/O6 | O4/O6 |
|---|---:|---:|---:|---:|---:|
| C | 46.1556 | 73.2810 | 87.7347 | 0.5261 | 0.8353 |
| H^2 | 6.1420 | 10.5877 | 12.8144 | 0.4793 | 0.8262 |
| M^2 | 27.2393 | 49.8918 | 62.1172 | 0.4385 | 0.8032 |

On SAME_LEVEL_SEAM cells alone:

| metric | O2 | O4 | O6 | O2/O6 | O4/O6 |
|---|---:|---:|---:|---:|---:|
| C | 23.1458 | 39.8386 | 48.8744 | 0.4736 | 0.8151 |
| H^2 | 5.6815 | 9.8574 | 11.9206 | 0.4766 | 0.8269 |
| M^2 | 14.5600 | 27.0691 | 34.0407 | 0.4277 | 0.7952 |

ACTIVE_ONLY cells are also order-sensitive (O2/O6 ratios 0.568 for C, 0.494
for H^2, and 0.439 for M^2), so derivative-order sensitivity is not confined
to ghosts. Nevertheless, the largest cell for every metric is the same at all
three orders, and pointwise magnitude correlations are high: O2/O6 Pearson
correlations are 0.9940, 0.9864, and 0.9935 for C, H^2, and M^2; O4/O6 exceeds
0.9992. All orders therefore identify the same localized structure. Lower
order attenuates it but does not spatially relocate it.

O2/O4/O6 values here are all produced by `Z4c::EvaluateDiagnosticConstraints`
on the same T3_06 state and the same exact masks. They are sensitivity evidence,
not evidence that a larger O6 norm is an O6 implementation bug.

## Deductions

1. Algebraic projection cannot be the dominant source because more than 99.3%
   of final C/H^2/M^2 is already present before it.
2. A wrong or stale same-level ghost copy cannot explain the audited seam:
   receiver ghosts exactly match the owner active bytes, and the stitched and
   local derivatives are identical.
3. Persistent coarse-fine reconstruction and axis parity are not dominant in
   this event by their measured contribution fractions.
4. The constraint amplification is tied to the explicitly refined physical
   region and especially to the new same-level seam separating the child
   representations above and below `z=0`.
5. The direct source of the production constraint spike is derivative
   incompatibility across that seam. This selects Case B under the prescribed
   decision tree.

## Hypotheses requiring further evidence

- The two neighboring P5 child representations may reproduce values while not
  enforcing enough derivative compatibility across their shared block seam.
- The parent state near `rho ~= 5.1, z=0` may already contain unresolved/high-k
  content that any independent child construction exposes.
- A block-edge closure or independently chosen interpolation orientation may
  amplify the mismatch even when all communicated bytes are correct.

These are not distinguished by the present event. In particular, the prior
P5_BLOCK/P5_STITCHED agreement on their common valid lattice is compatible with
all three: it checks value construction, not derivative compatibility across
every newly formed seam.

## Unsupported possibilities

- A simple stale/wrong same-level ghost-zone copy at this event.
- Dominant symmetry-axis parity or coarse-fine ghost injection.
- Algebraic projection as the main jump source.
- A fictitious collapsed-y normalization factor.
- P5 insufficiency specifically, parent under-resolution specifically, or an
  O6 derivative bug.
- Convergence, Figure-3 reproduction, or physical critical behavior.

## Natural next step

Do not change interpolation order yet. The smallest decisive follow-up is a
bounded zero-PDE parent-to-child seam audit for old parents 28 and 45:

1. dump the exact P5 source stencil and interpolation orientation for paired
   child samples on both sides of `z=0`;
2. evaluate value and derivative compatibility at their common child seam;
3. evaluate a diagnostic construction that derives both sides from one
   stitched parent neighborhood, while leaving the production path unchanged;
4. compare only the existing ACTIVE_ONLY and SAME_LEVEL_SEAM masks.

If a stitched-parent construction removes the derivative mismatch, the
block-local parent closure is isolated. If it does not, the parent state is the
leading unresolved source and only then is an earlier-refinement experiment
justified. No long evolution, floor, clipping, broad parameter sweep, or gate
relaxation follows from this report.

## Artifacts

- `existing_byte_analysis/phase_comparison.csv`
- `existing_byte_analysis/phase_field_changes.csv`
- `existing_byte_analysis/active_byte_t2_to_t3.csv`
- `existing_byte_analysis/stencil_provenance_cells.csv`
- `existing_byte_analysis/constraint_contributions_by_origin_and_stencil.csv`
- `existing_byte_analysis/constraint_jump_by_origin_and_stencil.csv`
- `existing_byte_analysis/parent_region_native_constraint_comparison.csv`
- `existing_byte_analysis/t3_constraint_maps.png`
- `existing_byte_analysis/stencil_and_hierarchy_masks.png`
- `existing_byte_analysis/constraint_jump_contribution_map.png`
- `same_level_seam_analysis/same_level_exact_adm_stencils.csv`
- `same_level_seam_analysis/same_level_derivative_comparison.csv`
- `derivative_order_analysis/` (production C++ O2/O4/O6 tables)
- `remote_derivative_audit/` (build/test/run/accounting and raw-byte hashes)
- `verdict.json`, `evidence_manifest.json`, `SHA256SUMS`

Raw binary arrays are intentionally not committed. Their exact selected-file
hashes, sizes, source manifest, and verification result are embedded in
`existing_byte_summary.json` and the strict evidence manifest.

## Qualification boundary

This is a bounded diagnostic result only. The prior owner-authoritative
`coarse_u0` repair remains valid and separately regression-tested, but it is
quantitatively secondary for this event. The existing 3D unit test verifies
prolongation on a populated coarse cube; it does **not** constitute a full
production mixed-level 3D send/receive ownership qualification.

**Final disposition: `same_level_seam_derivative_dominates`.**
