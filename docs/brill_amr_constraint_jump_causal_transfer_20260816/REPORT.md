# Brill N256 AMR constraint-jump causal transfer comparison

## Verdict

The prospective causal gate **failed**. Replacing only the exact cycle-1722
level-2-to-3 transaction with global `limited_o2` transfer did not reduce the
constraint jump; it made all three constraint families much worse. The
subsequent same-state derivative audit found substantial O2/O4/O6 sensitivity,
but no isolated defective derivative or transfer seam. The terminal disposition
is therefore:

`inconclusive_parent_resolution_or_derivative_sensitivity`

No `edge_limited_o2` implementation, three-method evolution, resolution
campaign, convergence claim, or Figure-3 reproduction was authorized or run.
`qualification_claim=false`.

## Fixed event and numerical identity

- Brill amplitude: `A=-0.047`; imported ADM mass:
  `2.660301967997158 M`.
- Resolution label: N256; `dchi_max=0.01`; derefinement threshold
  `0.5*dchi_max`; refinement interval one cycle.
- O6 bulk derivatives, RK4, CFL `0.15`, KO dissipation `0.02`, and Z4c
  constraint damping `(kappa1,kappa2)=(0,0)`.
- Restart SHA-256:
  `83e996d2d5069307888a69fff47a7524c2f63f11869fb628630bca54dd5943ea`,
  cycle 1721, `t=9.501562499999995 M`.
- Target: cycle 1722, `t=9.50625 M`, maximum level 2 to 3, 74 to 86
  MeshBlocks, 12 created and none deleted.
- Both causal arms used one executable, one restart, one input, one A100-SXM4
  80GB GPU, one MPI rank, and no post-event PDE cycle. Both retained production
  `high_order` through the preceding RK cycle, changed policy only for T1--T5,
  restored `high_order` after T5, and stopped before the next RHS.

The authoritative history/cell measure is already the Cartoon axisymmetric ring
measure `2*pi*rho*sqrt(gamma)*d(rho)*dz`; coordinate-ring
`2*pi*rho*d(rho)*dz` is retained as a cross-check. The result is not a
fictitious collapsed-y normalization effect.

## Phase 1: matched zero-PDE captures

Job `57137565` ran both arms successfully on `nid008205`. Slurm steps `.0` and
`.1` completed with exit `0:0`. The allocation parent ended `FAILED 1:0` only
because the first offline comparison divided by the zero volume of a valid empty
disjoint region. The immutable raw captures were preserved and sealed. Commit
`a96c49e454674df8d1567a6d489ac08ce40a6f01` corrected only that analyzer case:
an empty region now has `rms=null`, while inconsistent empty accounting fails
closed. No probe was rerun.

The reconstructed T0 evolved/ADM/constraint representations are byte-identical
between arms. Both transactions have the same accepted child lattice and close
their T0--T5 ledgers and regional budgets.

## Phase 2: prospective causal gate

The corrected v3 comparison gave these proper-ring global post/pre jump ratios:

| Family | `high_order` | `limited_o2` | high/limited factor | Gate result |
|---|---:|---:|---:|---|
| C | 95.2237 | 1304.7564 | 0.07298 | worse |
| H | 42.6188 | 1270.2369 | 0.03355 | worse |
| M | 195.8316 | 2040.8443 | 0.09596 | worse |

No family improved by the required factor of two, and every family violated the
no-more-than-25-percent-worse rule. Proper-ring and coordinate-ring reductions
agree on localization. The largest high-order excess is in the disjoint
`MESHBLOCK_EDGE_OR_CORNER` category, but the limited arm increases rather than
reduces the excess in that same category. Spatial edge correlation therefore
does not authorize an edge-only source correction.

## Gate-fail derivative-order audit

Commit `392522dda2737508697662982f688809a154d571` adds a default-off diagnostic
that evaluates O2, O4, and O6 constraints from one accepted T5 evolved state.
It does not modify the evolved variables or production derivatives. Serial and
MPI builds, the focused unit test, the complete runtime/rank/mutation suite, and
the CUDA build/focused tests passed.

Job `57140240` completed `0:0` on `nid008584`. The exact independently
recomputed O6 constraint bytes equal the production T5 bytes. Proper-ring global
integrals are:

| Family | O2 | O4 | O6 | O2/O6 | O4/O6 |
|---|---:|---:|---:|---:|---:|
| C | 32.8146 | 60.3145 | 74.8573 | 0.43836 | 0.80573 |
| H | 6.03992 | 10.4888 | 12.7106 | 0.47519 | 0.82520 |
| M | 26.6931 | 49.7402 | 62.0602 | 0.43012 | 0.80148 |

The spatial fractions are stable across derivative order rather than pointing
to a unique bad stencil. MeshBlock-edge/corner cells carry approximately
73 percent of C, 95 percent of H, and 68--69 percent of M for all O2/O4/O6;
coarse-fine cells contribute about 7, 3, and 8 percent respectively. The
largest O2--O6 disagreements occur near `(rho,z)=(5.07,-0.0078) M` for C/M and
`(5.12,-0.0234) M` for H. The target refined-child set contains no cells in the
axis/physical-boundary category, so this event cannot diagnose an axis-ghost
defect.

Block-local fluctuation spectra change mainly in amplitude: the fraction of
power above integer mode-radius 12 is similar across orders (about 9--10 percent
for C, 43--44 percent for H, and 10--11 percent for M). This is descriptive,
not a global Fourier analysis.

## Evidence versus interpretation

**Observed:** global limited transfer is far worse at the exact frozen event;
O2/O4/O6 constraint integrals differ monotonically on one evolved state;
localization fractions and spectral shapes are broadly order-stable; O6 audit
bytes reproduce production exactly.

**Inference:** the constraint jump is derivative-sensitive, and a blanket
low-order transfer is not a stabilizing causal correction. Edge localization
alone is insufficient evidence for an edge-only operator change.

**Hypothesis:** the accepted parent state may already lack the resolved content
needed by the newly created child lattice, or the event may expose broadly
distributed high-frequency representation error. This is not established as a
source bug, continuum instability, or convergence failure.

**Open question:** a separately authorized, bounded parent-resolution/no-PDE
diagnostic could determine whether the order spread decreases when the parent
state is refined earlier. This report does not authorize that run.

## Skipped phases and limitations

- Phase 3 `edge_limited_o2`: skipped because the causal gate failed.
- Phase 4 H/L/E evolved comparison: skipped because no candidate was authorized.
- No convergence order, long-time stability, collapse/dispersal, horizon, or
  Figure-3 claim follows from a single zero-PDE transaction.
- The regional precedence is diagnostic; it does not uniquely identify the most
  recent writer of each derivative error.
- The derivative spectra are per-MeshBlock fluctuation spectra on the selected
  level-3 children.
- This fail-closed result does not justify floors, clipping, weakened chi gates,
  transfer threshold tuning, or a production-default change.

## Evidence identities

- Source branch: `codex/brill-amr-frozen-hierarchy-20260816`.
- Capture source: `e6b1428cbe1fafe941ac6a41cbabe14430ed8d14`.
- Empty-region analyzer fix: `a96c49e454674df8d1567a6d489ac08ce40a6f01`.
- Derivative-audit source: `392522dda2737508697662982f688809a154d571`,
  tree `054d370479af2fc73d775b5e3ef8325bf288f90d`.
- V2 root manifest: `11dd1454943adbfc5a681fda8876e07a9f07346d43ab89cb6cf7d32d92096342`;
  detached: `35c686ae9429d7da53b0298b07c53713c18cdfe77ec600e31d239223f517780d`;
  sacct: `52a2ce4a0974e526ad615ff0e0befaf48a38fe1d6e661e7246e939a550ad62dc`.
- V3 gate verdict: `6f3b1d2d9b078594d3a56761892c84945a0a9562c148fdad47ef06cb3e841aff`;
  output manifest: `72ba4cb4b39ee9f191684844a88fe7151d032e4da064f3f058c0d3c8a15f1e52`.
- V4 executable: `0f91650c79da0f68fcfc6409ac073ca324f4ee70eea79a0625dd51b917203184`;
  diagnostic manifest: `d5857c3046a7144e206c06acdd6d7316bd72dccf4ce36d9344d796aabd441a8f`.
- V4 root manifest: `4e24d433faf92c9b052cd44cf48b7b4178c6609c92c06f4993f61a66e18d4539`;
  detached: `ccf6111a1777d9206bcb65e5fbe48b34ea36eaa3d7f5f6b38bd9b24f767c813f`;
  sacct: `7019f6b762c51f7a346919c401c08f238065f16d691c236b352697d826bf96a6`.
- V4 audit verdict: `73c7651c64fc0f54fe2d8c5b919f716be3569bbbc65a983691877027505514d0`;
  output manifest: `7e47d8113ad28489fa7ceb8ef8b9b0ea025c99ae92a16f818aba057a1cc9b756`.

All local submanifests and the final root/detached checksum layers verify.
