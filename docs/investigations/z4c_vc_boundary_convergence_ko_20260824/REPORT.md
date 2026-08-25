# Brill native-VC boundary convergence and KO scan

Date: 2026-08-24  
Base: `953f2724c00a2efd2f9fad91ae9a784639954a3b`  
Branch: `codex/z4c-vc-boundary-convergence-ko-scan-20260824`

## Executive verdicts

**Boundary:** `BOUNDARY_CONTAMINATION_CONFIRMED`

The apparent N256 to N512 flattening in the existing global Figure-2
constraint inventories is primarily an outer-boundary/shell effect. At
`axisTau=3.9861474M`, the Rout=16 full-domain direct fine-pair orders are
0.861, 1.482, 2.901, and 0.334 for C/H/M/Z. The corresponding Rout=128 orders
are 5.146, 6.821, 4.281, and 6.186. Inside `r<=4M`, both domains agree to
about one part in a billion and all four fine-pair orders are 7.80 or higher.
The central `axisKret` traces differ by at most `6.7e-9` dex.

**KO:** `KO_STRONG_EFFECT`

On the original Rout=16 N256 problem, increasing KO strongly delays the first
late refinement and suppresses constraint growth. The diss=0.02 baseline
stalls at `t=11.191917M` with `dt=7.98e-9M`, `C=1.61e16`, level 20, and
1,526 leaves. Diss=0.05, 0.10, 0.20, and 0.50 all reach `t=11.3M`. Their
first late refinements occur at 10.3658, 10.7394, 11.1045, and not by 11.3M,
respectively. Diss=0.50 ends with `C=0.2695` and no late accepted refinement,
while its maximum matched central-trace deviation from diss=0.02 is 0.110 dex.

## Part I: boundary/convergence experiment

The Cartoon history measure was audited first. It already uses
`2*pi*rho*sqrt(det(gamma))*drho*dz` with canonical native-VC trapezoid
ownership. There is no fictitious collapsed-y width.

The authenticated Rout=16 binaries were then re-integrated over
`r<=4,8,12,14M`, the full square, and radial shells. The small-domain N512 Z
inventory is 99.87% outside `r=12M`; 50.53% is in `12<r<=16M` and 49.34%
is in square corners with `r>16M`.

A new Rout=128 SMR domain preserves the old inner spacing exactly with three
dyadic minimum levels and raises only the absolute level ceilings by three.
The N256 inner initialization has identical coordinates and 23/25 variables
bitwise identical; Gamma_x/y differ by at most `1.94e-15`.

The N256 Rout=128 authority completes at t=6.5 with 206 final leaves. A
same-resolution replay has a bitwise-identical 264 by 71 history. N128 and
N512 replay the exact tree at physical event times and finish cleanly. The
large boundary is conservatively causally unable to reach `r<=12M`.

![Old global versus large-domain orders](analysis/boundary_comparison/figures/boundary_fine_pair_orders.png)

![Z localization](analysis/boundary_comparison/figures/z_boundary_contamination.png)

Detailed tables: [small-domain radial convergence](SMALL_DOMAIN_RADIAL_CONVERGENCE.md),
[large-domain convergence](LARGE_DOMAIN_CONVERGENCE.md), and
[direct boundary comparison](BOUNDARY_COMPARISON.md).

## Part II: independent N256 KO experiment

Only `z4c/diss` changes across 0.02, 0.05, 0.10, 0.20, and 0.50. Every case
uses a fresh native-AMR record authority; no baseline tree is replayed.

| diss | C(6.5) | C(~9.2) | first late refinement | max leaves | terminal t | status |
|---:|---:|---:|---:|---:|---:|---|
| 0.02 | 3.285e-3 | 1.032 | 10.2758 | 1526 | 11.191917 | bounded termination after timestep collapse |
| 0.05 | 2.693e-3 | 0.692 | 10.3658 | 65 | 11.3 | reached tlim |
| 0.10 | 2.224e-3 | 0.390 | 10.7394 | 86 | 11.3 | reached tlim |
| 0.20 | 1.850e-3 | 0.168 | 11.1045 | 50 | 11.3 | reached tlim |
| 0.50 | 1.518e-3 | 0.0437 | none | 50 | 11.3 | reached tlim |

![KO constraint scan](analysis/ko_stageC/figures/ko_global_constraints.png)

![KO physical trace](analysis/ko_stageC/figures/ko_axisKret_figure3_overlay.png)

The best conservative next campaign is Rout=128 SMR with the existing
common-tree record/replay workflow, zero Z4 damping, and diss=0.20, while
retaining diss=0.50 as a stability control. Diss=0.20 is chosen as a compromise,
not a qualified production default. A three-resolution comparison must show
that its remaining late growth converges away and that the small physical
trace change shrinks with resolution.

## Evidence boundaries and limitations

- This confirms the cause of the current Figure-2 global flattening through
  t=6.5. It does not qualify late-time Figure-3 convergence.
- The optional late Rout=128 N256 control was not run; the boundary and KO
  verdicts remain independent.
- The KO scan is one-resolution evidence. It identifies a strong numerical
  damping sensitivity but cannot by itself distinguish continuum, spatial,
  and AMR-interface error or set a production dissipation.
- No horizon/critical-scaling claim is made.
- No production numerical source was modified.
- One diss=0.20 HBM trace has an isolated post-step 36.6 GiB sample; raw data
  are retained and it is not interpreted as a physical memory dependence.
- Multi-gigabyte binaries/restarts remain at the exact Perlmutter paths in
  `EVIDENCE_MANIFEST.json`; the repository includes the derived integrals,
  histories, authorities, logs, plots, and hashes needed for read-only review.

## Artifact map

- [Static audit](STATIC_AUDIT.md)
- [Boundary causality](BOUNDARY_CAUSALITY.md)
- [SMR layout proof](SMR_LAYOUT_PROOF.md)
- [Large-domain authority](LARGE_DOMAIN_AUTHORITY.md)
- [KO scan](KO_SCAN_N256.md)
- [Strict evidence manifest](EVIDENCE_MANIFEST.json)
- `REPORT.tex` and `REPORT.pdf`

