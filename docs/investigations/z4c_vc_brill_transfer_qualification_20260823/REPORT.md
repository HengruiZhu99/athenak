# Vertex-centered Z4c Brill transfer qualification

Date: 2026-08-23

Branch: `codex/z4c-vc-brill-transfer-qualification-20260823`

Exact base: `2d59f85c11cb0da4614c84a695d64f032fb9eec7`

Qualified source under test: `278b63a740a947de55ad8bdd1c333095c68fedcd`

## Executive verdict

The new vertex-centered (VC) transfer selector, CUDA/MPI paths, and bit-exact restart behavior are qualified by focused synthetic tests. The matched O4 midpoint prolongator is **suborder** for an O4 Z4c operator: its interface state is fourth-order, but the second-derivative RHS defect is only approximately second-order. The elevated O6 midpoint prolongator restores approximately fourth-order semidiscrete and dynamic-AMR behavior in those synthetic tests.

The physical Brill gate nevertheless fails **before AMR is introduced**. Fixed-grid VC/O4 Brill runs at N128, N256, and N512 all reached central proper time `tau_c >= 3 M`, but their global proper-volume constraint norms worsen with resolution at late time. Region-resolved terminal diagnostics show mostly second-to-fourth-order convergence in the central `r <= 8 M` region, while the symmetry axis and outer boundary are nonconvergent. Therefore:

```text
overall = VC_PHYSICAL_GATE_FAILED
```

No production O4 transfer rule is selected. Common-tree Brill, native VC AMR, strong-field outcome, and performance phases were not run after this gate failure.

## Final classifications

| Gate | Classification | Basis |
|---|---|---|
| `vc_o4_prolongation` | `SUBORDER` | O4 state transfer, approximately O2 interface RHS; dynamic errors reach only 2.16–2.73 order |
| `vc_o6_prolongation` | `NOT_ESTABLISHED` | Synthetic tests restore O4 behavior, but nonlinear Brill/common-tree robustness was blocked by the fixed-grid gate |
| `selected_o4_transfer` | `NOT_ESTABLISHED` | Physical prerequisite failed before the decisive common-tree comparison |
| `vc_host` | `QUALIFIED` | Phase-0 portable suite and host/MPI controls passed |
| `vc_sycl` | `PENDING` | No current-source Aurora/SYCL runtime evidence was produced |
| `vc_cuda` | `QUALIFIED` | Current-source one-A100 CUDA and two-rank MPI tests passed |
| `vc_restart` | `BIT_EXACT_QUALIFIED` | Explicit q4/q6 refined and post-derefinement restart tests passed |
| `vc_brill_fixed_grid` | `NONCONVERGENT` | Late global constraints and axis/outer-boundary regions worsen with resolution |
| `vc_brill_common_tree` | `NOT_ESTABLISHED` | Not run after fixed-grid gate failure |
| `vc_brill_native_amr` | `NOT_ESTABLISHED` | Not run after fixed-grid gate failure |
| `vc_performance` | `NOT_MEASURED` | Correctness prerequisite failed |
| `overall` | `VC_PHYSICAL_GATE_FAILED` | Fixed-grid physical convergence gate failed |

## What changed

Five source/test commits were made on the qualification branch:

1. `11b9639336d7519121a4ddc4b5a7e1b4106897ec` — VC-only `vertex_prolongation_order=auto|4|6|8`, fail-closed compatibility, provenance, and transfer-dependent halo selection.
2. `23a475ed4c66aec5e9d6e1f9bf0508797b69b227` — semidiscrete VC interface diagnostic.
3. `06fe9cc94e712ba662651b18798bbbbf60478d19` — default-off state/RHS/constraint sampler for all 25 evolved Z4c variables and seven constraint fields.
4. `d8cc30f1d13a476789b726f26f6659026b1f619a` — explicit q4/q6 MPI AMR and restart coverage.
5. `278b63a740a947de55ad8bdd1c333095c68fedcd` — corrected the rank-change event oracle (`0 created, 3 deleted` after loading the refined checkpoint).

The selector preserves exact coincident-node injection. `auto` retains the pre-existing elevated-order VC behavior: spatial orders p2/p4/p6 map to q4/q6/q8. Cell-centered behavior remains on the legacy path and does not materialize the VC selector in legacy restart/input bytes.

## Evidence by phase

### Phase 0: repaired authority reproduced

The portable suite passed 123/123 tests. Cell-centered implicit/default and explicit-cell payloads matched, and repaired O2/O4/O6 2D/3D dynamic-AMR convergence controls passed. One literal binary fingerprint remains compiler-bound and is explicitly separated from the numerical equivalence result.

Phase-0 manifest hash:

```text
573a807c64f57a51417ae181aa23eae351c1cc4612b59fc59cd0918f34700481
```

### Phase 1: isolated transfer response

| Quantity | q4 | q6 |
|---|---:|---:|
| 1D coefficient L1 norm | 1.250000 | 1.390625 |
| 2D tensor L1 norm | 1.562500 | 1.933838 |
| 3D tensor L1 norm | 1.953125 | 2.689243 |
| Maximum image amplitude | 0.5 | 0.5 |
| Localized-pulse overshoot | 0.004974 | 0.005084 |

Both rules preserve constants, symmetry, and exact injection. q6 has the wider and more alternating stencil, but it produces materially less image power for smooth and representative mixed modes. At the exact coarse Nyquist limit both reach image amplitude 0.5. This is transfer characterization, not a nonlinear stability result.

### Phase 2: semidiscrete interface

The refined hierarchy was compared with a level-matched uniform reference at N16/N32/N64/N128. The q4 state defect is O4, but the worst second-derivative RHS family converges near O2. q6 gives O6 state transfer and approximately O4 RHS defects.

Representative worst-interface values at N128:

| Rule | Variable | Interface RHS RMS | Asymptotic order |
|---|---|---:|---:|
| q4 | `Gamy` | `4.987e-05` | `1.98` |
| q6 | `Gamy` | `3.888e-07` | `3.96` |

This directly confirms the expected two-order loss when interpolation error enters a second derivative.

### Phase 3: dynamic nonconstant AMR

| Test | q4 observed orders | q6 observed orders |
|---|---|---|
| 2D | 2.158, 2.726 | 4.117, 4.560 |
| 3D | 2.204, 2.714 | 4.166, 4.564 |

Thus q4 is classified `SUBORDER`; q6 is the only candidate that preserves O4-compatible convergence in the synthetic AMR tests.

### Phase 4: CUDA/MPI/restart

At source commit `278b63a7`, six one-GPU CUDA tests and four two-rank q4/q6 migration/rank-change tests passed on one Perlmutter A100. The q4/q6 2D/3D convergence tests also passed. Refined and post-derefinement restart accepted states were bit-exact.

The first MPI submission had only one Slurm task and could not launch the nested two-rank step. The next run exposed a wrong test oracle, not a numerical fault; the corrected oracle passed in the clean retry. Those failed attempts remain preserved on Perlmutter and are not silently treated as science evidence.

Current CUDA/MPI manifest hash:

```text
90982dcee8eb69432aea841a5a787a504b09d40d95bf3be7e6417f913409d15a
```

### Phase 5: direct VC Brill initial data

Authenticated IrisK initial data were evaluated directly at VC nodes for `A=-0.047`, `rho=[0,16]`, `z=[-16,16]`, and the same physical 4x8 MeshBlock lattice.

Observed facts:

- all 22 fields assigned directly by the importer agree exactly at common physical nodes across N128/N256/N512;
- `Gamma^rho` and `Gamma^z`, reconstructed by the O4 discrete derivative, converge at approximately 3.95 order;
- shared-node spread is exactly zero at all three resolutions;
- `min(chi)=0.33172537559295967` and `min(alpha)=0.3125975980178073`;
- `max|det(gtilde)-1|=4.44e-16`, `max|tr(Atilde)|=0`, and the minimum SPD pivot is `0.44427985708000284`;
- axis algebraic regularity residuals are at roundoff.

The raw proper-box Hamiltonian RMS is approximately `3.35e-4` and approaches a spectral-data/discrete-operator floor rather than cleanly decaying across all three resolutions. Common-node H differences give only 1.63 aggregate order because the origin and same-level `z=0` seam dominate; an independent interior subset approaches fourth order. This is a limitation of the initial-data qualification and not evidence of AMR transfer.

Authenticated authority hashes:

```text
IrisK header       23bc2187c29ccb2695a54fc5c59e08a2e7b9d3389a63c1081cf953a507fb0cdb
IrisK static lib   d4afad6d3a20a8dd8197eb7d70d5a23903a7e2401a5d8b034d32005bf07f3f39
coefficients       ff0993c390513c15d6aa65857a0a3c710f2e2c3faf5717d9d63245203ccf2d6b
ADM mass           2.660301967997158
spectral residual  6.9517373601955137e-13
```

### Phase 6: fixed-grid Brill gate

The physical configuration was O4/RK4, CFL 0.15, KO `diss=0.02`, max-domain-`|K|` telegraph lapse with `tau=kappa=1`, Gamma driver with `eta=2`, no Z4 damping, no chi floor, and no AMR. The initial lapse was `psi^-2`.

The first submission inherited `nlim=0` and stopped at cycle zero. It produced no evolution evidence and is preserved as a harness failure. The corrected retry reached coordinate time `t=5 M`:

| Resolution | Cycles | terminal `tau_c/M` |
|---|---:|---:|
| N128 | 385 | 3.0798629613 |
| N256 | 771 | 3.0798678484 |
| N512 | 1541 | 3.0798699123 |

The global proper ring RMS constraints are not in a common asymptotic regime at late time:

| `tau_c/M` | Family | N128 | N256 | N512 | observed order |
|---:|---|---:|---:|---:|---:|
| 0.5 | C | 7.000e-3 | 2.775e-3 | 2.701e-3 | 5.85 |
| 1.0 | C | 6.200e-3 | 4.892e-3 | 1.121e-2 | -2.27 |
| 2.0 | C | 2.831e-2 | 8.288e-2 | 2.444e-1 | -1.57 |
| 3.0 | C | 1.837e-1 | 5.701e-1 | 1.491 | -1.25 |
| 3.0 | H | 6.450e-3 | 9.289e-3 | 1.985e-2 | -1.90 |
| 3.0 | M | 2.076e-3 | 4.021e-3 | 9.560e-3 | -1.51 |
| 3.0 | Z | 3.693e-2 | 1.192e-1 | 3.140e-1 | -1.24 |

The terminal one-stage restart diagnostic proves that duplicate shared vertices remain bitwise identical for all state, RHS, and constraint fields. Region-resolved N128/N256/N512 orders at `tau_c=3.08 M` are:

| Quantity | core `r<=8` proper ring RMS | core MeshBlock interior | axis unweighted RMS | outer boundary proper ring RMS |
|---|---:|---:|---:|---:|
| chi | 4.59 | 4.55 | -0.55 | -0.50 |
| alpha | 2.89 | 2.83 | 1.50 | -2.48 |
| Theta | 2.72 | 3.94 | -0.72 | -0.34 |
| `B^rho` | 2.06 | 2.05 | n/a | 2.50 |
| H | 3.35 | 3.40 | -0.29 | 0.31 |
| M | 2.99 | 4.94 | -3.14 | -0.97 |
| C | 4.45 | 5.14 | -1.32 | -1.04 |
| Z | 5.59 | 7.27 | -1.11 | -1.06 |

The corresponding RHS core orders are lower for some gauge/constraint variables: the worst core values include `Theta~1.87`, `Axx~2.07`, and `alpha~2.16`. Excluding MeshBlock seams improves several constraint families but does not repair the axis.

These observations establish a physical fixed-grid failure. They do **not** isolate whether the continuum/gauge formulation, the Cartoon axis closure, the outer Sommerfeld boundary, or some interaction among them is the first source.

## Cartoon history normalization audit

The constraint history jump is not a fictitious collapsed-y normalization artifact. In Cartoon mode the VC history path uses

```text
2*pi*rho*dx1*dx2*w_rho*w_z*sqrt(det(gamma))
```

and does not include `dx3`. Nodal trapezoid weights make shared endpoints tile the dual volume. The global history columns are therefore proper axisymmetric ring integrals/RMS values. The axis itself has zero ring volume at `rho=0`; the separate axis diagnostic uses an unweighted canonical-node census, which is why the report labels its order separately.

Relevant source:

- `src/z4c/z4c_history_quadrature.hpp`
- `src/outputs/history.cpp`

## Observation, inference, and hypothesis

### Established observations

- q4 prolongation is suborder for the O4 semidiscrete Z4c operator.
- q6 restores O4-compatible synthetic interface and dynamic-AMR convergence.
- all current CUDA/MPI/restart selector tests pass.
- the direct VC importer and shared-node reconciliation are exact for directly assigned fields.
- the fixed-grid Brill sequence becomes nonconvergent before any AMR transfer is present.
- axis and outer-boundary subsets are nonconvergent; core interior subsets are substantially better.
- shared duplicate state/RHS/constraint values are bitwise identical at terminal time.

### Supported inference

AMR transfer is not the sole source of the previously observed refinement jumps or loss of convergence. Selecting q6 may remove one demonstrated interface-order defect, but it cannot qualify the physical Brill evolution while the fixed-grid axis/boundary gate fails.

### Open hypotheses

The strongest bounded hypotheses are an axis regularization/RHS closure defect, a nonconvergent outer-boundary closure feeding inward, or a gauge/constraint mode that is only partially resolved in the core. The t=0 origin-dominated Hamiltonian floor may share an axis-origin mechanism, but the present evidence does not prove that connection.

## Why later phases stopped

The governing plan requires fixed-grid Brill convergence before common-tree transfer selection. Running Phase 7 after this result would conflate a bulk/axis/boundary defect with AMR transfer and could incorrectly promote q6. Therefore no common-tree Brill, native-AMR, black-hole, horizon, or performance run was started.

## Smallest natural next diagnostic

Use the existing fixed-grid terminal restarts and diagnostic sampler to localize the **first physical-time loss of convergence**, with no AMR:

1. compute region-resolved state/RHS/constraint N128/N256/N512 orders at several saved times before `tau_c=1`, including the first time global C changes from decreasing to increasing with resolution;
2. split the axis into radial/axial stencil layers and classify the first nonconvergent component and RHS term;
3. repeat only a short bounded fixed-grid segment with the outer boundary moved while holding core spacing and MeshBlock geometry fixed, after the writer/component is identified;
4. only after the fixed-grid axis/boundary issue is corrected, rerun Phase 6 and then compare q4 versus q6 on the authenticated common tree.

This is narrower and more decisive than another AMR parameter sweep.

## Limitations

- No current-source SYCL runtime qualification was obtained.
- Phase-5 raw Hamiltonian residuals do not show clean global O4 convergence.
- The fixed-grid runs reach the requested trusted-window proper time but do not identify the first defective RHS term.
- The terminal regional analysis is a three-resolution self-difference study, not comparison with an analytic evolved solution.
- No common-tree or native-AMR Brill result, production transfer selection, convergence claim, horizon claim, Figure-3 reproduction, critical exponent, or DSS result is established.
- No performance measurement is scientifically meaningful before the physical gate is repaired.

## Key artifacts

- [Transfer response](artifacts/phase1_transfer/fourier_transfer_response.png)
- [Semidiscrete interface convergence](artifacts/phase2_semidiscrete/semidiscrete_interface_convergence.png)
- [Fixed-grid constraint history](artifacts/qualification_summary/phase6_fixed_grid_constraint_history.png)
- [Fixed-grid regional orders](artifacts/qualification_summary/phase6_fixed_grid_regional_orders.png)
- [Terminal regional CSV](artifacts/perlmutter_phase6-fixed-brill-terminal-rhs-analysis/analysis/orders.csv)
- [Strict evidence inventory](EVIDENCE_MANIFEST.md)
- [Transfer decision](TRANSFER_SELECTION.md)
- [Backend matrix](BACKEND_MATRIX.md)

All copied Perlmutter manifests verify after rebasing only their absolute path prefix; see `artifacts/rebased_manifest_verification.log`.
