# Bounded inner lapse, composed Hessians and damping-vector feasibility

The two bounded inner lapse variants and the matching composed-Hessian global control fail to improve the tested Cartesian pulse dynamics. A direct inertial damping-vector swap also fails the current source-stiffness and causal-admission requirements at the mathematical stage. None is promoted into production. These are finite-Ω experiments and a feasibility audit; they do not complete nonlinear scri regularization or the later Minkowski-reference wormhole-to-trumpet test.

## Bounded inner trace and direct combined lapse

Let c=1−Wgauge and define D=αbar(βref·dΩ)/αref−β·dΩ. On the physical-P branch the inner trace candidate adds

```
Δalpha_regular = 3c(αbar+2c) D/Ω.
```

The correction vanishes in the outer collar. For the tested gauge cutoff(.45,.85), its support has Ω≥.2775, so it introduces no new outer pole. In the exact Cauchy core dΩ=0. The combined variant directly replaces the complete regular lapse part by

```
c β·[dαbar−(αbar/αref)dαref]
 +W[β·dαbar−βref·dαref]−αbar ν log(αbar/αref)
 +Δalpha_regular.
```

It preserves the original physical-P pole and spatial-norm shift. At W=0 it recovers the conformal-Q gauge expression at finite Ω; it does not evolve a globally divided Q or alter the degenerate scri mass matrix. Near the reference, D uses differences; at collapsed lapse it uses direct contractions. This avoids the earlier additive relative-advection cancellation at αbar=1e−300. It is a pointwise arithmetic result, not puncture admission of native stencils or evolution.

The frozen local gate checks4004 nonlinear Release/ASan-UBSan states,112 crossbranch states down to1e−300, two360-case complete principal bases,3800 actual full20 matrices and16 unchanged leading pole matrices. Twelve commands pass with378 unchanged inputs. The strengthened nonzero-Ω-gradient principal fixture initially exposed geometric-background contamination in its extractor; both failed and corrected checks are preserved. Positive finite-Ω primitive roots remain, and the sampled worst reference root increases modestly.

Both native builds wrap `ResearchInnerTraceGauge` around the incoming spatial-norm `GaugeRHSParts` before `AssembleGaugeInterior`. The independent review reconstructs both complete header reversals, all12 private object/dependency recipes and both links. Geometry, physical-P pole, shift, geometric pole diagnostic, ghosts, upwinding, KO and final-only projection are unchanged.

| Native case | Trace-only | Combined |
| --- | --- | --- |
| Reference t=.05 | Exit0,16.6757s,drift1.30933e−14 | Exit0,17.0539s,drift1.30933e−14 |
| Pulse t=.02 | Exit0,7.13775s | Exit0,7.18411s |
| H/M/Z | .00311493443/.00478011264/.00117166850 | .00310668221/.00476482182/.00116875067 |
| Ratios to original spatial-norm | 1.001383/.976719/.978018 | .998730/.973595/.975583 |

All12 saved states pass all25 binary64 field, binary32-cast, initial equality, determinant/trace and positive spatial-metric checks. Native compact index is `6ec8049f02091e620bfa2bf95d79dc0e23490ef45169d3ea3dfd8c2d276cc706`; independent source-review index is `386e57e45b93f0600df111dd57e60b99c49deeeb286f0a23c8ccac44bbca30d3`.

Actual N16 full22/native final-step oracles and short independent canonical checks pass. Matrix differences contain exactly3872 predicted α←α/β value entries and no other entries; all672 outer points retain every C0 row exactly. The t2 exploratory global screen reverses the modest early native improvement:

| Gauge seed at t2 | H | M | Z | Component amplification |
| --- | ---: | ---: | ---: | ---: |
| Original spatial-norm | 2.19266 | 1.22040 | .369710 | 35.7290 |
| Trace-only | 30.9992 | 27.8604 | 9.31141 | 432.326 |
| Combined | 47.6719 | 47.5629 | 16.2349 | 674.384 |

Shell-seed constraints worsen as well. These are unit-Euclidean seeds and continuously projected J20 actions, not exact finite-step native SSPRK3 histories or a proved energy. No t6 extension or long native pulse is justified by these results.

## Matching composed Hessians in the global operator

The independently derived flat discrete Bianchi defect motivates replacing diagonal Dxx4 by Dx4Dx4 in both evolution and the Hamiltonian functional. The new control uses the original native composed bridge, radius4/ng4 continuation and the later matching-H diagnostic overlay. It exports actual stencil rows before one strict-interior donor substitution; it does not square an already halo-projected sparse derivative matrix. Mixed derivatives, upwinding, KO and M/Z/Θ diagnostics retain their original definitions.

Actual raw radius4 coefficients agree to3.55e−15; raw22 CSR/action to9.65e−16; nonlinear RHS Jv to5.44e−10; native final-only RK3 to3.80e−10; matching nonlinear H oracle to1.36e−11; short independent canonical action to1.19e−14. The ng4 physical coordinates map to ng3 within1.11e−16. All365 production src/root-CMake inputs remain unchanged. This is a fresh global control, with no new native evolution.

At t2, matching-H/original-H gauge ratios H/M/Z are .709480/1.835360/1.177815; shell ratios are .782877/1.212337/1.228819. Comparing both states with the common composed-H functional gives Hamiltonian ratios .904123/1.015637. M/Z rows are pointwise unchanged between diagnostics, so their increase reflects different evolved linear states. Component H1 ratios worsen1.08070/1.24420. Lower H partly reflects its functional; it does not establish improved dynamics. No t6 extension follows. The frozen index is `bad92d18a0a79b800253437f4ddf15a7d5d198ea07b3a55353647d962f11aac8`.

## Inertial damping-vector feasibility

The inspected [Gundlach et al. equation(2)](https://arxiv.org/pdf/gr-qc/0504114v2) permits a nonzero future timelike damping vector. The following projection is derived here; it is not a published regularization in this project's variables. With d=B n+V, J=V·Z, physical lapse A=αbar/Ω and rho=κ2,

```
Theta_t|damp=−A κ[(2+rho)B Theta−rho J],
Zi_t|damp=−A κ[B Zi+Vi Theta],
P_t|damp=+A κ[(1−rho)B Theta+(1+rho)J].
```

The full shear and connection projections are retained in the archived derivation. Exact 4D symbolic projections,1616 actual normal-specialization Release/ASan-UBSan checks and36 100-digit limits pass. No native candidate is built.

On the exact Minkowski height reference, d=∂T=∂t is parallel and future unit timelike, with B=αbar/Ω,V=β. Under the current κ=κinput/αbar normalization, its raw Θ damping and Λ damping scale as Ω⁻², while the Λ←Θ coupling scales as Ω⁻³. The damping value block has genuine Ω⁻² eigenvalue stiffness. Applying only the illustrative existing .03Ω cap gives scalar arguments about−83.56/−312.16/−368.41 on actual native N24/36/48 grids. This is a source-block mismatch, not an actual full20 or native instability test.

Moreover, ∂t need not remain timelike for arbitrary positive-lapse/SPD live fields satisfying the leading null relation. The explicit example αlive²=βref,n²−Ω² has physical norm+1. Normalizing that vector or imposing stronger Θ falloff is therefore not admitted. A normal-inner blend retains the outer causal and stiffness issues. No extra spacetime-Z variable or time mass matrix is algebraically required at finite Ω; regular variable weights and an appropriate source treatment remain unresolved. Constant parallel-vector mode/energy statements also do not transfer to current variable κ or the C0 subsidiary system. Frozen feasibility index is `7074d1a6210bbf6a9b2adcc13b5d048762edcfea9e9317ab10faf71ed9a62538`.

Source identities, actual inputs/commands, failed checks, implementation oracles and all compact results are preserved separately. Production remains implementation`27c19d20696ea6dd4704032c51dfd026218f64f2`. The next steps are identification of the growing discrete mode and derivation of constraint-compatible boundary Taylor conditions; neither follows automatically from these negative screens.

The [compact archive](validation/hyperboloidal-bounded-lapse-and-stencil-experiments-20261009/README.md) has379 cataloged blobs totaling9,451,154 bytes and119 finite JSON files. Catalog SHA256 is `8d75c6f35b484fc56232cfe1bba553e63d40d5df7908fb7d363a3d0bf538425b`. Original sources, build/launch HEADs and implementation identity are distinguished in every gate. The collector's initial list-versus-dictionary schema failure is explicit and affected no scientific data. Executable hashes are:

```
trace-only: d1593cfde3cafa88f81faad78c7aeaa23486217dc68a14d2f35f861bbf3f1235
combined: d5a2c6c5195858306357d1a1ecb68f2a60737e78ce08ed3227fcb1bebf2084bb
```

The [longer C0/live screens](hyperboloidal-long-window-and-resolution-audit.md) and [scri first-jet obstruction](hyperboloidal-scri-linear-hierarchy-audit.md) remain unchanged. The user-required later black-hole test must survive the inner wormhole-to-trumpet transition while retaining the Minkowski hyperboloidal reference throughout. None of these controls substitutes a black-hole reference or manufactures a fixed point by subtracting its evolution RHS.
