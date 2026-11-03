# Six-shell radial refinement of restored broad-smooth A

Authorized 2026-11-02 on branch `codex/restored-six-shells`: fresh N16 to10M on Aurora, with a quiet20-minute monitor. Preserve all previous runs. No new N24 run, no zero-restoration exploration, and no GPU smoke gate.

Scientific baseline is completed broad-smooth N16 job9033692 (NB1). Keep its restoration coefficient4, S8 collar compact r=.6–.9, C6 polynomial native initial slice P=I_z(7,7) over areal R=4M–12M, W .2–.56, KO8/originalscri, lapse restoration, auxiliaryD equation and all damping unchanged. Direct signed chi/lapse RK plus physical KO; M1/3, dt.0005, final coordinate10/3, lapsefloor1e-10, positivechi1e-8 inside r<.03 and finite nonpositivechi rescue everywhere. Keep pre-floor signs, angular2/scri1, horizon12/24/tol.001,diag.05. Never change timestep, floors or equations automatically.

Only intended experimental change: bisect each spherical layer [.16,.45],[.45,.75],[.75,1]. New outer radii [.305,.45,.6,.75,.875,1],16 cells per radial layer and16x16 angular cells per face. Cartesian core remains eight16-cubed blocks covering [-.18,.18]^3, continuous core treatment and inner overlap retained. There are44 volume blocks,96 primary radial cells, radial spacing .0090625/.009375/.0078125, unchanged core dx.01125. This changes radial sampling, adds three interfaces and refines the near-scri donor nodes; it is not uniform3D refinement or an isolated interface test.

Distinct `control_broad_smooth_six_shells` evolution/diagnostic targets use `HYPERBOLIC_SIX_SHELLS=1` and `HYP3D_MPI_N61` checkpoint magic. The168-byte scheme3 native layout binds six shell layers instead of three; source/binary hashes bind the exact radii. Old checkpoints cannot be loaded into this variant. Reports include actual block count, layer count and radial boundaries. A spherical-inner-radius constructor option validates spherical geometry without imposing the irrelevant cube-to-sphere first-radius restriction. Defaults preserve prior constructors and targets.

Mesh audit: actual scri is still the last layer; its donor-seven final-three-row closure and ghost spacing follow the refined outer layer. Original first4 operator remains selected everywhere; the outer-first6 selection radius does not affect this variant. Interpolation uses physical-radius chains across shell interfaces; the first-layer overlap and continuous Cartesian lookup follow the descriptor. Build/checkpoint reporting and admission must not retain hardcoded three-layer assumptions. Local production initialization and bounded evolution/communication checks are required for this new geometry; no repeated launcher tests or GPU audit gates.

Remote root /home/hzhu/hyperbolic-core-shell-20261028, SSH hzhu@aurora.alcf.anl.gov using /Users/hz0693/.ssh/codex-control/%C. Explicit /usr/bin/python3.11. One debug-scaling node12MPI ranks/GPUtiles,CompactBinaryMerger;worker3200/command3100/native2920. Build make-j64. Freeze separate source/build/launcher and actual binary hashes. Check remote state/saved submissions and all user debug/debug-scaling jobs before one submission. Never duplicate ambiguous/queued/running jobs or builds; preserve unrelated held capacity jobs.

Keep total home task below5GB including output and atomic reserves. Retire only obsolete outputs after fresh local hash/length verification and remote rehash with a receipt. Preserve all scientific snapshots and current restart until a successor is verified. No blanket purge or weakened guards.

At terminal collect and verify frozen N61/configuration/finitepayload/time/step/dt/progress/source/binary/input/artifact identities and scheduler/worker/native agreement. Healthy admitted walltime checkpoints below10M automatically continue same-case from the verified checkpoint/hash/horizon seed after matching terminal lock release. Numerical failures are terminal: preserve and localize, never replay. Remote verification may precede local download, but no purge before local preservation.

After disposition compare to three-shell broad-smooth NB1 and original A with distinct helpers/roots. Keep constraint histories, accepted mass gaps, scri null/pole/clock, sign/floor totals (boundary separate); reconstruct admitted endpoints with the matching diagnostic and common cut outside horizons. Plot radial/1D profiles around original interface .75 and new interfaces, measure peak height/width/integrated error and timestep/CFL/runtime. No fixed-cutoff historical claim from candidate cuts, no endpoint claim from failed states, and no fourth-order claim from this anisotropic two-grid comparison. Commit/push evidence, report outcome, then delete monitor after analysis and preservation.

## Local verification completed

Fresh N16 t0 has232928 evolved volume cells (180224 primary),44blocks,6layers and116294312-byte N61 checkpoint. Cartesian primary initial fields are byte-identical to the three-shell NB1 initial checkpoint. Exact t0 configuration and finitepayload verified. Two-step one-rank and12-rank SERIAL puncture checkpoints are byte-identical, SHA2566b6968333dbe3e01bdd39d78cca11ff754f2852454b6b969566ca1c9529cbe47. Initial/two-step bulk CFL .171262 versus .156138 for three-shell t0, same dt.0005; this is a measured local control, not a nonlinear stability bound. Null surface/ghost below9e-16 initially. See checks/local-verification.json and verify_local.py. No GPU smoke is added.

Remote obsolete E failed capture/checkpoint and completed NB1 checkpoint copies were locally reverified and remotely rehashed before retirement;318935040allocated bytes freed. All local originals and scientific snapshots retained, receipt retired-before-restored-six-shells.json.

At t0 with independent common cutoff r>.063, six-shell H/M/Mr RMS are .00122524737/.00127521885/.00128804296 versus three-shell .01433266963/.01923371422/.01948501616. This is an encouraging91.45%/93.37% H/M reduction under radial refinement, not evolved accuracy or a uniform fourth-order convergence claim. Diagnostic quadrature volumes differ slightly with radial resolution; both raw integrals/norms are preserved. See checks/initial-comparison.json.

## Aurora submission

Build completed with make-j64 and both evolution/diagnostic targets passed freeze. Frozen scientific source `812667b2e4124929cab1aae663ab6db5c480ce62`; manifest SHA256 `bb16e53c43e5012ff84e8e1031f8dd9f604e3d68f2c6e2afa977ae3e54f60661`; science SHA256 `72ab6ef1bfc3d096bc6ab36d37c578e0925268bb2746d021983cbd13a012b74a`. All frozen launcher files and manifest locally verified; compiler/build/provenance/science records in aurora-build/, mirrored launcher in output/restored-six-shells/aurora. No rebuild or GPU smoke.

Fresh N16 N61 job **9033869** submitted on debug-scaling,one node/12ranks. Plan `campaigns/n16-restored-six-shells-v1-a-evolution-segment01/plan.json`, SHA256 `feb93394476235b3d97178705c7a4deddbd0bd2266ae38ff02a9612dc5d0e749`. Both user debug queues were empty immediately before submission; unrelated held capacity jobs untouched. Storage at binding4,317,516,854bytes, planned196,070,900 plus atomic126,294,312 leaves360,117,934bytes headroom. No further retirement required. Follow state.json and saved submission; never duplicate.

The run and qualified analysis are complete. No continuation or automatic parameter change is scheduled. The20-minute monitor was deleted after evidence preservation, qualified analysis and push; final state and monitor-retired.json record its disposition.


## Terminal result: native completion, campaign acceptance failure

Job9033869 reached10M in6667steps, native return0, with a finite native-admitted N61 checkpoint. The worker and PBS nevertheless failed (exit1): the sole failed frozen check is `scri_penalty_scale_min == 1`, observed0.6510416666660155. The unchanged frozen campaign verifier rejected this result. This is not a numerical crash and not an accepted matched-parameter radial-refinement campaign. We do not restart it, weaken verification, or relabel its status.

All six worker artifacts were copied and locally hash/length verified. The native-admitted checkpoint is116294312bytes, SHA256 `b0b0f5767148d435c9ece954a17fb2746f0e59817bf38aeffb36d42c30e8119d`. The matching terminal allocation lock is released. `audit_completed.py` independently checks all identities, all original worker checks and the exact single policy failure before permitting qualified read-only diagnostics. It does not grant campaign acceptance. See evolution-disposition.json, evolution-local-verification.json and frozen-verification-rejection.json. No additional remote output was retired during this follow-up.

### Why the penalty check failed

The existing timestep-aware scri penalty uses c=12 dt/h and multiplies the nominal4/h and8/h penalties by min(1,0.5(1−1e−12)/c). Halving the outer spacing from0.015625 to0.0078125 at unchanged dt0.0005 raises c from0.384 to0.768. The cap therefore applies scale0.651041666666, reducing c to0.5. The effective4/h boundary-penalty coefficient is333.333 rather than the refined-grid nominal512 (three-shell value256). This limiter does not change the separate extra shift-restoration coefficient4 in the bulk collar; that restoration remains present.

This limiter was already active in the saved local initial-state report: preflight review missed that field while inspecting bulk CFL and communication parity. The campaign guard correctly detected the mismatch. It is not a late-time instability event. `penalty-limiter-analysis.json` records the exact source, initial report hash, formula and threshold. To avoid clipping on this grid requires dt below approximately0.000325520833333; dt0.00025 is a possible future controlled test, not an automatically authorized rerun. A clean spatial-attribution study would also check temporal sensitivity or compare both grids at a common smaller timestep.

### Qualified endpoint and time-series comparison

Matching original native constraint operators reproduce full-domain report norms to roundoff. The common coordinate-volume cutoff r>0.12 lies outside all final harmonic horizon surfaces: bounds0.1097764(A),0.1085408(NB1),0.1084931(N61). The small quadrature-volume difference is retained in radial-comparison.json; sensitivities at0.1099 and0.13 are also recorded.

| At10M | Original A | Broad/smooth,3 shells NB1 | Broad/smooth,6 shells N61* |
|---|---:|---:|---:|
| H RMS, r>0.12 |0.01533010|0.02271645|0.00903143|
| M RMS, r>0.12 |0.02897131|0.05755221|0.01788015|
| Accepted mass error |+0.34132%|+0.15798%|+0.13689%|
| Accepted horizon rows |45/68|37/68|37/68|
| Chi stage/final floors |56/8|0/0|0/0|
| Maximum pole residual |0.028168|0.052004|0.185217|
| Maximum clock residual |0.022001|0.038820|0.321548|
| Native command seconds |1331.90|1375.89|1633.43|

*Native completed, but penalty-policy check failed. The six-shell endpoint H/M RMS decreases60.24%/68.93% against NB1 and41.09%/38.28% against A. Native runtime increases18.72% against NB1 (27.22 versus22.93minutes on one12-tile node), not a factor of two. This is a single-run timing comparison, not a throughput benchmark. All reported lapse/surface/boundary sign/floor interventions are zero in NB1 and N61, with boundary interpolation counts kept separate. Surface/ghost null residuals remain below1.56e−15; that algebraic result does not imply small pole/clock residuals or convergence.

The histories show substantially smaller early exterior transients than NB1 and lower late-time exterior constraints, but N61 is not below original A at every time. Full-domain and own-candidate exterior histories, accepted mass gaps and running maximum boundary residuals are plotted separately. These binaries did not emit fixed-cutoff time series; those cannot be recovered from one terminal checkpoint and are not invented. Fixed-cutoff results here are endpoint reconstructions. See comparison/runtime-and-history-scope.json.

### Where the errors and sharp features remain

Against NB1, squared H/M integrals decrease89.54%/91.55% in0.7–0.775,92.18%/93.24% in0.775–0.9, and89.58%/91.65% over0.6–0.9. The inner0.16–0.30 H integral instead rises40.79%, while its M integral changes+0.13%. Against original A, the former outer0.775–0.9 concentration drops97.88%/96.11%, but the inward0.7–0.775 band still exceeds original A. This is substantial redistribution and reduction, not uniform improvement everywhere.

Unsmoothed near-face-center and near-corner rays use the same angular directions,48 versus96 primary radial samples; extra overlap cells remain in CSV but are not double plotted. At the near-center ray, the0.6–0.9 Hamiltonian peak decreases from0.079816(NB1) to0.034836(N61), and the momentum norm peak from0.232054 to0.086185. They remain near the old r=0.75 interface (refined maximum r=0.753906). The sampled contiguous absolute half-peak H width brackets shrink from[0.07031,0.10469] to[0.01641,0.03359]; momentum-width brackets overlap. These are grid-sample brackets without fitted/interpolated widths, not regularity norms.

Primary field profiles and most first gradients almost overlap between NB1 and N61. The auxiliary-D transition is still sharp: near-center max|d_r D_r| increases43.65 to53.87, with its maximum near0.74531. Better sampled gradients can increase while constraints decrease. The larger pole/clock residuals and retained narrow D feature preclude claiming that regularity is resolved. Two rays, one resolution change, added interfaces, changed donor nodes and an active penalty limiter cannot establish fourth-order convergence or isolate the cause of improvement.

### Evidence and reproduction

Figures in comparison/ were rendered and visually inspected: histories.png, full-and-boundary.png, radial.png, field-slices.png, gradient-slices.png and constraint-slices.png. Native slice volume integrals agree with production volume sums at2.27e−16 relative for N61. The checkpoint hashes are checked before and after all diagnostic reads; command, local diagnostic hash and compile definitions are preserved. A/NB1 previously verified slice artifacts are reused with source/hash records, not evolved again.

To reproduce into a fresh directory (with preserved raw campaigns and matching local diagnostic executables available), run from the repository root:

```sh
python3 docs/hyperbolic-shift-3d/reproduction/restored-six-shells-v1/audit_completed.py
python3 docs/hyperbolic-shift-3d/reproduction/restored-six-shells-v1/compare.py --plans campaigns/n16-restored-six-shells-v1-a-evolution-segment01/plan.json --output /tmp/six-shell-comparison-new
python3 docs/hyperbolic-shift-3d/reproduction/restored-six-shells-v1/reconstruct_endpoints.py --output /tmp/six-shell-comparison-new
python3 docs/hyperbolic-shift-3d/reproduction/restored-six-shells-v1/plot_radial.py --directory /tmp/six-shell-comparison-new
python3 docs/hyperbolic-shift-3d/reproduction/restored-six-shells-v1/plot_slices.py --directory /tmp/six-shell-comparison-new
python3 docs/hyperbolic-shift-3d/reproduction/restored-six-shells-v1/summarize.py
```

Reconstruction deliberately refuses to overwrite existing radial/slice evidence. There is no new long evolution in these commands. The next useful controlled experiment is to remove the timestep-induced penalty mismatch while retaining this formulation/grid, with explicit authorization for any changed timestep. The current evidence supports radial refinement as a promising direction, not a clean convergence result.
