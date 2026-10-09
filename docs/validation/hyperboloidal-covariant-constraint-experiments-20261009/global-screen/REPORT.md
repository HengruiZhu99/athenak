# Covariant C1 Cartesian tangent: negative exploratory screen

This tests the mechanical Appendix-B C1 additions plus the separately derived covector Lambda correction on the actual 3D Cartesian full22/native20 infrastructure. Finite-Omega tensor/principal/local-pole gates passed first. The short independent numerical propagation gate passes, but the longer exploratory screen worsens M/Z and gives no basis for adoption or a costly long canonical/native pulse. It is not an all-time failure theorem. Production runtime remains27c19d20; launch HEADaef47b0a. Original global/stage/wave frozen archives are unchanged.

Parameters match the prior completed global control: N16,span2.2,h=.1375,1640 active nodes,32800 free20/36080 raw22,S1,a.5,wide(.05,.95),kappa10,symmetric quadratic ghosts,nativeKO.1. Both production and frozen private spatialnorm gauge branches are retained. The source addition changes P,Theta,A,Lambda only and uses live values/jets. The exact ignored CartesianPatch copy adds C1 after the existing C0 analytic-reference residual subtraction; C1 itself is never reference-subtracted. Cached Point adds the identical delta. No gauge changes, ghost/stencil changes, fields/floors/Theta falloff conditions, or production options were introduced. Strict interior/nonrecursive support is unchanged.

## Actual native implementation gate

Prepare changes the reference by0. Reference RHS max6.08357e-12 and H/M/Z3.23947e-14/1.33057e-15/6.64311e-16 match the old operator. All63576 donor references are strictly active; weight sum error1.33e-15. Raw CSR agrees with actual cached native LoadMeshJet/mixed/Lx/KO application to1.11e-15 relative. Six-amplitude nonlinear native22 RHS sweeps agree to worst best7.25e-10. The actual final-only nonlinear SSPRK3 derivative agrees with P_ref R3(dtJ22)Lift to worst best5.69e-10 over all tested seeds anddt/2,dt,2dt; nominaldt=.03minOmega=8.0859375e-5. All twelve stage consistency checks pass. These are implementation checks, not physical acceptance criteria.

Continuous propagation uses J20=P_ref J22Lift. It is distinct from the actual finite native final-only RK3 map. Independent raw Taylor expm_multiply versus m50/80 two-pass Arnoldi at t=.025,.05 agrees to maxrelative2.78e-14 for both seeds and both gauges. Each short Arnoldi run costs~5.9s; raw canonical controls cost105.78/93.47s. The matrix and all RHS/one-step/source checks use the actual native spherical ghost continuation; no scalar or primitive local Fourier substitute is used here.

The long t2 runs cost83.58/81.21s and have empirical local coarse/fine difference<=1e-10. Their all-time forward error is not independently certified: a full t2 Taylor comparison is intentionally pending, because the negative screen does not justify that cost. Native signed constraint derivative amplitude comparisons remain below1e-7. No generator eigensolve/Ritz search is used. All saved states and scalar receipts are finite. The long results below are explicitly exploratory.

## Matched comparison with frozen C0

The same original seed arrays are normalized to unit Euclidean free20 L2: native width.5 angular gauge pulse (.1 lapse/.02 shift before normalization) and controlled RNG690 shell random data with the original radial profile. Their generation source and exact vector hashes are retained. No new asymptotic condition is imposed by using these particular test vectors; raw white22 RHS and one-step gates separately include generic finite physicalTheta on the strict interior grid.

| Gauge/seed at t2 | C1 H/M/Z | C1/C0 H/M/Z | C1 cfgH1+momL2 amplification (C0) |
| --- | --- | --- | --- |
| Production gauge |1.648961/1.403795/.349237|.9584/1.1528/1.2813|44.1803(47.0983)|
| Spatialnorm gauge |2.335094/1.873042/.591101|1.0650/1.5348/1.5988|35.7733(35.7290)|
| Production shell |.00333776/.00388160/.00105359|.9593/1.0020/1.0872|.072925(.074887)|
| Spatialnorm shell |.00273790/.00262309/.00123051|1.0932/1.1481/1.2354|.017908(.016994)|

H is the native physical Hamiltonian; M/Z use reference Penrose inverse contractions and unweighted active-cell RMS. The stored-component units norm uses h^3sqrtgamma quadrature, configuration={chi,g,alpha,beta} H1 and momenta={P,A,Lambda,Theta} L2, with S1. It is not an invariant tensor energy, symmetrizer, or stability bound. Shell sampled maxima in this norm remain1 at initial time. At t=.05 the smooth M errors were already36.2%/40.6% larger, despite slightly lower H/Z; these negative short observations are independently Taylor-verified.

Final smooth squared r>=.9 fractions H/M/Z are.05778/.54038/.66572 production and.04798/.75784/.79728 spatialnorm. Peak H is in the bulk (r=.22802/.35724); peakM/Z near r=.99865. This locates the measured constraint error but does not isolate boundary causation. The local C1 gate separately retains large raw finiteTheta transients from the genuine Lambda double pole. Its Omega A/Omega Lambda similarity is analysis only, not imposed falloff or a runtime modification. No uniform unweighted propagator, regular nonlinear scri closure, finite-pulse stability, or accepted generator eigenvalues are claimed.

## Provenance and command correction

Exact source copies, original and patched CartesianPatch diffs, cached Point diff, source/math hashes, four AppleClang21 arm64 build/dependency inventories, static link archives, commands/logs and all stage/sweep/short/long observations are archived. Compiled production headers match27c19d20; the only geometry evolution change is the explicit C1 overlay. The frozen math is908d655ad8a43261d8e0cd66b3c0da485fc131b57a5203019aea01acc21771b7. Large CSR/vector/state arrays and executables are metadata-only with original paths, shapes, byte counts and SHA256.

A copied native20 build command initially retained the old working scratch output path. Before any candidate result used it, that scratch binary was restored from its frozen projected-v1 copy and the command corrected. Both original working oracle hashes again match565f428.../495647..., and the correction receipt preserves the exact erroneous command, unused output hash, restoration path/hash and corrected source. No production source/executable or frozen archive bytes changed. Historical source-preparation.json predates this correction; authoritative compiled-source identity is build-provenance.json plus source-identity-verification.json and the current source catalog. The failed scratch binary is retained metadata-only and was never used for an observation.

This is a rejected exploratory screen with a verified short numerical gate, not an independently verified long stability result. Any blended lower-order follow-up requires its own formula/principal/coefficient/Einstein gates and separate source/receipt; this archive stays frozen.
