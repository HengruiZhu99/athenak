# Inner relative-lapse advection: negative Cartesian tangent screen

This source alone worsens the longer projected-continuous tangent screen on the C0 spatial-norm baseline. At t=2 the smooth gauge seed has 34% higher H/M, and the shell seed has 33--44% higher H/M/Z. The short independent canonical check passes, while longer propagation remains exploratory. No long native or long canonical run, production change, or stabilization adoption follows.

## Candidate and provenance

The shared helper adds only to the regular physical-P lapse RHS:

```
c = 1 - W_gauge
Delta alpha_t = -c sum_i [(beta^i-beta_ref^i)
                         + beta^i*(alpha-alpha_ref)/alpha_ref] d_i alpha_ref.
```

It returns zero for the nonphysical lapse flag. At c=1 the total regular advection becomes alpha*beta.grad(log(alpha/alpha_ref)); c=0 retains the original analytic-reference advection subtraction. The shift, physical-P source/pole, C0 geometry, kappa2=0, KO, mask, ghost continuation and algebraic projector are unchanged. No C1, damping profile, stronger Theta falloff or floor is combined with this candidate.

The immutable local gate index is 4ff923dfabbae02c509b1cfeecbe18aa173bdcd5640e6d562d1d04934c8a7e8c; receipt b85ec14d091969f798ea8d58287bfc8b9dd0262cc147db42a630b79e6d1e4721. All 40 indexed files, 10 successful commands and 376 unchanged current recorded inputs were independently verified before compiling. The exact helper SHA256 is 39f125347e050bbf662ce3dc1e354791b37cafa7949fdf1ff19cedda3d8d85e1. Its nonlinear/principal/full20 gates cover both the global span2.2 and native span2.1 frequency samples. Positive finite primitive roots remain unclassified; no negative-spectrum assumption is used here. The gate also preserves its first symbolic-rendering failure and successful corrected rerun. Finite positive-lapse admission is not a claim of uniform collapsed-lapse accuracy: additive cancellation in the relative-advection formula remains a limitation.

Fresh templates came from the preserved original C0 full22-v2/projected-v1 sources. The spatial-norm wrapper and math remain byte-pinned cc83d40e1ee3fc6da0baa6be31069b200d9f6ac1ec0ecca0f448208212ef64fe and c44d0f2b3940d94d77f9b6c920401d2b1eb7b20dbee67a160e873a5a604781ae. The copied CartesianPatch adds one include and one helper addition to assembled gauge_rhs.alpha before native upwind advection. Cached Point adds the identical scalar source. No new reference subtraction is added; the helper is exactly zero at the reference.

Root's independent native patch places the alpha addition before the geometry-only roundoff subtraction; this tangent copy places it after that block. The block never accesses gauge_rhs, so alpha/gauge/upwind/KO operation order is identical. The include locations also differ. Both private headers strip back to bitwise production, and the shared helper is byte-identical. The explicit source comparison is retained; the private patch headers themselves are not claimed byte-identical. Independent native time runs belong to root's separate receipt.

Build launch HEAD is 2392ccd1345430fe11786a7583793c2060d16f9c. Compiled production dependencies remain implementation 27c19d20696ea6dd4704032c51dfd026218f64f2 plus the explicit helper/overlay. Freeze HEAD is recorded separately after parent documentation commit 1f58f7b1. Two exact AppleClang21 arm64 oracle commands, dependency/static archive hashes, and field/reference callback builds are recorded and reverified. Source templates and prior frozen archives are unchanged.

## Actual native derivative and support gates

Parameters are N16, span2.2, h=.1375, 1640 active Cartesian ball nodes, 32800 free20/36080 raw22, S1, a=.5, geometry(.05,.95), gauge(.45,.85), kappa10, symmetric quadratic ghosts and native KO.1. Minimum Omega is .0026953124999994555; nominal pole.03 dt is 8.085937499998367e-5. This is one grid, not a refinement sequence. The reference outward crossing time is .7457643839234269, giving t2=2.68181217 crossings.

Prepare changes the reference by zero. Projected native reference RHS is 6.083573831355486e-12; H/M/Z are 3.239473509281292e-14/1.3305685154844717e-15/6.643108248740043e-16, exactly matching the C0 receipts. All 63576 donor references are strictly interior/nonrecursive, with constant-weight error 1.33e-15. Actual native22 RHS amplitude sweeps agree with the cached raw operator to worst best 7.25e-10. The actual final-only SSPRK3 derivative agrees with P_ref R3(dt J22)Lift to 1.91e-10 across tested seeds and dt/2,dt,2dt. All six implementation-consistency checks pass; they are not physical acceptance criteria. Validation costs 12.29s.

Both raw22 and projected20 sparse matrices have only the predicted local alpha-row change:

```
Delta J_(alpha,alpha) = -c*(beta_ref.grad alpha_ref)/alpha_ref
Delta J_(alpha,beta_i) = -c*d_i alpha_ref.
```

There are 3872 nonzero changed entries, with maximum magnitude 1.3742470. Actual coefficients match the independently emitted analytic reference coefficients within 7.06e-13 absolute. Every non-alpha row is numerically exactly C0. Every raw22/projected20 row at all 672 outer nodes r>=.85 is exactly C0. The N16 cell-centred grid has no r<=.05 nodes, so its exact-core statement relies on the independent frozen local gate. No derivative/principal or boundary extension row is altered. The gauge-seed action changes only alpha, with L2 .0513233. Since native H/M/Z in physical-P evolved variables have no alpha dependence, this alpha-only RHS addition does not remove the instantaneous discrete gauge-to-H/M/Z source; that is an inference from the exact support identity and constraint formulas.

## Short independent propagation and exploratory t2

Evolution here is exp(t J20), where J20=P_ref J22 Lift is the continuously projected semidiscrete generator. It is distinct from the exact finite native final-only RK3 map validated above. All algebraic-independent fields and unrestricted finite interior Theta are retained. No ARPACK, primitive eigenvalue or Ritz interpretation is used.

Two-pass adaptive Arnoldi m50/80 and independent canonical Taylor action agree at t=.025,.05 within 1.61126e-14 for both seeds; Taylor costs 68.05s. At t=.05 pulse H/M/Z ratios to C0 are 1.000535/1.003131/.993189, with component amplification 1.021824 versus 1.024124. Shell ratios are .999892/.999880/.999752. These small early changes do not establish stabilization.

The exploratory t2 Arnoldi run costs 72.95s and 7200 sparse matvecs. Its maximum local coarse/fine relative difference is 7.13e-11. There is no independent long canonical comparison; the negative screen does not justify that cost. Local empirical truncation checks are not rigorous nonnormal forward-error bounds. The values below are single-grid finite-window exploratory observations, not exact native t2 evolutions or an all-time instability proof.

| Seed at t2 | Candidate H/M/Z | Candidate/C0 H/M/Z | Component amplification (C0) |
| --- | --- | --- | --- |
| Gauge pulse |2.944674/1.630659/.365746|1.34297/1.33616/.98928|46.0072(35.7290)|
| Shell |.00334058/.00329872/.00137138|1.33388/1.44384/1.37682|.020503(.016994)|

The gauge Euclidean free20 amplification is 107.4667 versus C0 91.9671. Gauge final squared outer r>=.9 H/M/Z fractions are .03554/.50734/.61324; H peaks at r=.35724 and M/Z at r=.99865. Shell fractions are .02372/.70089/.95292. These localizations do not prove boundary causation. Native signed-constraint amplitude checks differ by at most 2.66e-9 relative on sampled propagated directions.

The component diagnostic integrates h^3 sqrt(gamma) times configuration{chi,g,alpha,beta} H1 and momentum{P,A,Lambda,Theta} L2 sums at S1; stored upper tensor components count once. It is component scaling, not invariant tensor energy, a symmetrizer or a proven bound. The shell's sampled maximum is 1 initially. Linear vectors do not supply positive/SPD nonlinear-state acceptance. H is the native physical Hamiltonian; M/Z use reference Penrose-inverse contractions and the unweighted active-cell RMS.

Seed arrays and normalization exactly match frozen C0: the smooth angular lapse/shift pulse and independent RNG690 radial-shell random vector, each normalized to unit initial Euclidean free20 L2. Generation sources and array hashes are retained. This linear screen is not the finite native pulse amplitude experiment.

## Frozen evidence and decision

The historical source-only hold is superseded by the verified gate authorization and build provenance. This bundle contains exact sources/diffs, scientific gate identity, native-stage sweeps, short canonical agreement, long empirical receipts, actual matrix attribution, native constraint/field diagnostics and exact commands. Executables and large CSR/state/seed/reference-coefficient arrays are metadata-only with sizes/shapes/hashes. Frozen collectors must not be rerun in archived paths.

The longer tangent result is substantially worse in H/M and shell constraints despite small early changes. This source alone is rejected for further expensive validation on the current evidence. That screening decision establishes no generator eigenvalue, uniform energy estimate, nonlinear scri closure, resolution acceptance, finite angular pulse acceptance, black-hole result or production adoption.
