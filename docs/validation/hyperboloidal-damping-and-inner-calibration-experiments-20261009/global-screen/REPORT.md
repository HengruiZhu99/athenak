# C0 prescribed damping profile: marginal global tangent screen

This separate scratch experiment changes only the existing C0 damping coefficient to

```
kappa1 = kappa_input / alpha
kappa2 = (2*S/(a*a)/kappa_input - 1) * (1 - Omega).
```

For S=1,a=.5,kappa_input10 this is .2*(Omega-1). It retains the production physical-P gauge or the same private spatial-norm gauge, all native spatial stencils, algebraic projection, spherical ghost plan and KO. There is no C1/covector addition, imposed Theta falloff, floor, runtime option or production edit. The finite-Omega short gate passes; exploratory t2 results are marginal and mixed. No stabilization is adopted.

## Scientific and implementation admission

The frozen local v2 gate index is c9180b5bbedb2a0069a54853768a96f8b0f69736c30a8bc9c277b77f11fc39ea, receipt d832017b2091f9f9e618fd9f4b6cfdbce0afcb19972ad43eb8217bd4681bdbaa. Before compiling, all32 small indexed files, all12 successful commands and all378 unchanged recorded inputs were independently checked. The copied shared damping_profile.hpp is byte-pinned64ba382f509e81fe347b96e915d3933ee7188c97fd6a054524b1aa187a891532. The older v1 prose/index remains preserved by that gate; this screen binds v2.

The local evidence includes actual nonlinear/principal/Einstein-sector checks and coefficient-aware constraint differentiation, including d kappa2. Its exact pole statements concern rationally reconstructed analytic pole matrices; positive finite primitive frozen roots and raw reference cancellation up to1.08e-10 remain recorded. Those local observations are finite-Omega exploratory admission, not a global/scri or finite-pulse theorem.

The private CartesianPatch has exactly one new include and four changed kappa2 ConformalRHS arguments: live/reference evolution and live/reference pole diagnostics. Cached Point uses the same prescribed analytic p.omega, reference S/a and input kappa. Existing C0 analytic-reference roundoff subtraction uses the same profile. There is no new source subtraction. The actual constraint operator is unchanged; native spatial differentiation samples the profile variation. Analytic d kappa2 is explicitly included in the independently derived subsidiary gate, not manually added to the evolution RHS.

Actual parameters match the previous C0 and C1 screens: N16,span2.2,h=.1375,1640 active Cartesian ball nodes,32800 free20/36080 raw22,S1,a.5,referencewide(.05,.95),gauge(.45,.85),kappa10,symmetric quadratic ghosts,nativeKO.1. Minimum Omega=.0026953124999994555 and nominal pole.03 dt=8.085937499998367e-5. There is no resolution sequence here. The reference outward crossing time is .7457643839234269, so t2 is2.68181217 crossings.

Prepare reference change is zero. Projected native reference RHS is6.083573831355486e-12, with H/M/Z3.239473509281292e-14/1.3305685154844717e-15/6.643108248740043e-16, matching C0. All63576 donor references remain strictly interior/nonrecursive; constant-weight error is1.33e-15. Raw CSR versus actual cached native stencil error is1.11e-15. Six-amplitude actual native22 RHS derivatives agree to worst best7.25e-10; actual final-only SSPRK3 derivatives versus P_ref R3(dt J22)Lift agree to1.84e-10 over tested seeds and dt/2,dt,2dt. Both gauges pass all six implementation consistency checks. These retrospective thresholds are not physical acceptance criteria. Native-stage validations cost12.03/12.52s.

## Exact matrix attribution

The actual projected matrix difference from C0 has only3280 local entries: the P and Theta rows in the Theta column at every active point. All unexpected entries are numerically exactly zero. The difference agrees with

```
Delta P_t = Delta Theta_t = -kappa_input*kappa2(Omega)*Theta/Omega
```

to6.77e-11 absolute, for a maximum coefficient740.0289855. This is a matrix support/formula identity, not an energy statement. Initial pure-gauge pulse action J20*v is exactly equal to C0 because its Theta perturbation is zero. The profile therefore does not remove the initial native discrete gauge-to-constraint source; it changes subsequent off-constraint evolution.

## Verified short propagation and exploratory t2

Propagation uses J20=P_ref J22 Lift, the continuously projected semidiscrete generator. It is distinct from the actual finite native final-only RK3 map P_ref R3(dt J22)Lift validated above. All algebraic-independent fields and unrestricted finite interior Theta are retained. No primitive eigenvalue or Ritz search is used.

Independent Taylor expm_multiply and two-pass adaptive Arnoldi m50/80 agree at t=.025,.05 within1.15048e-14 production and7.65745e-15 spatial norm, across both seeds. Taylor costs66.09/64.36s. At t=.05 the gauge-pulse H/M/Z ratios to C0 are.990736/.997388/.996523 production and.990562/.993968/.995889 spatial norm. Shell H is about7% lower, while M/Z are1--2% worse. Configuration-H1/momentum-L2 component amplification is essentially unchanged.

The t2 Arnoldi screen costs75.91/76.02s and7040 sparse matvecs per gauge. Its maximum local coarse/fine truncation differences are9.25e-11/8.98e-11. There is no independent long canonical comparison: the marginal result does not justify that cost. Local empirical Arnoldi checks are not rigorous nonnormal forward-error bounds. These are exploratory single-grid finite-window results, not exact native t2 evolutions or all-time failure/success claims.

| Gauge/seed at t2 | Profile H/M/Z | Profile/C0 H/M/Z | Profile component amplification (C0) |
| --- | --- | --- | --- |
| Production gauge pulse |1.672653/1.225100/.268220|.97221/1.00605/.98405|47.3266(47.0983)|
| Spatial norm gauge pulse |2.130484/1.193051/.360541|.97164/.97759/.97520|35.6915(35.7290)|
| Production shell |.00339946/.00396956/.00099355|.97701/1.02474/1.02523|.074973(.074887)|
| Spatial norm shell |.00243411/.00235280/.00100983|.97193/1.02982/1.01384|.016828(.016994)|

The component diagnostic integrates h^3 sqrt(gamma) times the configuration{chi,g,alpha,beta} H1 and momentum{P,A,Lambda,Theta} L2 sums at S1. Stored upper tensor components are counted once. It is component scaling, not invariant tensor energy, a symmetrizer or a proven bound. Both shell sampled maxima are1 initially. Finite fields formed by adding a linear vector are not audited as positive/SPD nonlinear states.

H is the native physical Hamiltonian constraint. M/Z are native conformal covectors contracted with the reference Penrose inverse; RMS is the unweighted active-cell mean. The signed constraint amplitude checks differ by at most4.85e-9 relative over sampled propagated directions. For the gauge pulse, final squared outer r>=.9 H/M/Z fractions are.05052/.36609/.62467 production and.02447/.43835/.76468 spatial norm. H/M peaks are at r=.35724; Z peaks at r=.99865. These localizations do not prove boundary causation.

## Reproducibility and limits

Exact seed arrays/normalization match the frozen controls: the smooth angular lapse/shift pulse and independent RNG690 radial-shell random vector, each normalized to unit initial Euclidean free20 L2. Seed-generation sources and array hashes are retained. No physical pulse-amplitude acceptance follows from these tangent vectors.

Build launch HEAD is2392ccd1345430fe11786a7583793c2060d16f9c; production compiled source dependencies remain byte-identical27c19d20696ea6dd4704032c51dfd026218f64f2 plus the explicit profile header/overlay. Freeze HEAD is separately recorded. All four exact AppleClang21 arm64 build commands, compiler dependency hashes and static archive hashes are preserved/reverified. The field-only native derivative callback has its own exact compiler dependency receipt. Original C0 scratch oracles and previous frozen archives remain unchanged.

The two historical SOURCE_*_HOLD files are superseded by gate-authorization.json and build-provenance.json. Sources/overlay diffs, exact matrix attribution, all native stage sweeps, short canonical controls, exploratory long receipts, diagnostics and commands are copied. Executables, CSR/state/seed/diagnostic arrays are metadata-only with sizes/shapes/hashes. Frozen collectors must not be rerun in archived paths.

The profile produces only marginal/mixed changes and does not supply useful stabilization in this screen. There is no global energy estimate, nonlinear scri closure, finite angular pulse acceptance, resolution acceptance, black-hole result or production adoption. Root's independently built actual native profile preflights are separate evidence and are not replaced by this projected continuous experiment.
